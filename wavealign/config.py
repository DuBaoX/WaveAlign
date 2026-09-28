"""Typed configuration for the WaveAlign FLUX cascade."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .frequency import validate_lfm


@dataclass(frozen=True)
class StageConfig:
    resolution: int
    time_shift: float
    steps: int
    jump_tau: float | None = None
    alpha_lock: float | None = None
    alpha_free: float | None = None
    tile_steps: int = 0
    ntk_factor: float = 1.0

    def validate(self, *, is_root: bool) -> None:
        if self.resolution <= 0 or self.steps <= 0 or self.time_shift <= 0:
            raise ValueError(f"invalid stage: {self}")
        if is_root:
            if self.jump_tau is not None or self.tile_steps:
                raise ValueError("root stage must not jump or tile")
            return
        if self.jump_tau is None or not 0 < self.jump_tau < 1:
            raise ValueError("child jump_tau must be in (0, 1)")
        if self.alpha_lock is None or self.alpha_free is None:
            raise ValueError("child alpha thresholds are required")
        if not 0 <= self.alpha_free <= self.alpha_lock <= 1:
            raise ValueError("expected 0 <= alpha_free <= alpha_lock <= 1")
        if not 0 <= self.tile_steps <= self.steps:
            raise ValueError("tile_steps must be within the child tail")


@dataclass(frozen=True)
class CascadeConfig:
    model_path: Path = Path("models/FLUX.1-dev")
    stages: tuple[StageConfig, ...] = field(
        default_factory=lambda: (
            StageConfig(1024, 3.0, 30),
            StageConfig(2048, 6.0, 14, 0.90, 0.90, 0.70, 6, 10.0),
            StageConfig(4096, 12.0, 12, 0.90, 0.90, 0.70, 5, 10.0),
        )
    )
    reference_steps: int = 30
    guidance_scale: float = 3.5
    seed: int = 1
    dtype: Literal["fp16", "bf16"] = "fp16"
    tile_latent_size: int = 128
    tile_latent_stride: int = 96
    tile_seed_offset: int = 2
    tile_swin: bool = True
    rope_train_sequence_length: int = 64**2 + 512
    swin_levels: tuple[int, ...] = (1, 2)
    initial_noise_path: Path | None = None
    noise_mode: Literal["legacy", "paper_old"] = "paper_old"
    release_schedule: Literal["linear", "cosine"] = "linear"
    # Optional bridge-only experiment; zero preserves the existing baseline.
    lfm_strength: float = 0.0
    lfm_scale_factor: float = 0.25
    lfm_filter: Literal["resize", "haar1", "haar2", "haar_mid"] = "resize"

    def validate(self) -> None:
        if self.noise_mode not in ("legacy", "paper_old"):
            raise ValueError("noise_mode must be legacy or paper_old")
        if self.release_schedule not in ("linear", "cosine"):
            raise ValueError("release_schedule must be linear or cosine")
        validate_lfm(self.lfm_strength, self.lfm_scale_factor)
        if self.lfm_filter not in ("resize", "haar1", "haar2", "haar_mid"):
            raise ValueError("lfm_filter must be resize, haar1, haar2 or haar_mid")
        if self.lfm_filter == "haar1" and self.lfm_scale_factor != 0.5:
            raise ValueError("haar1 has a fixed 1/2 LL grid; set lfm_scale_factor=0.5")
        if self.lfm_filter in ("haar2", "haar_mid") and self.lfm_scale_factor != 0.25:
            raise ValueError("haar2 has a fixed 1/4 LL grid; keep lfm_scale_factor=0.25")
        if len(self.stages) < 2:
            raise ValueError("a cascade requires at least two stages")
        for index, stage in enumerate(self.stages):
            stage.validate(is_root=index == 0)
            if index and stage.resolution != 2 * self.stages[index - 1].resolution:
                raise ValueError("this implementation requires adjacent 2x stages")
        if self.reference_steps < max(stage.steps for stage in self.stages):
            raise ValueError("reference_steps must cover every tail grid")
        if not 0 < self.tile_latent_stride <= self.tile_latent_size:
            raise ValueError("invalid tile stride")
        if self.rope_train_sequence_length <= 0:
            raise ValueError("rope_train_sequence_length must be positive")

    @classmethod
    def baseline(cls) -> "CascadeConfig":
        cfg = cls()
        cfg.validate()
        return cfg
