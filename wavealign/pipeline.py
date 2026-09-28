"""1K -> 2K -> 4K WaveAlign inference pipeline."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxPipeline
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps

from .config import CascadeConfig, StageConfig
from .frequency import latent_frequency_mix
from .model import configure_transformer_runtime, load_baseline_transformer
from .noise import build_noise_hierarchy, build_paper_noise_hierarchy
from .schedules import (
    alpha_sigma_linear,
    alpha_sigma_cosine,
    fixed_entry_tail_grid,
    interpolate_logit_trajectory,
    parent_sigma_at_child_sigma,
    reference_raw_grid,
    scheduler_phase_lead,
)
from .tiling import TileScheduler, blend_window, global_image_ids, tile_image_ids
from .wavelet import common_ll_guidance, haar_ll, pack_latents, unpack_latents


class WaveAlignPipeline:
    """The frozen production path, without historical experimental branches."""

    latent_channels = 16
    vae_downsample = 8

    def __init__(self, pipe: FluxPipeline, config: CascadeConfig):
        config.validate()
        self.pipe = pipe
        self.config = config
        self.device = pipe._execution_device
        self.dtype = next(pipe.transformer.parameters()).dtype

    @classmethod
    def from_pretrained(cls, config: CascadeConfig) -> "WaveAlignPipeline":
        config.validate()
        dtype = torch.float16 if config.dtype == "fp16" else torch.bfloat16
        transformer = load_baseline_transformer(config.model_path, dtype)
        pipe = FluxPipeline.from_pretrained(config.model_path, transformer=None, torch_dtype=dtype)
        pipe.transformer = transformer
        pipe.scheduler.config.use_dynamic_shifting = False
        pipe.to("cuda")
        if max(stage.resolution for stage in config.stages) > 2048:
            pipe.enable_vae_tiling()
        return cls(pipe, config)

    def _encode_prompt(self, prompt: str, prompt_2: str | None = None):
        with torch.inference_mode():
            prompt_embeds, pooled, _ = self.pipe.encode_prompt(
                prompt=prompt,
                prompt_2=prompt_2,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=512,
                lora_scale=None,
            )
        return prompt_embeds.to(dtype=self.dtype), pooled.to(dtype=self.dtype)

    def _schedule(self, stage: StageConfig):
        scheduler = copy.deepcopy(self.pipe.scheduler)
        scheduler.config.use_dynamic_shifting = False
        scheduler._shift = float(stage.time_shift)
        scheduler.config.shift = float(stage.time_shift)
        raw = (
            reference_raw_grid(self.config.reference_steps)
            if stage.jump_tau is None
            else fixed_entry_tail_grid(
                stage.jump_tau, stage.time_shift, stage.steps, self.config.reference_steps
            )
        )
        timesteps, count = retrieve_timesteps(
            scheduler, stage.steps, self.device, sigmas=np.asarray(raw, dtype=np.float64)
        )
        if count != stage.steps:
            raise RuntimeError(f"scheduler returned {count} steps for {stage.resolution}")
        return scheduler, timesteps, scheduler.sigmas[:count].float()

    @staticmethod
    def _image_ids(
        grid_height: int,
        grid_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return global_image_ids(
            grid_height // 2,
            grid_width // 2,
            device=device,
            dtype=dtype,
        )

    def _transformer_forward(
        self,
        packed: torch.Tensor,
        timestep: torch.Tensor,
        image_ids: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        guidance: torch.Tensor | None,
        *,
        ntk_factor: float,
        proportional_attention: bool,
        rope_train_length: int | None = None,
    ) -> torch.Tensor:
        if packed.shape[0] != 1:
            raise ValueError("the clean baseline supports one image per process")
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=self.device, dtype=self.dtype)
        configure_transformer_runtime(
            self.pipe.transformer,
            ntk_factor=ntk_factor,
            rope_train_length=(
                self.config.rope_train_sequence_length
                if rope_train_length is None
                else rope_train_length
            ),
            proportional_attention=proportional_attention,
        )
        return self.pipe.transformer(
            hidden_states=packed,
            timestep=timestep.expand(packed.shape[0]) / 1000,
            guidance=guidance,
            pooled_projections=pooled,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids[0],
            return_dict=False,
        )[0]

    def _full_or_swin_forward(
        self,
        grid: torch.Tensor,
        timestep: torch.Tensor,
        step: int,
        level: int,
        stage: StageConfig,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> torch.Tensor:
        padded = level in self.config.swin_levels and step % 2 == 0
        model_grid = F.pad(grid, (1, 1, 1, 1)) if padded else grid
        packed = pack_latents(model_grid.to(self.dtype))
        ids = self._image_ids(
            model_grid.shape[-2],
            model_grid.shape[-1],
            packed.device,
            self.dtype,
        )
        velocity = self._transformer_forward(
            packed,
            timestep,
            ids,
            prompt_embeds,
            pooled,
            guidance,
            ntk_factor=stage.ntk_factor,
            proportional_attention=True,
        )
        velocity_grid = unpack_latents(velocity, model_grid.shape[-2], model_grid.shape[-1]).float()
        return velocity_grid[..., 1:-1, 1:-1] if padded else velocity_grid

    def _tile_forward(
        self,
        grid: torch.Tensor,
        timestep: torch.Tensor,
        step: int,
        stage: StageConfig,
        prompt_embeds: torch.Tensor,
        pooled: torch.Tensor,
        guidance: torch.Tensor | None,
        tile_scheduler: TileScheduler,
    ) -> torch.Tensor:
        padded = self.config.tile_swin and step % 2 == 0
        model_grid = F.pad(grid, (1, 1, 1, 1)) if padded else grid
        height, width = model_grid.shape[-2:]
        boxes = tile_scheduler.boxes(height, width, step)
        overlap = 1.0 - self.config.tile_latent_stride / self.config.tile_latent_size
        taper = min(0.5, max(0.05, overlap))
        accumulator = torch.zeros_like(model_grid, dtype=torch.float32)
        weights = torch.zeros(
            model_grid.shape[0], 1, height, width, device=grid.device, dtype=torch.float32
        )
        for box in boxes:
            tile = model_grid[
                ..., box.top : box.top + box.height, box.left : box.left + box.width
            ]
            packed = pack_latents(tile.to(self.dtype))
            ids = tile_image_ids(box, device=packed.device, dtype=self.dtype)
            velocity = self._transformer_forward(
                packed,
                timestep,
                ids,
                prompt_embeds,
                pooled,
                guidance,
                ntk_factor=stage.ntk_factor,
                proportional_attention=False,
                rope_train_length=0,
            )
            velocity_grid = unpack_latents(velocity, box.height, box.width).float()
            window = blend_window(
                box.height,
                box.width,
                alpha=taper,
                device=grid.device,
                dtype=torch.float32,
            )
            region = (..., slice(box.top, box.top + box.height), slice(box.left, box.left + box.width))
            accumulator[region] += velocity_grid * window
            weights[..., box.top : box.top + box.height, box.left : box.left + box.width] += window
        velocity = accumulator / weights.clamp_min(1e-6)
        return velocity[..., 1:-1, 1:-1] if padded else velocity

    def _pixel_lift(self, parent_clean: torch.Tensor, ratio: int = 2) -> torch.Tensor:
        scale = self.pipe.vae.config.scaling_factor
        shift = self.pipe.vae.config.shift_factor
        raw = (parent_clean / scale + shift).to(device=self.device, dtype=self.pipe.vae.dtype)
        with torch.inference_mode():
            image = self.pipe.vae.decode(raw, return_dict=False)[0].float().clamp(-1, 1)
            upsampled = F.interpolate(
                image, scale_factor=ratio, mode="bicubic", align_corners=False
            ).clamp(-1, 1)
            encoded = self.pipe.vae.encode(upsampled.to(self.pipe.vae.dtype)).latent_dist.mode()
        return ((encoded - shift) * scale).float()

    def _bridge_anchor(self, parent_clean: torch.Tensor) -> torch.Tensor:
        """Keep the historical parent_clean[-1] source; optionally refine RU."""
        pixel_lift = self._pixel_lift(parent_clean)
        if self.config.lfm_strength == 0:
            return pixel_lift
        return latent_frequency_mix(
            parent_clean, pixel_lift, strength=self.config.lfm_strength,
            scale_factor=self.config.lfm_scale_factor,
            filter_kind=self.config.lfm_filter,
        )

    def _decode(self, latent: torch.Tensor):
        scale = self.pipe.vae.config.scaling_factor
        shift = self.pipe.vae.config.shift_factor
        with torch.inference_mode():
            image = self.pipe.vae.decode(
                (latent / scale + shift).to(self.pipe.vae.dtype), return_dict=False
            )[0]
        return self.pipe.image_processor.postprocess(image, output_type="pil")[0]

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        *,
        prompt_2: str | None = None,
        seed: int | None = None,
        base_height: int | None = None,
        base_width: int | None = None,
        output_dir: Path | None = None,
        save_intermediate: bool = True,
        step_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        seed = self.config.seed if seed is None else int(seed)
        root_resolution = self.config.stages[0].resolution
        base_height = root_resolution if base_height is None else int(base_height)
        base_width = root_resolution if base_width is None else int(base_width)
        if base_height <= 0 or base_width <= 0:
            raise ValueError("base dimensions must be positive")
        if max(base_height, base_width) != root_resolution:
            raise ValueError(
                f"the root long side must equal {root_resolution}, got "
                f"{base_width}x{base_height}"
            )
        if base_height % 16 or base_width % 16:
            raise ValueError("base dimensions must be divisible by 16")
        prompt_embeds, pooled = self._encode_prompt(prompt, prompt_2)
        shapes = [
            (
                1,
                self.latent_channels,
                base_height * (2**level) // self.vae_downsample,
                base_width * (2**level) // self.vae_downsample,
            )
            for level, _stage in enumerate(self.config.stages)
        ]
        coarsest = None
        if self.config.initial_noise_path is not None:
            loaded = torch.load(self.config.initial_noise_path, map_location="cpu", weights_only=True)
            coarsest = loaded.get("noise", loaded.get("eps_1k")) if isinstance(loaded, dict) else loaded
        if self.config.noise_mode == "paper_old":
            noise = build_paper_noise_hierarchy(shapes, seed=seed, coarsest=coarsest)
        else:
            noise = build_noise_hierarchy(
                shapes,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                device="cpu",
                coarsest=coarsest,
            )
        noise_contract = []
        for level, value in enumerate(noise):
            residual = None
            if level:
                residual = float(
                    (haar_ll(value) - noise[level - 1]).norm()
                    / noise[level - 1].norm().clamp_min(1e-12)
                )
            noise_contract.append(
                {"level": level, "std": float(value.std()), "parent_ll_relative_l2": residual}
            )
        noise = [value.to(device=self.device, dtype=torch.float32) for value in noise]
        guidance = (
            torch.full((1,), self.config.guidance_scale, device=self.device, dtype=torch.float32)
            if self.pipe.transformer.config.guidance_embeds
            else None
        )
        parent_sigmas: list[float] = []
        parent_clean: list[torch.Tensor] = []
        final_latents: list[torch.Tensor] = []
        records: list[dict[str, Any]] = []
        tile_scheduler = TileScheduler(
            self.config.tile_latent_size,
            self.config.tile_latent_stride,
            seed=seed + self.config.tile_seed_offset,
        )

        for level, stage in enumerate(self.config.stages):
            scheduler, timesteps, sigmas = self._schedule(stage)
            if level == 0:
                grid = noise[level].clone()
            else:
                anchor = self._bridge_anchor(parent_clean[-1])
                grid = (1.0 - stage.jump_tau) * anchor + stage.jump_tau * noise[level]
            clean_trajectory: list[torch.Tensor] = []
            stage_sigmas: list[float] = []
            phase_lead = (
                scheduler_phase_lead(self.config.stages[level - 1].time_shift, stage.time_shift)
                if level
                else 0.0
            )
            for step, (timestep, sigma_tensor) in enumerate(zip(timesteps, sigmas)):
                sigma = float(sigma_tensor)
                # Observations own their CPU storage; callbacks cannot mutate the solver.
                observed_grid = grid.detach().to("cpu", copy=True) if step_observer else None
                tiled = level > 0 and step < stage.tile_steps
                if tiled:
                    velocity = self._tile_forward(
                        grid, timestep, step, stage, prompt_embeds, pooled, guidance, tile_scheduler
                    )
                else:
                    velocity = self._full_or_swin_forward(
                        grid, timestep, step, level, stage, prompt_embeds, pooled, guidance
                    )
                clean = grid - sigma * velocity
                observed_clean = clean.detach().to("cpu", copy=True) if step_observer else None
                alpha = 0.0
                parent_query_sigma = None
                if level:
                    alpha_fn = alpha_sigma_cosine if self.config.release_schedule == "cosine" else alpha_sigma_linear
                    alpha = alpha_fn(
                        sigma, stage.alpha_lock, stage.alpha_free, cap=1.0
                    )
                    parent_query_sigma = parent_sigma_at_child_sigma(sigma, phase_lead)
                    reference = interpolate_logit_trajectory(
                        parent_query_sigma, parent_sigmas, parent_clean
                    ).to(self.device)
                    clean = common_ll_guidance(clean, reference, alpha)
                    velocity = (grid - clean) / max(sigma, 1e-8)
                packed_grid = pack_latents(grid.to(self.dtype))
                packed_velocity = pack_latents(velocity.to(self.dtype))
                packed_grid = scheduler.step(
                    packed_velocity, timestep, packed_grid, return_dict=False
                )[0]
                grid = unpack_latents(packed_grid, grid.shape[-2], grid.shape[-1]).float()
                clean_trajectory.append(clean.detach().cpu())
                stage_sigmas.append(sigma)
                records.append(
                    {
                        "level": level,
                        "resolution": stage.resolution,
                        "step": step,
                        "sigma": sigma,
                        "alpha": alpha,
                        "forward": (
                            "tile_swin"
                            if tiled and self.config.tile_swin and step % 2 == 0
                            else "tile"
                            if tiled
                            else "swin"
                            if level in self.config.swin_levels and step % 2 == 0
                            else "full"
                        ),
                        "parent_sigma": parent_query_sigma,
                    }
                )
                if step_observer is not None:
                    step_observer({
                        **records[-1],
                        "sigma_next": float(scheduler.sigmas[step + 1]),
                        "state": observed_grid,
                        "clean_raw": observed_clean,
                        "clean_guided": clean.detach().to("cpu", copy=True),
                        "state_next": grid.detach().to("cpu", copy=True),
                    })
            parent_sigmas, parent_clean = stage_sigmas, clean_trajectory
            final_latents.append(grid.detach().cpu())
            if output_dir is not None and save_intermediate:
                output_dir.mkdir(parents=True, exist_ok=True)
                stage_image = self._decode(grid)
                stage_height = base_height * (2**level)
                stage_width = base_width * (2**level)
                stage_image.save(output_dir / f"I_{stage_width}x{stage_height}.png")
                stage_image.save(output_dir / f"I_L{level}_{stage_width}x{stage_height}.png")

        result = {
            "image": self._decode(final_latents[-1].to(self.device)),
            "latents": final_latents,
            "records": records,
        }
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            final_height = base_height * (2 ** (len(self.config.stages) - 1))
            final_width = base_width * (2 ** (len(self.config.stages) - 1))
            result["image"].save(output_dir / f"I_H_{final_width}x{final_height}.png")
            # Keep the historical 4K basename only for an actual 4K image.
            if (final_width, final_height) == (4096, 4096):
                result["image"].save(output_dir / "I_H_4096.png")
            metadata = {
                "prompt": prompt,
                "prompt_2": prompt_2,
                "seed": seed,
                "base_size": {"width": base_width, "height": base_height},
                "final_size": {"width": final_width, "height": final_height},
                "config": asdict(self.config),
                "noise_contract": noise_contract,
                "records": records,
            }
            metadata["config"]["model_path"] = str(metadata["config"]["model_path"])
            metadata["config"]["initial_noise_path"] = (
                str(metadata["config"]["initial_noise_path"])
                if metadata["config"]["initial_noise_path"] is not None
                else None
            )
            (output_dir / "run.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        return result
