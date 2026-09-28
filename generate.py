#!/usr/bin/env python3
"""WaveAlign inference on FLUX.1-dev with old noise and cosine release."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import uuid


def paper_config(model: Path, seed: int, resolution: int, release_schedule: str = "cosine"):
    from wavealign.config import CascadeConfig

    if resolution not in (2048, 4096):
        raise ValueError("resolution must be 2048 or 4096")
    if release_schedule not in ("linear", "cosine"):
        raise ValueError("release_schedule must be linear or cosine")
    if not model.is_dir() or not (model / "model_index.json").is_file():
        raise ValueError(f"expected a local Diffusers FLUX model directory: {model}")
    base = CascadeConfig.baseline()
    config = replace(
        base,
        model_path=model.resolve(),
        seed=seed,
        stages=base.stages[:2 if resolution == 2048 else 3],
        noise_mode="paper_old",
        release_schedule=release_schedule,
        lfm_strength=1.0,
        lfm_filter="haar1",
        lfm_scale_factor=0.5,
    )
    config.validate()
    return config


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def allocate_run(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    path = output_root / name
    path.mkdir(exist_ok=False)
    return path


class WaveAlignRunner:
    """One reusable FLUX process for CLI or a serialized local Gradio queue."""

    def __init__(self, model: Path):
        self.model = Path(model)
        self.pipeline = None

    def generate(
        self,
        prompt: str,
        *,
        seed: int = 42,
        resolution: int = 4096,
        output_root: Path = Path("output/wavealign"),
        lora: Path | None = None,
        lora_scale: float = 0.8,
        release_schedule: str = "cosine",
        sample_id: str | None = None,
    ) -> Path:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if seed < 0 or seed >= 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if not 0 <= lora_scale <= 2:
            raise ValueError("LoRA scale must be in [0, 2]")
        if lora is not None and (not lora.is_file() or lora.suffix != ".safetensors"):
            raise ValueError("LoRA must be an existing .safetensors file")
        config = paper_config(self.model, seed, resolution, release_schedule)
        import numpy as np
        import torch
        import diffusers
        import transformers
        from wavealign.pipeline import WaveAlignPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("WaveAlign requires a CUDA GPU; check the PyTorch CUDA installation and driver")
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        if self.pipeline is None:
            self.pipeline = WaveAlignPipeline.from_pretrained(config)
            self.pipeline.pipe.enable_vae_tiling()
        self.pipeline.config = config
        lora_digest = digest(lora) if lora is not None else None
        run_dir = allocate_run(Path(output_root))
        try:
            try:
                if lora is not None:
                    self.pipeline.pipe.load_lora_weights(str(lora), adapter_name="wavealign")
                    self.pipeline.pipe.set_adapters("wavealign", adapter_weights=lora_scale)
                result = self.pipeline(prompt.strip(), seed=seed, output_dir=run_dir)
            finally:
                if lora is not None:
                    try:
                        self.pipeline.pipe.unload_lora_weights()
                    except Exception as error:
                        self.pipeline = None
                        raise RuntimeError("LoRA cleanup failed; cached pipeline discarded") from error
            if lora is not None and digest(lora) != lora_digest:
                raise RuntimeError("LoRA file changed during generation")
            names = [f"I_{resolution}x{resolution}.png", "I_1024x1024.png"]
            if resolution == 4096:
                names.insert(1, "I_2048x2048.png")
            images = {name: digest(run_dir / name) for name in names}
            record = {
                "method": "WaveAlign",
                "protocol": f"old-noise-{release_schedule}-release",
                "prompt": prompt.strip(),
                "sample_id": sample_id,
                "seed": seed,
                "resolution": resolution,
                "model": str(config.model_path),
                "model_index_sha256": digest(config.model_path / "model_index.json"),
                "lora": {"file": str(lora), "sha256": lora_digest, "scale": lora_scale} if lora else None,
                "images": images,
                "steps": len(result["records"]),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "runtime": {
                    "torch": torch.__version__,
                    "diffusers": diffusers.__version__,
                    "transformers": transformers.__version__,
                },
            }
            temp_record = run_dir / "wavealign.json.tmp"
            temp_record.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            temp_record.replace(run_dir / "wavealign.json")
            return run_dir
        except Exception as error:
            (run_dir / "error.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="local FLUX.1-dev Diffusers directory")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--prompts-jsonl", type=Path, help="JSONL with prompt, optional id and seed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, choices=(2048, 4096), default=4096)
    parser.add_argument("--output-dir", type=Path, default=Path("output/wavealign"))
    parser.add_argument("--lora", type=Path, help="local Diffusers-compatible .safetensors LoRA")
    parser.add_argument("--lora-scale", type=float, default=0.8)
    parser.add_argument("--release-schedule", choices=("cosine", "linear"), default="cosine")
    args = parser.parse_args()
    runner = WaveAlignRunner(args.model)
    rows = [{"prompt": args.prompt, "seed": args.seed, "id": None}] if args.prompt is not None else read_prompt_rows(args.prompts_jsonl, args.seed)
    for row in rows:
        path = runner.generate(
            row["prompt"], seed=row["seed"], resolution=args.resolution,
            output_root=args.output_dir, lora=args.lora, lora_scale=args.lora_scale,
            release_schedule=args.release_schedule, sample_id=row["id"],
        )
        print(path.resolve(), flush=True)


def read_prompt_rows(path: Path, default_seed: int) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"line {number} needs a nonempty prompt")
        identifier = str(row.get("id", f"prompt_{number:04d}"))
        if identifier in seen:
            raise ValueError(f"duplicate prompt id: {identifier}")
        seen.add(identifier)
        seed = int(row.get("seed", default_seed))
        if seed < 0 or seed >= 2**63:
            raise ValueError(f"line {number} seed must be in [0, 2**63)")
        rows.append({"id": identifier, "prompt": row["prompt"], "seed": seed})
    if not rows:
        raise ValueError(f"no prompts in {path}")
    return rows


if __name__ == "__main__":
    main()
