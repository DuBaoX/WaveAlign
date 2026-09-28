"""Public inference protocol and browser checks; no checkpoint required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from app import list_runs, view_run
from wavealign.noise import build_paper_noise_hierarchy
from wavealign.schedules import alpha_sigma_cosine, alpha_sigma_linear
from wavealign.wavelet import haar_ll, replace_ll
from generate import WaveAlignRunner, digest, paper_config, read_prompt_rows


def test_old_noise_matches_root_draw_and_couples_each_stage():
    shapes = [(1, 2, 4, 4), (1, 2, 8, 8), (1, 2, 16, 16)]
    noise = build_paper_noise_hierarchy(shapes, seed=42)
    expected_root = torch.randn(shapes[0], generator=torch.Generator().manual_seed(42))
    assert torch.equal(noise[0], expected_root)
    assert torch.equal(noise[0], build_paper_noise_hierarchy(shapes[:1], seed=42)[0])
    assert torch.equal(noise[1], build_paper_noise_hierarchy(shapes[:2], seed=42)[1])
    detail_seeds = (6914601942797029815, 3457824733418885122)
    for level, detail_seed in enumerate(detail_seeds, 1):
        detail = torch.randn(shapes[level], generator=torch.Generator().manual_seed(detail_seed))
        assert torch.equal(noise[level], replace_ll(detail, noise[level - 1]))
    for child, parent in zip(noise[1:], noise[:-1]):
        torch.testing.assert_close(haar_ll(child), parent, rtol=0, atol=1e-6)
    assert not torch.equal(noise[1][..., ::2, ::2], noise[0])


def test_cosine_is_default_and_linear_remains_explicit(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}")
    cosine = paper_config(model, 42, 4096)
    linear = paper_config(model, 42, 4096, "linear")
    from wavealign.config import CascadeConfig
    assert CascadeConfig.baseline().noise_mode == "paper_old"
    assert cosine.release_schedule == "cosine" and linear.release_schedule == "linear"
    assert cosine.noise_mode == "paper_old" and cosine.lfm_filter == "haar1"
    assert cosine.stages == linear.stages
    assert alpha_sigma_cosine(.9, .9, .7) == 1
    assert alpha_sigma_cosine(.7, .9, .7) == 0
    assert alpha_sigma_cosine(.8, .9, .7) == pytest.approx(.5)
    assert alpha_sigma_cosine(.85, .9, .7) > alpha_sigma_linear(.85, .9, .7)
    with pytest.raises(ValueError):
        alpha_sigma_cosine(float("nan"), .9, .7)


def test_history_only_accepts_completed_intact_runs(tmp_path: Path):
    complete = tmp_path / "complete"
    complete.mkdir()
    image = complete / "I_1024x1024.png"
    image.write_bytes(b"sample image bytes")
    (complete / "wavealign.json").write_text(json.dumps({"images": {image.name: digest(image)}}))
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    assert list_runs(tmp_path) == ["complete"]
    paths, _ = view_run(tmp_path, "complete")
    assert paths == [str(image.resolve())]
    assert view_run(tmp_path, "../complete")[0] == []
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        view_run(tmp_path, "complete")


def test_batch_manifest_preserves_ids_and_seeds(tmp_path: Path):
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"id":"image-1","prompt":"A bird","seed":7}\n'
                       '{"prompt":"A fox"}\n', encoding="utf-8")
    assert read_prompt_rows(prompts, 42) == [
        {"id": "image-1", "prompt": "A bird", "seed": 7},
        {"id": "prompt_0002", "prompt": "A fox", "seed": 42},
    ]
    prompts.write_text('{"id":"x","prompt":"one"}\n{"id":"x","prompt":"two"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        read_prompt_rows(prompts, 42)
    prompts.write_text('{"prompt":"one","seed":-1}\n')
    with pytest.raises(ValueError, match="seed"):
        read_prompt_rows(prompts, 42)


def test_lora_is_unloaded_before_reusing_pipeline(tmp_path: Path, monkeypatch):
    from wavealign import pipeline as pipeline_module

    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}")
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"test adapter")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class FakePipe:
        def __init__(self):
            self.adapter = None
            self.tiling = False
            self.unloads = 0

        def enable_vae_tiling(self):
            self.tiling = True

        def load_lora_weights(self, path, adapter_name):
            assert path == str(lora) and adapter_name == "wavealign"
            self.adapter = adapter_name

        def set_adapters(self, name, adapter_weights):
            assert name == self.adapter and adapter_weights == 0.8

        def unload_lora_weights(self):
            self.adapter = None
            self.unloads += 1

    class FakePipeline:
        def __init__(self, config):
            self.config = config
            self.pipe = FakePipe()
            self.seen = []

        @classmethod
        def from_pretrained(cls, config):
            return cls(config)

        def __call__(self, prompt, *, seed, output_dir):
            self.seen.append(self.pipe.adapter)
            for size in (1024, 2048):
                (output_dir / f"I_{size}x{size}.png").write_bytes(f"{prompt}:{size}:{seed}".encode())
            return {"records": [1, 2]}

    monkeypatch.setattr(pipeline_module, "WaveAlignPipeline", FakePipeline)
    runner = WaveAlignRunner(model)
    first = runner.generate("An image", resolution=2048, lora=lora, output_root=tmp_path / "runs")
    second = runner.generate("Another image", resolution=2048, output_root=tmp_path / "runs")
    assert runner.pipeline.seen == ["wavealign", None]
    assert runner.pipeline.pipe.unloads == 1
    assert runner.pipeline.pipe.tiling
    assert json.loads((first / "wavealign.json").read_text())["lora"]["sha256"] == digest(lora)
    assert json.loads((second / "wavealign.json").read_text())["lora"] is None


def test_failed_lora_cleanup_discards_cached_pipeline(tmp_path: Path, monkeypatch):
    from wavealign import pipeline as pipeline_module

    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text("{}")
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"test adapter")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class BrokenPipeline:
        def __init__(self, config):
            self.config = config
            self.pipe = self

        @classmethod
        def from_pretrained(cls, config):
            return cls(config)

        def enable_vae_tiling(self):
            pass

        def load_lora_weights(self, *args, **kwargs):
            pass

        def set_adapters(self, *args, **kwargs):
            pass

        def unload_lora_weights(self):
            raise RuntimeError("unload failed")

        def __call__(self, prompt, *, seed, output_dir):
            return {"records": []}

    monkeypatch.setattr(pipeline_module, "WaveAlignPipeline", BrokenPipeline)
    runner = WaveAlignRunner(model)
    with pytest.raises(RuntimeError, match="cached pipeline discarded"):
        runner.generate("An image", resolution=2048, lora=lora, output_root=tmp_path / "runs")
    assert runner.pipeline is None
    folder, = (tmp_path / "runs").iterdir()
    assert (folder / "error.txt").is_file()
    assert not (folder / "wavealign.json").exists()
