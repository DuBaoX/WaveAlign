#!/usr/bin/env python3
"""Local WaveAlign generation and result browser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate import WaveAlignRunner, digest


def list_runs(output_root: Path) -> list[str]:
    if not output_root.is_dir():
        return []
    root = output_root.resolve()
    return [p.name for p in sorted(output_root.iterdir(), reverse=True)
            if p.is_dir() and p.resolve().parent == root and (p / "wavealign.json").is_file()]


def view_run(output_root: Path, name: str) -> tuple[list[str], str]:
    if not name or name not in list_runs(output_root):
        return [], "Select a completed run."
    folder = output_root / name
    record = json.loads((folder / "wavealign.json").read_text(encoding="utf-8"))
    images = []
    for filename, expected in record["images"].items():
        path = folder / filename
        if path.name != filename or path.resolve().parent != folder.resolve() or not path.is_file() or digest(path) != expected:
            raise ValueError(f"image missing or changed: {filename}")
        images.append(str(path.resolve()))
    return images, json.dumps(record, indent=2, ensure_ascii=False)


def build_app(model: Path, output_root: Path, lora_dir: Path | None = None):
    import gradio as gr

    runner = WaveAlignRunner(model)
    adapters = sorted(lora_dir.glob("*.safetensors")) if lora_dir and lora_dir.is_dir() else []
    choices = ["None"] + [item.name for item in adapters]
    by_name = {item.name: item for item in adapters}

    def generate(prompt: str, seed: float, resolution: str, adapter: str, scale: float, release: str):
        try:
            folder = runner.generate(
                prompt,
                seed=int(seed),
                resolution=int(resolution),
                output_root=output_root,
                lora=by_name.get(adapter),
                lora_scale=float(scale),
                release_schedule=release,
            )
            images, details = view_run(output_root, folder.name)
            return images, details
        except Exception as error:
            raise gr.Error(f"{type(error).__name__}: {error}") from error

    def refresh():
        return gr.update(choices=list_runs(output_root), value=None)

    def show(name: str):
        try:
            return view_run(output_root, name)
        except Exception as error:
            raise gr.Error(f"{type(error).__name__}: {error}") from error

    with gr.Blocks(title="WaveAlign · FLUX.1-dev") as demo:
        gr.Markdown("# WaveAlign\nFLUX.1-dev 1K → 2K → 4K generation. Default: old coupled noise and cosine guidance release.")
        with gr.Tab("Generate"):
            prompt = gr.Textbox(label="Prompt", lines=3)
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                resolution = gr.Radio(["2048", "4096"], value="4096", label="Final resolution")
            with gr.Row():
                adapter = gr.Dropdown(choices, value="None", label="Local LoRA")
                scale = gr.Slider(0, 2, value=0.8, step=0.05, label="LoRA scale")
            release = gr.Radio(["cosine", "linear"], value="cosine", label="Guidance release (linear reproduces the paper baseline)")
            run_button = gr.Button("Generate", variant="primary")
            gallery = gr.Gallery(label="Generated stages", columns=3)
            details = gr.Textbox(label="Run record", lines=12)
            run_button.click(generate, [prompt, seed, resolution, adapter, scale, release], [gallery, details])
        with gr.Tab("History"):
            refresh_button = gr.Button("Refresh")
            history = gr.Dropdown(choices=list_runs(output_root), label="Completed runs")
            previous_gallery = gr.Gallery(label="Original PNGs", columns=3)
            previous_details = gr.Textbox(label="Run record", lines=12)
            refresh_button.click(refresh, outputs=history, queue=False)
            history.change(show, inputs=history, outputs=[previous_gallery, previous_details], queue=False)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/wavealign"))
    parser.add_argument("--lora-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not (args.model / "model_index.json").is_file():
        parser.error("--model must point to a local Diffusers FLUX directory")
    app = build_app(args.model, args.output_dir, args.lora_dir)
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
