# WaveAlign

Reference FLUX.1-dev implementation of **WaveAlign: Wavelet-Based Alignment for Training-Free High-Resolution Image Generation**. It generates a 1K parent, then 2K and optionally 4K children with a frozen model.

The public entry points are `generate.py` for command-line generation and `app.py` for a local Gradio demo. The Python package is `wavealign`.

## Examples

The [30-image LoRA gallery](examples/README.md) contains 20 selections from the original 50-image review and 10 additional architecture, library, and aerial subjects. Each image is a web-size preview; [the manifest](examples/manifest.json) records its prompt, seed, LoRA identifier and scale, source-image hash, and preview hash. These historical examples used the original linear release, while the public demo defaults to cosine. The source 4K PNGs and third-party LoRA weights are not redistributed.

## Method and release setting

WaveAlign couples stage noise in an orthonormal Haar hierarchy, mixes the first-level latent low frequencies with a pixel-upsampled anchor, queries a completed parent clean-prediction cache at the phase given by scale-time conjugacy, and applies progressively released LL2 structural guidance. The three stages use 30/14/12 model steps and shifts 3/6/12. The implementation uses the original FLUX attention path and performs no training.

**The default noise is the old CPU FP32 coupled-noise protocol. The default guidance release is cosine**, as requested for this public demo: `0.5 × (1 − cos(π × clip((σ − 0.7) / 0.2, 0, 1)))`. Use `--release-schedule linear` to reproduce the original paper baseline's linear release. The paper's reported baseline metrics belong to that linear setting; the cosine default is a later variant and has separate outputs. Every run records the selected schedule in `wavealign.json` and the full sampling config in `run.json`.

## Setup

Python 3.10+, a CUDA GPU, and a local Diffusers-format FLUX.1-dev checkpoint are required. A 4K run requires substantial GPU memory. Create a fresh environment, [install PyTorch for your CUDA version](https://docs.pytorch.org/get-started/locally/), then install WaveAlign:

```bash
python -m venv .venv
source .venv/bin/activate
# Install the CUDA-enabled PyTorch wheel appropriate for this machine first.
python -m pip install -e '.[demo]'
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Accept the [FLUX.1-dev model license](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md), then download the gated checkpoint separately. With the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli), one option is:

```bash
hf auth login
hf download black-forest-labs/FLUX.1-dev --local-dir models/FLUX.1-dev
```

The model directory must contain `model_index.json` and the usual Diffusers FLUX components. Model weights and LoRA weights are not part of this repository. The [Diffusers FLUX documentation](https://huggingface.co/docs/diffusers/api/pipelines/flux) describes the checkpoint layout. The CLI checks for the directory and CUDA before loading weights.

## Command line

```bash
wavealign-generate \
  --model models/FLUX.1-dev \
  --prompt 'An African jacana standing among water lilies' \
  --seed 42 --resolution 4096 \
  --output-dir output/wavealign
```

For the paper's original linear release, add `--release-schedule linear`. For a local [Diffusers-compatible FLUX LoRA](https://huggingface.co/docs/diffusers/api/loaders/lora), add `--lora /path/to/adapter.safetensors --lora-scale 0.8`. LoRA changes the sampling setup and should be reported separately from the no-LoRA paper benchmark. The CLI creates a unique run directory and never overwrites an earlier run.

For a batch, use `--prompts-jsonl examples/prompts.jsonl` in place of `--prompt`. Each line should be a JSON object with `prompt` and optional `id` and `seed`; the CLI validates the file before generating, reuses the loaded model, and records each ID in its run metadata. This produces images only; benchmark metrics and the private evaluation dataset are not bundled.

## Gradio

```bash
wavealign-demo --model models/FLUX.1-dev \
  --lora-dir /path/to/local/loras \
  --output-dir output/wavealign --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860` locally or forward the port over SSH. The single-method interface supports 2K/4K, optional local LoRA, cosine or linear release, and a history view of completed runs. Generation is serialized on one GPU; the browser reads only runs with a completed `wavealign.json` and verifies image hashes. It does not depend on other method repositories, experiment jobs, a fixed LoRA catalog, or shared checkpoints.

## Outputs and reproducibility

Each run has stage PNGs, `run.json` with per-step `sigma` and `alpha`, and `wavealign.json` with prompt, seed, method protocol, model-index hash, dependency versions, LoRA hash, and image hashes. A failed run keeps `error.txt` and does not appear in the completed-run browser. Keep the same model revision, dependency versions, seed, release schedule, and LoRA setting for comparisons. A 2K run is the first two stages of the same cascade.

The original private experiments, evaluation data, and cluster submission scripts remain outside the public export. `python tools/export_public.py /path/to/new/repo` writes an allowlisted, standalone release tree with the curated example previews, without weights, full-resolution experiment outputs, benchmark artifacts, or cluster scripts.

## Development checks

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

GitHub Actions runs model-free CPU checks on pushes and pull requests. A real 2K/4K run and LoRA loading require a CUDA machine with the checkpoint and are not covered by CPU CI.

The WaveAlign source code is licensed under [Apache-2.0](LICENSE). FLUX.1-dev and any third-party LoRA retain their own licenses. The demo uses the public [Diffusers LoRA loading API](https://huggingface.co/docs/diffusers/api/loaders/lora) and [Gradio Blocks API](https://www.gradio.app/docs/gradio/blocks).
