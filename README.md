# Patch PRC crop and transformation experiments

This folder contains the code, frozen prompts, and saved scores for the five-layout experiment in **Patchwise PRC Watermarks under Crop and Resize**. It can regenerate the 100-image comparison or run the same protocol on a new image transformation. It includes the small set of original Gunn PRC and Stable Diffusion pipeline files used by the study. It contains **no Stable Diffusion weights, secret keys, or bulk images**.

## Fast check, without a GPU

From this folder, use Python 3.10 with NumPy, SciPy, and Pillow (or install `requirements-cpu.txt`):

```bash
python run.py reproduce
python -m unittest discover -s tests -v
```

`reproduce` recomputes detections, false positives, AUROC, and each frozen threshold from `data/reference_scores.json` and `data/reference_calibration.json`. It should print:

```text
 1 patches: 28/100 detected, 5/100 false positives, AUROC 0.7537
 4 patches:  7/100 detected, 0/100 false positives, AUROC 0.7011
16 patches: 73/100 detected, 1/100 false positives, AUROC 0.9679
32 patches: 73/100 detected, 1/100 false positives, AUROC 0.9848
64 patches: 74/100 detected, 1/100 false positives, AUROC 0.9799
```

This checks the saved analysis. Generating new images and recoveries requires an NVIDIA GPU.

## GPU setup

Use Linux, Python 3.10, a CUDA-compatible NVIDIA driver, and enough GPU memory for SD2.1 in FP32 with VAE decoder optimization. The completed study used four GPUs; the quick run needs one. Install PyTorch 2.1.0 and torchvision 0.16.0 with a wheel compatible with the host CUDA driver, then install the other packages from `requirements.txt`. For example, on a CUDA 12.1 host:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0
pip install -r requirements.txt
python run.py verify-prc
```

The `verify-prc` command embeds one small PRC on CPU and checks that this package's continuous score gives the same binary decision as Gunn's reference detector. The original code is in `third_party/gunn` with its MIT license and pinned upstream commit in `provenance/upstream.json`.

Download the pinned **Stable Diffusion 2.1 base** snapshot (roughly several GB) separately:

```bash
python scripts/download_model.py --output models/sd2.1-base
```

`provenance/sd_model.json` records the exact public mirror revision and file hashes. The script verifies the downloaded files. The historical run used this mirror because the original endpoint was unavailable. You can also point the `model` field of a configuration at an existing verified local checkpoint. Do not place a model checkpoint inside the shared archive.

## A small end-to-end run

```bash
python run.py setup --config configs/quick.json --output runs/quick
python run.py generate --config configs/quick.json --output runs/quick --device 0
python run.py recover --config configs/quick.json --output runs/quick --device 0
python run.py evaluate --config configs/quick.json --output runs/quick
```

This uses two test prompts, three calibration negatives, and the unpatched and 16-patch layouts. Its counts are only a pipeline check; three negatives are too few to estimate a 5% false-positive rate. Inspect `runs/quick/results/crop6/summary.json` and the saved attacked images/recovered latents. A completed sample is checksum sealed and skipped on a resumed run.

## Recreate the 100-image study

```bash
python run.py setup --config configs/reference.json --output runs/reference
python run.py generate --config configs/reference.json --output runs/reference --device 0
python run.py recover --config configs/reference.json --output runs/reference --device 0
python run.py evaluate --config configs/reference.json --output runs/reference
```

The five layouts are one PRC across the whole noise grid, then 4, 16, 32, and 64 independent PRCs. They use 1×1, 2×2, 4×4, 4×8, and 8×8 grids. Every marked image has 512 total message bits. The reference attack removes six pixels **from each border** and bilinearly resizes the 500×500 center to 512×512. There is no geometric alignment before or after recovery.

The runner uses 20 separate unmarked calibration images per layout and transformation. It freezes each threshold at the maximum calibration score, then tests 100 marked images per layout and 100 shared unmarked images. The image score is the **mean continuous patch PRC score**. No clean image, original seed, prompt, or attack parameters are supplied to the detector. The configuration specifies the evaluation transformation, and calibration controls are attacked in the same way. The historical result is not guaranteed to be bitwise identical across GPU/CUDA/library versions; the frozen saved scores above reproduce the exact published table.

To use four GPUs, set `num_devices` to 4 **before `setup`**, then launch `generate --device 0` through `--device 3` in separate processes, wait for all of them, and do the same for `recover`. Run `evaluate` once after all recovery processes finish. Worker indices deterministically shard the jobs; do not change `num_devices` midway through a run. Each GPU loads the model once. `runs/reference/image_cache` contains generated clean images, while `runs/reference/recoveries/<attack>` contains transformed images, VAE latents, and recovered noise arrays.

## Add or vary transformations

Create a new config by copying `configs/other_transforms.example.json`. Built-in choices are:

| Kind | Parameters | Meaning |
|---|---|---|
| `identity` | none | No image change |
| `crop_resize` | `pixels` | Crop that many pixels per border, then bilinearly resize to 512×512 |
| `translation` | `dx`, `dy` | Move visible content right/down for positive values; replicate edges |
| `jpeg` | `quality` | JPEG encode/decode |
| `gaussian_blur` | `radius` | Pillow Gaussian blur |

Every attack must have a unique `name`, used as its results directory. To add another transformation, edit `patch_prc/attacks.py:apply` and add a CPU check to `tests/test_cpu.py`. The function receives a 512×512 RGB PIL image and returns another one. Do not change an existing output directory's frozen config. Use a **new output directory** for new attacks, and supply `--cache OLD_RUN/image_cache` to both `generate` and `recover` to reuse the original clean generations when the model, prompt file, seed offset, and layouts match:

```bash
python run.py setup --config configs/other_transforms.example.json --output runs/new_attacks
python run.py generate --config configs/other_transforms.example.json --output runs/new_attacks --cache runs/reference/image_cache --device 0
python run.py recover --config configs/other_transforms.example.json --output runs/new_attacks --cache runs/reference/image_cache --device 0
python run.py evaluate --config configs/other_transforms.example.json --output runs/new_attacks
```

The example config uses a different `seed_offset`, so it intentionally generates new images. To reuse `runs/reference/image_cache`, set `seed_offset` to `0`, `model` to the same checkpoint path, and include the needed layouts. The cache checks model, prompt hash, and seed offset before reuse. New layouts generate only their missing images. Every attack has its **own calibrated threshold**; comparing to the crop result with a mismatched threshold would not be fair. The calibration rank guarantee requires calibration and future unmarked images to come from the same received-image distribution.

## Files

- `run.py`, `patch_prc/study.py`: one command-line runner for setup, generation, recovery, evaluation, and saved-score reproduction.
- `patch_prc/watermark.py`: spatial partitions, fixed-key PRC embedding, continuous score.
- `patch_prc/pipeline.py`: pinned SD2.1 generation and optimized VAE plus diffusion inversion.
- `patch_prc/attacks.py`: transformation registry.
- `patch_prc/storage.py`: immutable run config, checksums, and resumption.
- `configs/`: full, quick, and transformation examples.
- `data/prompts.json`: the fixed disjoint prompt splits.
- `data/reference_*.json`: the completed crop study's inspectable scores and results.
- `provenance/`: original source hashes, fixed model download metadata, historical protocol.

The saved data contain continuous scores, not private PRC keys. The package does not reconstruct original recovered noise arrays from the score records. Re-running the GPU pipeline creates new arrays in the output directory.
# watermarking-experiments
