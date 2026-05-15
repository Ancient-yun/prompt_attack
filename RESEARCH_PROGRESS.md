# Research Progress: Learnable Soft Prompt Natural Adversarial Attack

Updated: 2026-05-15 KST

## 1. Research Direction

이 연구의 목표는 사람이 직접 작성한 prompt로 adversarial image를 찾는 것이 아니라, text-to-image 또는 image-editing 생성 모델의 continuous conditioning space를 learnable soft token으로 탐색해서 vision classifier의 natural failure case를 자동으로 찾는 것이다.

핵심 아이디어는 다음과 같다.

```text
original image + learnable soft prompt tokens + hard class label
-> image-editing generator
-> natural adversarial candidate
-> victim classifier loss로 soft token optimization
```

현재 구조에서는 hard class label을 prompt 끝에 유지한다.

```text
[V1] [V2] ... [V8] a photo of {class_label}
```

여기서 `[V1] ... [V8]`은 사람이 읽을 수 있는 discrete token이 아니라, FLUX prompt embedding space에서 학습되는 continuous soft token이다.

## 2. Motivation

기존 natural adversarial attack의 필요성은 "자연스러운 공격 이미지가 중요하다"는 점에 있다. 하지만 이 연구의 차별점은 자연스러운 adversarial image 자체가 아니라, 사람이 직접 상상하거나 prompt engineering하지 않아도 모델의 숨은 failure condition을 자동으로 찾는다는 점이다.

사람이 만든 prompt는 discrete하고 사람이 말로 표현할 수 있는 조건에 제한된다.

예시:

```text
snowy background
dark lighting
side view
occlusion
unusual texture
```

반면 learnable soft token은 continuous embedding space를 직접 최적화한다. 따라서 사람이 이름 붙이기 어려운 복합 조건도 탐색할 수 있다.

예시:

```text
subtle texture shift
+ weak class-relevant feature
+ generator-specific geometry change
+ background statistics
+ ambiguous viewpoint
```

세상에서의 필요성은 다음과 같이 정리할 수 있다.

- 생성형 AI가 synthetic data 생산 도구로 사용되고 있으므로, 생성 모델을 단순 data generator가 아니라 model vulnerability explorer로 사용할 수 있다.
- 배포 전 vision model red teaming과 QA에 사용할 수 있다.
- 실패 이미지가 자연스러운 image artifact로 남기 때문에 hard dataset, regression test, adversarial training data로 축적할 수 있다.
- 사람이 prompt로 표현하지 못하는 OOD 방향을 자동으로 찾는 도구가 될 수 있다.

## 3. Current Project Setup

Project path:

```text
D:\code\promtp_attack
```

Dataset path:

```text
D:\code\promtp_attack\dataset
```

Docker setup:

```text
docker/
  Dockerfile
  docker-compose.yml
  .env
```

Main runtime:

```text
python scripts/run_attack.py --config <config.yaml>
```

Current major dependencies:

- PyTorch / torchvision
- diffusers
- transformers
- FLUX.2 Klein pipeline
- DINOv2
- pyiqa
- W&B
- vendored pytorch-fid implementation

Vendored external code:

```text
external/pytorch_fid/
```

The external FID code was copied into this project instead of mounting the previous `unified` project.

## 4. Code Architecture

Main modules:

```text
src/prompt_attack/
  data/
  models/
  generators/
  attacks/
  metrics/
  utils/
```

Important files:

- `src/prompt_attack/generators/flux2.py`
  - FLUX.2 Klein adapter
  - prompt embedding composition
  - soft token injection
  - differentiable pipeline call

- `src/prompt_attack/attacks/runner.py`
  - attack loop
  - loss computation
  - image saving
  - metric aggregation
  - W&B logging

- `src/prompt_attack/attacks/losses.py`
  - attack loss definitions
  - DINO semantic loss

- `src/prompt_attack/attacks/soft_tokens.py`
  - learnable soft token initialization
  - prompt template construction

- `src/prompt_attack/metrics/summary.py`
  - ASR, semantic ASR, confidence drop, decision logit gap drop, IQA/FID summary

- `src/prompt_attack/metrics/nr_iqa.py`
  - NIMA-AVA
  - HyperIQA
  - MUSIQ-AVA
  - MUSIQ-KonIQ
  - TReS

- `src/prompt_attack/metrics/fid.py`
  - FID computation over original/adversarial image sets

## 5. Current Models

Generator:

```yaml
name: flux2_klein_4b
model_id: black-forest-labs/FLUX.2-klein-4B
height: 512
width: 512
num_inference_steps: 4
```

Victim classifier:

```yaml
name: resnet18
weights: IMAGENET1K_V1
```

Semantic encoder:

```yaml
name: dinov2_vitb14
feature: cls
```

## 6. Loss Design

### 6.1 Original Loss

The original optimization objective was:

```text
L_total = L_attack + lambda_sem * L_DINO
```

Current attack loss:

```text
L_attack = - CE(logits, true_label)
```

Because the optimizer minimizes `L_total`, minimizing `-CE` maximizes the true-label cross entropy. This pushes the victim classifier away from the correct class.

Semantic loss:

```text
L_DINO = 1 - cosine(DINO(original), DINO(generated))
```

This encourages the generated image to preserve the original image's high-level semantic content.

### 6.2 Removed Loss

The token regularization loss was removed:

```text
L_token = ||E_soft - E_init||_2^2
```

Reason: semantic preservation is already handled by DINO similarity, and token regularization could unnecessarily restrict the search space.

### 6.3 No-DINO Loss Variant

A no-DINO loss version was implemented.

When:

```yaml
lambda_sem: 0.0
```

the optimization loss becomes:

```text
L_total = L_attack
```

DINO is still computed for evaluation and W&B logging, but it is no longer part of the gradient loss.

Implementation detail:

- If `lambda_sem > 0`, DINO similarity is computed inside the differentiable loss graph.
- If `lambda_sem == 0`, DINO similarity is computed under `torch.no_grad()` after the attack loss backward pass.

## 7. Metrics Implemented

Per-image metrics:

- clean prediction
- adversarial prediction
- attack success
- semantic-constrained success
- clean true confidence
- adversarial true confidence
- confidence drop
- clean decision logit gap
- adversarial decision logit gap
- decision logit gap drop
- DINO similarity
- SSIM
- pixel L1 mean
- pixel L2
- pixel L2 mean
- pixel Linf
- first success step
- best attack step
- runtime seconds

Image quality metrics:

- NIMA-AVA
- HyperIQA
- MUSIQ-AVA
- MUSIQ-KonIQ
- TReS
- FID

W&B logging improvements:

- Per-image training curves are separated.
- Loss curves no longer continue from one image into the next image.
- `margin` naming in W&B was changed to decision/logit-gap terminology.
- Generated images and result tables are logged to W&B.

## 8. Completed Experiments

### 8.1 3-Image Run: lambda_sem = 0.5, negative CE

Config characteristics:

```yaml
objective: negative_cross_entropy
lambda_sem: 0.5
lr_scheduler: cosine
num_inference_steps: 4
guidance_scale: 1.0
```

Output:

```text
outputs/flux2_klein_4b_lambda0.5_cosine_negce_fid_iqa/
```

W&B run:

```text
bewukw6c
```

Summary:

| Metric | Value |
|---|---:|
| Count | 3 |
| Success count | 2 |
| ASR | 0.6667 |
| Semantic-constrained success count | 2 |
| Semantic-constrained ASR | 0.6667 |
| Mean DINO similarity | 0.9133 |
| Mean SSIM | 0.6283 |
| Mean confidence drop | 0.2525 |
| Mean decision logit gap drop | 1.5645 |
| Mean NIMA-AVA | 5.0574 |
| Mean HyperIQA | 0.5355 |
| Mean MUSIQ-AVA | 4.5193 |
| Mean MUSIQ-KonIQ | 68.7261 |
| Mean TReS | 70.6393 |
| FID | 85.5033 |
| Mean runtime seconds | 381.45 |

Per-image result:

| Class | Adv prediction | Success | Clean true conf | Adv true conf | DINO | SSIM |
|---|---|---:|---:|---:|---:|---:|
| dung beetle | ground beetle | true | 0.8270 | 0.0362 | 0.8892 | 0.8422 |
| bull mastiff | boxer | true | 0.7427 | 0.4783 | 0.9444 | 0.7083 |
| folding chair | folding chair | false | 0.6537 | 0.9513 | 0.9062 | 0.3343 |

Observation:

- `dung beetle` and `bull mastiff` were successfully pushed into visually related neighboring classes.
- `folding chair` failed. The generated image became a clearer, more canonical folding chair, increasing true-class confidence.

### 8.2 50-Image Run: lambda_sem = 0.5, negative CE

Config characteristics:

```yaml
objective: negative_cross_entropy
lambda_sem: 0.5
lr_scheduler: cosine
num_inference_steps: 4
guidance_scale: 1.0
```

Output:

```text
outputs/flux2_klein_4b_lambda0.5_cosine_negce_fid_iqa_max50/
```

W&B run:

```text
n18jdy3h
```

Summary:

| Metric | Value |
|---|---:|
| Count | 50 |
| Success count | 18 |
| ASR | 0.3600 |
| Semantic-constrained success count | 11 |
| Semantic-constrained ASR | 0.2200 |
| Mean DINO similarity | 0.8522 |
| Mean SSIM | 0.6269 |
| Mean confidence drop | 0.2324 |
| Mean decision logit gap drop | 1.9045 |
| Mean NIMA-AVA | 5.0871 |
| Mean HyperIQA | 0.5800 |
| Mean MUSIQ-AVA | 4.8403 |
| Mean MUSIQ-KonIQ | 69.2324 |
| Mean TReS | 76.1728 |
| FID | 81.1801 |
| Mean runtime seconds | 325.88 |

Initial sample rows:

| Class | Adv prediction | Success | Semantic success | Clean true conf | Adv true conf | DINO | SSIM |
|---|---|---:|---:|---:|---:|---:|---:|
| dung beetle | ground beetle | true | true | 0.8270 | 0.3595 | 0.9650 | 0.9105 |
| bull mastiff | boxer | true | true | 0.7427 | 0.3507 | 0.9475 | 0.6783 |
| folding chair | folding chair | false | false | 0.6537 | 0.7860 | 0.9072 | 0.7697 |
| beaker | beaker | false | false | 0.6193 | 0.9004 | 0.7080 | 0.7430 |
| buckeye | custard apple | true | true | 0.9067 | 0.0223 | 0.9609 | 0.9008 |

Observation:

- The 50-image run is the first meaningful small-scale estimate.
- Overall ASR was 36%.
- Semantic-constrained ASR dropped to 22%, indicating that some attack successes come with semantic drift.
- Some classes become more confident after generation, suggesting that the generator often denoises/canonicalizes the object instead of weakening it.

### 8.3 3-Image Run: No-DINO Loss

Config characteristics:

```yaml
objective: negative_cross_entropy
lambda_sem: 0.0
lr_scheduler: cosine
num_inference_steps: 4
guidance_scale: 1.0
```

Output:

```text
outputs/flux2_klein_4b_negce_nodino_max3/
```

W&B run:

```text
kt51m7dq
```

Summary:

| Metric | Value |
|---|---:|
| Count | 3 |
| Success count | 2 |
| ASR | 0.6667 |
| Semantic-constrained success count | 2 |
| Semantic-constrained ASR | 0.6667 |
| Mean DINO similarity | 0.9101 |
| Mean SSIM | 0.6546 |
| Mean confidence drop | 0.3044 |
| Mean decision logit gap drop | 2.1325 |
| Mean NIMA-AVA | 5.0286 |
| Mean HyperIQA | 0.5528 |
| Mean MUSIQ-AVA | 4.5540 |
| Mean MUSIQ-KonIQ | 69.1997 |
| Mean TReS | 73.1116 |
| FID | 90.0311 |
| Mean runtime seconds | 323.02 |

Per-image result:

| Class | Adv prediction | Success | Clean true conf | Adv true conf | DINO | SSIM |
|---|---|---:|---:|---:|---:|---:|
| dung beetle | ground beetle | true | 0.8270 | 0.0249 | 0.8908 | 0.8567 |
| bull mastiff | boxer | true | 0.7427 | 0.4713 | 0.9412 | 0.7212 |
| folding chair | folding chair | false | 0.6537 | 0.8138 | 0.8984 | 0.3860 |

Observation:

- No-DINO loss kept the same ASR on this 3-image subset.
- For `folding chair`, removing DINO reduced the confidence increase:

```text
lambda_sem = 0.5: 0.6537 -> 0.9513
lambda_sem = 0.0: 0.6537 -> 0.8138
```

- This suggests that DINO semantic preservation can sometimes encourage class-preserving canonicalization, especially for object categories like folding chair.
- However, no-DINO still failed on folding chair.

### 8.4 Guidance Scale 4.0 Attempt

The config was changed to:

```yaml
guidance_scale: 4.0
```

but the run was stopped after observing this warning:

```text
Guidance scale 4.0 is ignored for step-wise distilled models.
```

Conclusion:

- For the current `FLUX.2-klein-4B` diffusers pipeline, `guidance_scale` is not an effective control knob.
- It should not be used as evidence for stronger prompt conditioning.
- The current config filenames/output names include `gs4`, but the experiment should not be interpreted as a valid guidance-scale ablation unless the generator backend changes.

## 9. Important Observations So Far

### 9.1 White-box Does Not Guarantee Success

Even though the attack is white-box with respect to the victim classifier, the optimized variable is not the pixel image. The optimized variable is the soft prompt embedding.

Therefore, the actual search space is constrained by:

- FLUX.2 edit manifold
- hard class label prompt
- image conditioning
- soft token capacity
- denoising schedule
- optimization landscape

This explains why some images fail even under white-box optimization.

### 9.2 Folding Chair Is a Hard Case

The `folding chair` image repeatedly fails.

Reason:

- The original image contains multiple red folding chairs.
- FLUX editing tends to create a more canonical, foregrounded, clean folding chair.
- ResNet-18 confidence increases because the generated image strengthens the class-relevant features:

```text
red fabric seat
wooden folding frame
clear X-shaped chair structure
larger foreground object
```

This is why the true-class confidence increased in both lambda=0.5 and no-DINO settings.

### 9.3 DINO Loss Helps Semantic Preservation but Can Help the Classifier

DINO similarity is useful for semantic preservation, but for some classes it can preserve or strengthen exactly the features that the classifier uses.

This creates a tension:

```text
attack loss: reduce true-class evidence
DINO loss: preserve image semantics
hard class label: keep class identity
generator prior: make image natural and canonical
```

For some classes, these forces oppose each other.

### 9.4 Natural Attack Success Often Moves to Neighboring Classes

Successful examples so far often move to semantically nearby or visually related classes:

```text
dung beetle -> ground beetle
bull mastiff -> boxer
buckeye -> custard apple
```

This is consistent with a natural adversarial setting: the image remains plausible but crosses a classifier boundary toward a related category.

## 10. Current Limitations

### 10.1 Edit Strength Is Not Directly Exposed

The current FLUX.2 Klein pipeline signature supports:

```text
image
prompt
prompt_embeds
num_inference_steps
sigmas
guidance_scale
latents
```

but it does not expose a standard img2img `strength` parameter.

Additionally, `guidance_scale` is ignored for the current step-wise distilled model, so it cannot be used to increase deformation strength.

### 10.2 Soft Token Initialization Is Fixed

Current soft token initialization:

```python
values = torch.randn(num_tokens, token_dim) * 0.02
```

This is hard-coded and conservative. Stronger initial scale may help the optimization escape weak edit directions.

### 10.3 Prompt Template Is Still Class-Anchored

Current prompt:

```text
[V1] ... [V8] a photo of {class_label}
```

The phrase `a photo of` and the hard class label can encourage the generator to produce a clean class-consistent image. This can work against attack success.

### 10.4 FID on Very Small Runs Is Not Reliable

FID is computed for 3-image runs, but this should be interpreted only as a sanity check. FID becomes more meaningful on larger sample sizes.

## 11. Current Active Configs

Main config:

```text
configs/flux2_resnet18_imagenet10.yaml
```

Important settings:

```yaml
victim:
  name: resnet18

attack:
  objective: negative_cross_entropy
  lambda_sem: 0.5

generator:
  guidance_scale: 4.0
```

Note: `guidance_scale: 4.0` is currently set in the config, but FLUX.2 Klein logs that guidance is ignored for step-wise distilled models.

No-DINO config:

```text
configs/flux2_resnet18_imagenet10_nodino.yaml
```

Important settings:

```yaml
victim:
  name: resnet18

attack:
  objective: negative_cross_entropy
  lambda_sem: 0.0

generator:
  guidance_scale: 4.0
```

Same warning applies: guidance scale is ignored by the current generator backend.

## 12. Recommended Next Experiments

Because `guidance_scale` is ignored, stronger deformation should be pursued through other knobs.

### 12.1 Add Configurable Soft Token Initialization Scale

Add:

```yaml
attack:
  soft_token_init_std: 0.1
```

Candidate values:

```text
0.02 baseline
0.05 moderate
0.10 strong
0.20 aggressive
```

### 12.2 Increase Number of Soft Tokens

Current:

```yaml
num_soft_tokens: 8
```

Next:

```yaml
num_soft_tokens: 16
```

This increases the conditioning capacity of the attack.

### 12.3 Increase Denoising Steps

Current:

```yaml
num_inference_steps: 4
```

Next:

```yaml
num_inference_steps: 8
```

This may allow deeper image edits, although runtime will increase.

### 12.4 Prompt Template Ablation

Current:

```text
[V1] ... [V8] a photo of {class_label}
```

Candidate:

```text
[V1] ... [V8] {class_label}
```

Motivation:

- Remove the extra natural-photo anchor.
- Reduce generator tendency to produce canonical class images.

### 12.5 Suggested Immediate 3-Image Ablation Plan

Run these on the same first 3 images:

| Run name | lambda_sem | num_soft_tokens | init std | inference steps | prompt template |
|---|---:|---:|---:|---:|---|
| nodino_tokens8_init002_steps4 | 0.0 | 8 | 0.02 | 4 | `a photo of class` |
| nodino_tokens16_init002_steps4 | 0.0 | 16 | 0.02 | 4 | `a photo of class` |
| nodino_tokens16_init010_steps4 | 0.0 | 16 | 0.10 | 4 | `a photo of class` |
| nodino_tokens16_init010_steps8 | 0.0 | 16 | 0.10 | 8 | `a photo of class` |
| nodino_tokens16_init010_steps8_noaphoto | 0.0 | 16 | 0.10 | 8 | `class` |

Primary readout:

- ASR
- confidence drop
- decision logit gap drop
- DINO similarity
- SSIM
- pixel L1/L2/Linf
- visual inspection in W&B

## 13. Open Research Questions

1. Does higher soft-token capacity produce stronger but still natural edits?
2. Does removing `a photo of` reduce class-canonicalization failures?
3. Is DINO similarity the right semantic constraint, or does it preserve classifier-relevant features too strongly?
4. Should semantic preservation be enforced as a post-hoc filter instead of a differentiable loss?
5. Are certain classes systematically resistant because the generator canonicalizes them?
6. Does transferability hold across ResNet-18, ResNet-50, ViT, ConvNeXt, etc.?
7. Can generated failure modes be clustered into interpretable OOD directions?

## 14. Current Best Interpretation

The current pipeline already demonstrates that learnable soft prompt tokens can produce natural adversarial images for some ImageNet classes. However, the current edit strength is limited by the FLUX.2 Klein image-conditioning behavior and the small soft-token search space.

The strongest result so far is not raw ASR, but the observation that the method can automatically find semantically plausible class-boundary crossings without human-written adversarial prompts.

Examples:

```text
dung beetle -> ground beetle
bull mastiff -> boxer
buckeye -> custard apple
```

The main failure mode is generator canonicalization:

```text
folding chair -> clearer folding chair
beaker -> clearer beaker
```

This suggests the next phase should focus on increasing controllable edit strength while still preserving naturalness.

## 15. Full Environment Setup and Operating Guide

This section records the practical setup needed to reproduce and continue the experiments.

### 15.1 Host Machine Assumptions

Current expected environment:

```text
OS: Windows
GPU: NVIDIA GPU, target RTX 3090 Ti 24GB
Container runtime: Docker Desktop with NVIDIA GPU support
Project path: D:\code\promtp_attack
Dataset path: D:\code\promtp_attack\dataset
```

The project folder name is intentionally spelled:

```text
promtp_attack
```

Do not rename it unless all Docker mount paths and scripts are updated.

### 15.2 Project Directory

Current project structure:

```text
D:\code\promtp_attack
  configs/
    flux2_resnet18_imagenet10.yaml
    flux2_resnet18_imagenet10_nodino.yaml
  dataset/
    images.csv
    images/
  docker/
    Dockerfile
    docker-compose.yml
    .env
  external/
    pytorch_fid/
  outputs/
  scripts/
    run_attack.py
    smoke_test.py
    select_imagenet_subset.py
    time_attack.py
  src/prompt_attack/
  tests/
  pyproject.toml
  README.md
  RESEARCH_PROGRESS.md
```

### 15.3 Dataset Format

Current active data mode:

```yaml
data:
  imagenet_root: /data/imagenet
  split:
  class_mode: csv_images
```

Because `split` is empty, the Docker path `/data/imagenet` directly points to the mounted host dataset folder.

Required host dataset layout:

```text
D:\code\promtp_attack\dataset
  images.csv
  images/
    0c7ac4a8c9dfa802.png
    4fc263d35a3ad3ee.png
    cc13c2bc5cdd1f44.png
    ...
```

CSV columns currently used by the loader:

```text
ImageId
TrueLabel
```

Important detail:

- `TrueLabel` in `images.csv` is treated as 1-indexed ImageNet label.
- The code converts it internally using:

```python
class_idx = int(row["TrueLabel"]) - 1
```

The image file must exist at:

```text
dataset/images/{ImageId}.png
```

Example row:

```csv
ImageId,URL,x1,y1,x2,y2,TrueLabel,TargetClass,...
0c7ac4a8c9dfa802,...,306,779,...
```

This becomes ImageNet class index:

```text
305
```

which maps to:

```text
dung beetle
```

### 15.4 Alternative ImageNet Folder Mode

The code still supports the original fixed 10-class ImageNet folder mode:

```yaml
data:
  imagenet_root: /data/imagenet
  split: val
  class_mode: fixed_10
```

Expected layout for that mode:

```text
/data/imagenet/val/
  n01685808/
  n02113624/
  ...
```

The fixed 10 classes are:

| ImageNet index | Label |
|---:|---|
| 41 | whiptail, whiptail lizard |
| 265 | toy poodle |
| 394 | sturgeon |
| 430 | basketball |
| 497 | church, church building |
| 523 | crutch |
| 776 | sax, saxophone |
| 864 | tow truck |
| 911 | wool |
| 988 | acorn |

This fixed-10 mode is not the current active run mode. Current experiments use `csv_images`.

### 15.5 Docker Files

Dockerfile:

```text
docker/Dockerfile
```

Base image:

```dockerfile
nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
```

Python:

```text
Python 3.11 venv at /opt/venv
```

Important packages:

- torch / torchvision / torchaudio from CUDA 12.8 wheel index
- diffusers from GitHub
- transformers
- accelerate
- pyiqa
- wandb
- pandas
- pillow
- opencv-python-headless
- scikit-learn
- ruff / mypy / pytest for dev checks

Extra system libraries are installed because `pyiqa` needs image/model runtime dependencies:

```text
libglib2.0-0
libgl1
libsm6
libxext6
libxrender1
libxcb1
```

Compose file:

```text
docker/docker-compose.yml
```

Important mounts:

```yaml
volumes:
  - ..:/workspace/promtp_attack
  - ../dataset:/data/imagenet:ro
  - ../.cache/huggingface:/root/.cache/huggingface
  - ../.cache/torch:/root/.cache/torch
```

Important environment:

```yaml
environment:
  - HF_HOME=/root/.cache/huggingface
  - HF_HUB_ENABLE_HF_TRANSFER=1
  - PYTHONPATH=/workspace/promtp_attack/src
  - TORCH_HOME=/root/.cache/torch
  - NVIDIA_VISIBLE_DEVICES=all
  - NVIDIA_DRIVER_CAPABILITIES=compute,utility
```

GPU:

```yaml
gpus: all
shm_size: "16gb"
```

### 15.6 Environment Variables

The Docker compose service reads:

```text
docker/.env
```

Do not commit or paste real API keys into documentation.

Template:

```env
WANDB_MODE=online
WANDB_PROJECT=prompt-soft-token-attack
WANDB_NAME=flux2_resnet18_experiment_name
WANDB_API_KEY=<your_wandb_api_key>
HF_TOKEN=<optional_huggingface_token>
```

Current important behavior:

- `WANDB_MODE=online` sends metrics and images to W&B.
- `WANDB_NAME` can be overridden per run.
- `HF_TOKEN` is optional but can help avoid Hugging Face rate limits.

Inside Python, the W&B logger resolves values in this order:

```text
environment variable > config yaml value
```

Therefore, when using `docker compose exec -e WANDB_NAME=...`, that value overrides the YAML run name.

### 15.7 Build the Docker Image

From host PowerShell:

```powershell
Set-Location D:\code\promtp_attack
docker compose -f docker\docker-compose.yml build prompt-attack
```

If dependencies changed in `pyproject.toml` or `Dockerfile`, rebuild:

```powershell
docker compose -f docker\docker-compose.yml build --no-cache prompt-attack
```

Usually `--no-cache` is not needed.

### 15.8 Start a Persistent Container

Preferred workflow is to start the container once and then enter it or execute commands inside it.

From host PowerShell:

```powershell
Set-Location D:\code\promtp_attack
docker compose -f docker\docker-compose.yml up -d prompt-attack
```

Enter the container:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash
```

Inside the container, the working directory should be:

```text
/workspace/promtp_attack
```

Confirm:

```bash
pwd
```

Expected:

```text
/workspace/promtp_attack
```

Stop the container:

```powershell
docker compose -f docker\docker-compose.yml stop prompt-attack
```

Recreate it after `.env` changes:

```powershell
docker compose -f docker\docker-compose.yml up -d --force-recreate prompt-attack
```

### 15.9 Sanity Checks

Run these inside the container.

GPU check:

```bash
nvidia-smi
```

Python / CUDA check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Import-only smoke:

```bash
python scripts/smoke_test.py \
  --config configs/flux2_resnet18_imagenet10.yaml \
  --imports-only
```

Mock-generator smoke test:

```bash
python scripts/smoke_test.py \
  --config configs/flux2_resnet18_imagenet10.yaml \
  --max-images 4
```

Real-generator smoke test:

```bash
python scripts/smoke_test.py \
  --config configs/flux2_resnet18_imagenet10.yaml \
  --real-generator \
  --max-images 1
```

Code quality checks:

```bash
ruff check .
mypy src scripts tests
pytest -q
```

### 15.10 Main Configs

Main semantic-loss config:

```text
configs/flux2_resnet18_imagenet10.yaml
```

The current victim model inside the config is:

```yaml
victim:
  name: resnet18
  weights: IMAGENET1K_V1
```

Current main attack:

```yaml
attack:
  num_soft_tokens: 8
  soft_token_init_std: 0.02
  lr: 1.0e-2
  lr_scheduler:
    name: cosine
    warmup_steps: 5
    min_lr: 1.0e-4
  steps: 100
  lambda_sem: 0.5
  semantic_threshold: 0.85
  objective: negative_cross_entropy
```

No-DINO config:

```text
configs/flux2_resnet18_imagenet10_nodino.yaml
```

Current no-DINO attack:

```yaml
attack:
  num_soft_tokens: 8
  soft_token_init_std: 0.02
  lr: 1.0e-2
  lr_scheduler:
    name: cosine
    warmup_steps: 5
    min_lr: 1.0e-4
  steps: 100
  lambda_sem: 0.0
  semantic_threshold: 0.85
  objective: negative_cross_entropy
```

Generator settings:

```yaml
generator:
  name: flux2_klein_4b
  model_id: black-forest-labs/FLUX.2-klein-4B
  precision: bf16
  height: 512
  width: 512
  num_inference_steps: 4
```

Important caveat:

```yaml
guidance_scale: 4.0
```

is currently set in both configs, but `FLUX.2-klein-4B` logs:

```text
Guidance scale 4.0 is ignored for step-wise distilled models.
```

Therefore, `guidance_scale` should not be interpreted as an active edit-strength control for this generator.

### 15.11 Running Experiments from Inside the Container

Enter the container first:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash
```

Then run commands below inside the container.

#### 3-image main run with DINO loss

```bash
export WANDB_NAME=flux2_resnet18_lambda0.5_negce_max3
python scripts/run_attack.py \
  --config configs/flux2_resnet18_imagenet10.yaml \
  --max-images 3
```

Recommended with log file:

```bash
export WANDB_NAME=flux2_resnet18_lambda0.5_negce_max3
python scripts/run_attack.py \
  --config configs/flux2_resnet18_imagenet10.yaml \
  --max-images 3 \
  > outputs/run_3_images_lambda0.5_negce.log 2>&1
```

#### 3-image no-DINO run

```bash
export WANDB_NAME=flux2_resnet18_negce_nodino_max3
python scripts/run_attack.py \
  --config configs/flux2_resnet18_imagenet10_nodino.yaml \
  --max-images 3 \
  > outputs/run_3_images_negce_nodino.log 2>&1
```

#### 50-image run

```bash
export WANDB_NAME=flux2_resnet18_lambda0.5_negce_max50
python scripts/run_attack.py \
  --config configs/flux2_resnet18_imagenet10.yaml \
  --max-images 50 \
  > outputs/run_50_images_lambda0.5_negce.log 2>&1
```

#### 100-image run

```bash
export WANDB_NAME=flux2_resnet18_lambda0.5_negce_max100
python scripts/run_attack.py \
  --config configs/flux2_resnet18_imagenet10.yaml \
  --max-images 100 \
  > outputs/run_100_images_lambda0.5_negce.log 2>&1
```

### 15.12 Running Experiments from Host PowerShell

If you do not want to enter the container, use `docker compose exec`.

Example 3-image no-DINO run:

```powershell
docker compose -f docker\docker-compose.yml exec -d `
  -e WANDB_NAME=flux2_resnet18_negce_nodino_max3 `
  prompt-attack `
  bash -lc "python scripts/run_attack.py --config configs/flux2_resnet18_imagenet10_nodino.yaml --max-images 3 > outputs/run_3_images_negce_nodino.log 2>&1"
```

Example 50-image main run:

```powershell
docker compose -f docker\docker-compose.yml exec -d `
  -e WANDB_NAME=flux2_resnet18_lambda0.5_negce_max50 `
  prompt-attack `
  bash -lc "python scripts/run_attack.py --config configs/flux2_resnet18_imagenet10.yaml --max-images 50 > outputs/run_50_images_lambda0.5_negce.log 2>&1"
```

### 15.13 Monitoring a Running Experiment

Check running process:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash -lc "ps -ef | grep -E 'scripts/run_attack.py' | grep -v grep || true"
```

Tail log:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash -lc "tail -n 100 outputs/run_50_images_lambda0.5_negce.log"
```

Follow log:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash -lc "tail -f outputs/run_50_images_lambda0.5_negce.log"
```

Check number of completed rows:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash -lc "wc -l outputs/<run_dir>/metrics/results.csv"
```

Remember:

- `wc -l` includes the CSV header.
- `51` lines means `50` image results.

### 15.14 Stopping a Running Experiment

Find process:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash -lc "ps -ef | grep -E 'scripts/run_attack.py' | grep -v grep"
```

Kill by PID:

```powershell
docker compose -f docker\docker-compose.yml exec prompt-attack bash -lc "kill <PID>"
```

Use this when:

- wrong config was used
- W&B name is wrong
- generator warning invalidates the run
- GPU memory is needed for another run

### 15.15 Output Layout

Each run writes to:

```text
outputs/<run_name>/
  metrics/
    results.csv
    summary.json
  images/
    {class_label}/
      {image_id}/
        original.png
        adv.png
  grids/
    {class_label}_{image_id}.png
```

Example:

```text
outputs/flux2_klein_4b_negce_nodino_max3/
  metrics/results.csv
  metrics/summary.json
  images/folding chair/cc13c2bc5cdd1f44/original.png
  images/folding chair/cc13c2bc5cdd1f44/adv.png
  grids/folding chair_cc13c2bc5cdd1f44.png
```

W&B outputs:

```text
outputs/wandb/
```

### 15.16 Reading Results

Print summary:

```powershell
Get-Content outputs\<run_name>\metrics\summary.json
```

Read per-image CSV in PowerShell:

```powershell
Import-Csv outputs\<run_name>\metrics\results.csv |
  Select-Object class_label,adv_pred_label,success,semantic_constrained_success,clean_true_conf,adv_true_conf,confidence_drop,dino_similarity,ssim |
  Format-Table -AutoSize
```

Read per-image CSV inside container:

```bash
python - <<'PY'
import pandas as pd

p = "outputs/<run_name>/metrics/results.csv"
df = pd.read_csv(p)
cols = [
    "image_id",
    "class_label",
    "adv_pred_label",
    "success",
    "semantic_constrained_success",
    "clean_true_conf",
    "adv_true_conf",
    "confidence_drop",
    "dino_similarity",
    "ssim",
    "first_success_step",
    "best_attack_step",
]
print(df[cols].to_string(index=False))
PY
```

### 15.17 W&B Usage

The current W&B project is:

```text
prompt-soft-token-attack
```

Current W&B logs include:

- per-image training curves
- image-level result metrics
- generated images
- result tables
- summary metrics

Important W&B naming rule:

Always include meaningful changes in the run name.

Good examples:

```text
flux2_resnet18_lambda0.5_negce_max50
flux2_resnet18_negce_nodino_max3
flux2_resnet18_negce_nodino_tokens16_init010_steps8_max3
```

Avoid:

```text
test
run1
latest
```

### 15.18 Runtime Estimates

Measured runtime:

```text
3-image lambda_sem=0.5 run: mean 381 sec/image
3-image no-DINO run: mean 323 sec/image
50-image lambda_sem=0.5 run: mean 326 sec/image
```

Practical estimates:

| Images | Expected time |
|---:|---:|
| 3 | 16-20 min |
| 10 | 55-70 min |
| 50 | 4.5-5.5 hours |
| 100 | 9.5-11 hours |
| 200 | 19-22 hours |

The first run may be slower because model weights are downloaded or cached.

### 15.19 Known Warnings and What They Mean

#### Guidance Scale Warning

Warning:

```text
Guidance scale 4.0 is ignored for step-wise distilled models.
```

Meaning:

- `guidance_scale` is not active for the current FLUX.2 Klein step-wise distilled pipeline.
- Do not treat `guidance_scale=4.0` as stronger prompt guidance.
- Stop the run if it was intended as a guidance ablation.

#### xFormers Warning

Warnings:

```text
xFormers is not available (Attention)
xFormers is not available (SwiGLU)
```

Meaning:

- The model still runs.
- It may be slower or use more memory than an xFormers-enabled setup.

#### pyiqa Downloads

First IQA run may download weights:

```text
NIMA
HyperIQA
MUSIQ
TReS
```

After caching, later runs should load from:

```text
/root/.cache/torch/hub/pyiqa/
```

### 15.20 Current Recommended Next Run

Because `guidance_scale` is ignored, do not spend more runs on guidance-scale ablation with the current generator.

Recommended next experiment config:

```yaml
attack:
  num_soft_tokens: 16
  soft_token_init_std: 0.1
  lambda_sem: 0.0

generator:
  num_inference_steps: 8
```

`soft_token_init_std` is now configurable through `AttackConfig` and passed into
`initialize_soft_tokens`.

Then run:

```bash
export WANDB_NAME=flux2_resnet18_negce_nodino_tokens16_init010_steps8_max3
python scripts/run_attack.py \
  --config configs/<new_config>.yaml \
  --max-images 3 \
  > outputs/run_3_images_tokens16_init010_steps8.log 2>&1
```

Primary question:

```text
Does stronger soft-token capacity increase deformation and ASR without destroying DINO similarity?
```

### 15.21 Minimal Reproduction Checklist

Use this checklist when starting from a clean machine.

```text
1. Install Docker Desktop with NVIDIA GPU support.
2. Put project at D:\code\promtp_attack.
3. Put dataset at D:\code\promtp_attack\dataset.
4. Create docker\.env with W&B variables.
5. Build Docker image.
6. Start persistent container.
7. Run GPU sanity check.
8. Run import-only smoke test.
9. Run mock smoke test.
10. Run 1-image real-generator smoke test.
11. Run 3-image experiment.
12. Inspect results.csv, summary.json, W&B images.
13. Only then run 50+ images.
```
