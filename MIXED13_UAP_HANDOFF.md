# Mixed13 UAP Handoff Notes

작성일: 2026-07-02 KST  
프로젝트: FLUX.2 learnable textual-inversion token 기반 natural UAP attack  
현재 주요 보고서: `outputs/uap_mixed13_report/mixed13_uap_loss_report.html`

이 문서는 다른 AI 또는 연구자가 현재 코드와 실험 상태를 바로 이어받을 수 있도록 정리한 핸드오프 문서다.

## 1. 핵심 요약

- 기존 ImageNet-1K victim 대신 `mixed_13` superclass victim을 추가했다.
- `mixed_13` victim은 ImageNet pretrained ResNet-18의 final layer를 13-way로 교체해 fine-tune했다.
- UAP 공격은 FLUX.2-klein-4B image editing 경로에서 `<v1> ... <v32>` learnable textual-inversion token을 학습한다.
- 현재 mixed13 UAP 실험은 token 32, global batch 16, generator batch 1, num inference steps 4 기준이다.
- 높은 Raw ASR은 `margin_dino`, `margin_clip_img2img`에서만 나온다.
- 하지만 높은 ASR run은 DINO/SSIM이 크게 낮아져 이미지 collapse 가능성이 높다.
- 현재 결론: **ASR만 올리는 것은 가능하지만, 의미 보존된 공격 성공은 아직 해결되지 않았다.**

## 2. 현재 환경

### Local Docker

- Workspace: `D:\code\promtp_attack`
- Docker container: `prompt_attack_tmux`
- Container workdir: `/workspace/promtp_attack`
- Dataset mount inside Docker: `/data/imagenet`
- Current local GPU: RTX 3090 Ti

자주 쓰는 접속 명령:

```bash
docker exec -it -w /workspace/promtp_attack prompt_attack_tmux bash
```

tmux 확인:

```bash
docker exec -it -w /workspace/promtp_attack prompt_attack_tmux tmux ls
```

GPU 확인:

```bash
docker exec -w /workspace/promtp_attack prompt_attack_tmux \
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,power.draw \
  --format=csv,noheader,nounits
```

### B200 환경 참고

- Code path: `/NHNHOME/WORKSPACE/0226010134_A/daeyun/prompt_attack`
- Dataset path: `/NHNHOME/WORKSPACE/0226010134_A/data/Imagenet`
- B200에서는 Docker가 아니라 `uv` 기반으로 실행했던 흐름이 있다.
- GPU 선택 예:

```bash
CUDA_VISIBLE_DEVICES=4 uv run python scripts/run_uap.py ...
```

## 3. Mixed13 데이터셋

`mixed_13`은 ImageNet WordNet hierarchy 기반 superclass subset이다.

| id | superclass | synset |
|---:|---|---|
| 0 | dog | `n02084071` |
| 1 | bird | `n01503061` |
| 2 | insect | `n02159955` |
| 3 | furniture | `n03405725` |
| 4 | fish | `n02512053` |
| 5 | monkey | `n02484322` |
| 6 | car | `n02958343` |
| 7 | cat | `n02120997` |
| 8 | truck | `n04490091` |
| 9 | fruit | `n13134947` |
| 10 | fungus | `n12992868` |
| 11 | boat | `n02858304` |
| 12 | computer | `n03082979` |

관련 코드:

- `src/prompt_attack/data/superclass.py`
- `src/prompt_attack/data/imagenet.py`
- `external/imagenet_hierarchy_wordnet/`

## 4. Mixed13 Victim Classifier

Checkpoint:

```text
outputs/victims/mixed13_resnet18/best.pt
```

요약:

- Architecture: ResNet-18
- Backbone init: `IMAGENET1K_V1`
- Head: 13-way classifier
- Dataset: `mixed_13`
- Train records: 408,885
- Val records: 16,000
- Best epoch: 5
- Best val accuracy: 95.52%
- Mapping hash: `be34a6211554651bf6b4866b9ce89ac1dc9cb1bda416b90651085eda0140abbf`

Victim training summary:

```text
outputs/victims/mixed13_resnet18/metrics/summary.json
```

Training script:

```text
scripts/train_superclass_victim.py
```

재학습 예:

```bash
python scripts/train_superclass_victim.py \
  --dataset mixed_13 \
  --imagenet-root /data/imagenet \
  --imagenet-info-root external/imagenet_hierarchy_wordnet \
  --output-root outputs/victims/mixed13_resnet18 \
  --architecture resnet18 \
  --weights IMAGENET1K_V1 \
  --epochs 5 \
  --batch-size 256 \
  --workers 8 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --balanced-sampler \
  --amp
```

## 5. UAP Attack Code Map

주요 파일:

| file | role |
|---|---|
| `scripts/run_uap.py` | UAP 실험 CLI. 이름은 fixed10이지만 현재 mixed13도 지원한다. |
| `src/prompt_attack/config.py` | Config dataclass, CLI override, `VictimConfig`, attack/generator settings. |
| `src/prompt_attack/attacks/runner.py` | Universal/imagewise attack loop, clean-correct filtering, micro-batch accumulation, eval, CSV/summary 저장. |
| `src/prompt_attack/attacks/losses.py` | `cr`, `cr_dino`, `untargeted_margin`, `margin_dino`, `margin_clip`, semantic-only objectives. |
| `src/prompt_attack/attacks/learnable_tokens.py` | Learnable textual-inversion token prompt state. |
| `src/prompt_attack/models/flux2.py` | FLUX.2 adapter. Tokenizer token 추가, embedding hook, generation 경로. |
| `src/prompt_attack/models/victim.py` | ImageNet victim과 checkpoint 기반 mixed13 victim build. |
| `scripts/run_mixed13_loss_sweep.sh` | mixed13 e5 loss sweep. |
| `scripts/run_mixed13_loss_sweep_e10.sh` | mixed13 e5 checkpoint에서 추가 5 epoch continuation sweep. |
| `scripts/build_mixed13_uap_report.py` | mixed13 HTML 결과 보고서 생성. |
| `scripts/evaluate_collapse.py` | collapse metric 평가용 스크립트. |
| `src/prompt_attack/metrics/collapse.py` | collapse 판정 관련 metric helper. |
| `src/prompt_attack/metrics/perceptual.py` | LPIPS/DreamSim 등 perceptual metric helper. |

현재 git tree는 dirty 상태다. 다른 AI는 clean repository라고 가정하면 안 된다.

## 6. Learnable Token 방식

현재 공격 prompt:

```text
<v1> <v2> ... <vN> a photo of {class_label}
```

기본 mixed13 실험:

```text
num_tokens = 32
initializer = object
soft_token_init_std = 0.02
```

구현 방식:

- tokenizer에 `<v1> ... <vN>`을 추가한다.
- text encoder embedding output에서 `<v*>` 위치만 learnable `nn.Parameter`로 치환한다.
- FLUX.2 자체는 frozen이다.
- universal mode에서는 하나의 shared token set을 모든 train image가 공유한다.

중요: 이 방식은 old direct `prompt_embeds` prepend 방식이 아니다. 실제 tokenizer token을 추가해 textual-inversion 방식으로 학습한다.

## 7. Loss 정의

관련 파일:

```text
src/prompt_attack/attacks/losses.py
```

### `cr`

```text
loss = -CE(logits, true_label)
```

minimize하면 true label CE를 키워 classifier confidence를 낮춘다.

### `cr_dino`

```text
loss = (1 - lambda_sem) * cr_loss + lambda_sem * dino_loss
```

mixed13 sweep에서는 `lambda_sem = 0.5`.

### `untargeted_margin`

```text
loss = true_logit - max_other_logit
```

minimize하면 true class logit을 다른 class logit보다 낮추려 한다.

### `margin_dino`, `margin_clip_img2img`

현재 코드에서는 bounded hinge attack과 semantic preservation loss를 더한다.

```text
attack_loss = relu(true_logit - max_other_logit + attack_margin)
semantic_loss = 1 - similarity
loss = attack_loss + semantic_loss_weight * semantic_loss
```

mixed13 sweep 설정:

```text
attack_margin = 0.0
semantic_loss_weight = 3.0
```

### `clip_img2img`, `dino_img2img`

semantic-only objective다.

```text
attack_loss = 0
loss = 1 - similarity
```

따라서 ASR이 낮은 것이 정상이다. classifier를 속이라는 loss term이 없다.

## 8. Mixed13 UAP 실험 설정

공통 설정:

```text
class_mode = mixed_13
victim_name = resnet18_mixed13
victim_checkpoint = outputs/victims/mixed13_resnet18/best.pt
train_images_per_class = 20
test_images_per_class = 20
train records = 260
test records = 260
training_mode = universal
num_tokens = 32
global_batch_size = 16
generator_batch_size = 1
num_inference_steps = 4
lr = 0.1
lr_scheduler = cosine
wandb_mode = disabled
image_format = png
grid_save_policy = representative
max_saved_grids = 60
```

e5 sweep:

```bash
bash scripts/run_mixed13_loss_sweep.sh
```

e10 sweep:

```bash
bash scripts/run_mixed13_loss_sweep_e10.sh
```

e10은 scratch 10 epoch이 아니라, e5 `learned_prompt.pt`에서 추가 5 epoch을 학습한 continuation이다. Optimizer state는 이어받지 않고 새 Adam으로 시작한다.

## 9. 주요 결과

### e5 결과

| loss | ASR | success | DINO | CLIP | SSIM | confidence drop |
|---|---:|---:|---:|---:|---:|---:|
| `margin_dino` | 81.2% | 211/260 | 0.703 | - | 0.019 | 0.774 |
| `cr` | 8.1% | 21/260 | 0.812 | - | 0.586 | 0.073 |
| `cr_dino` | 3.5% | 9/260 | 0.980 | - | 0.959 | 0.013 |
| `untargeted_margin` | 2.7% | 7/260 | 0.987 | - | 0.968 | 0.014 |
| `margin_clip_img2img` | 2.7% | 7/260 | 0.988 | 0.970 | 0.962 | 0.015 |
| `clip_img2img` | 0.8% | 2/260 | 0.978 | 0.963 | 0.945 | 0.003 |
| `dino_img2img` | 2.7% | 7/260 | 0.982 | - | 0.964 | 0.010 |

### e10 결과

| loss | ASR | success | DINO | CLIP | SSIM | confidence drop |
|---|---:|---:|---:|---:|---:|---:|
| `margin_dino` | 82.3% | 214/260 | 0.737 | - | 0.025 | 0.785 |
| `margin_clip_img2img` | 80.0% | 208/260 | 0.102 | 0.396 | 0.024 | 0.776 |
| `cr` | 4.6% | 12/260 | 0.903 | - | 0.781 | 0.027 |
| `clip_img2img` | 3.8% | 10/260 | 0.927 | 0.916 | 0.812 | 0.020 |
| `untargeted_margin` | 1.9% | 5/260 | 0.990 | - | 0.978 | 0.012 |
| `dino_img2img` | 1.9% | 5/260 | 0.994 | - | 0.986 | 0.007 |
| `cr_dino` | 0.8% | 2/260 | 0.982 | - | 0.960 | 0.009 |

### margin_dino rerun

Run:

```text
mixed13_resnet18_t32_mdino_ipc20_gb16_e10_rerun1
```

Result:

| loss | ASR | success | DINO | SSIM | confidence drop |
|---|---:|---:|---:|---:|---:|
| `margin_dino` rerun | 81.2% | 211/260 | 0.720 | 0.017 | 0.773 |

해석:

- `margin_dino`의 높은 Raw ASR은 재현된다.
- 동시에 낮은 SSIM도 재현된다.
- 즉 이 결과는 우연한 1회성 실패가 아니라, 현재 objective가 collapse 방향으로 수렴하는 경향이 있다는 증거다.

## 10. 결과 보고서

HTML report:

```text
outputs/uap_mixed13_report/mixed13_uap_loss_report.html
```

생성 명령:

```bash
python scripts/build_mixed13_uap_report.py
```

보고서 포함 내용:

- e5 vs e10 수치 비교
- 10 epoch ASR / DINO / SSIM / DINO-SP ASR 그래프
- ASR-DINO trade-off plot
- 각 loss별 classwise ASR
- 각 loss별 대표 예시
  - 보존 성공 후보
  - 의미 변화 성공 후보
  - 공격 실패 후보
- 각 loss별 저장된 전체 grid image gallery

주의: 현재 HTML report는 `margin_dino e10 rerun1`을 포함하지 않는다. rerun 결과까지 넣으려면 `scripts/build_mixed13_uap_report.py`의 `RUNS`에 rerun을 별도 spec으로 추가하거나 comparison section을 확장해야 한다.

## 11. 현재 해석

### 왜 `margin_dino`, `margin_clip`만 Raw ASR이 높은가?

- 이 두 objective만 hinge attack loss와 semantic loss를 동시에 사용한다.
- attack signal이 명확해서 classifier decision boundary를 넘기 쉽다.
- 하지만 semantic loss는 hard constraint가 아니라 soft penalty다.
- 따라서 attack loss가 이득이면 generator가 이미지를 크게 바꾸는 방향으로 갈 수 있다.

### 왜 나머지는 실패가 많은가?

- `clip_img2img`, `dino_img2img`는 attack loss가 0이라 ASR이 낮은 것이 정상이다.
- `cr_dino`는 DINO 보존이 강하게 작동해 이미지가 거의 유지되고 공격이 약하다.
- `cr`, `untargeted_margin`은 attack term은 있지만 universal shared token이 13개 superclass 전체에 대해 공통 decision-boundary 방향을 찾지 못한 것으로 보인다.

### 현재 가장 중요한 결론

현재 실험에서는 **Raw ASR과 의미 보존이 분리되어 있다.**

- 높은 Raw ASR: `margin_dino`, `margin_clip`
- 높은 보존성: `dino only`, `margin`, `cr_dino`
- 둘 다 높은 방법: 아직 없음

따라서 다음 실험에서는 Raw ASR이 아니라 **semantic-preserved ASR**을 최적화/선택 기준으로 써야 한다.

## 12. 다음 실험 제안

### 12.1 Best checkpoint 기준 변경

현재는 마지막 epoch 결과를 본다. 앞으로는 매 epoch 또는 N step마다 test 또는 validation subset을 평가하고:

```text
SP-ASR = mean(attack_success and semantic_preserved)
Collapse Rate = mean(attack_success and not semantic_preserved)
```

를 계산해 `SP-ASR`이 가장 높은 checkpoint를 저장해야 한다.

### 12.2 Lagrangian constraint loss

고정 `semantic_loss_weight=3` 대신 adaptive Lagrangian 방식을 권장한다.

```text
attack_loss = relu(true_logit - max_other_logit + margin)

semantic_violation =
    relu(tau_dino - dino_similarity)^2
  + relu(tau_clip - clip_similarity)^2
  + relu(lpips - tau_lpips)^2

loss = attack_loss + lambda_sem * semantic_violation
```

그리고 batch마다:

```text
if semantic_violation is high:
    lambda_sem 증가
else:
    lambda_sem 유지 또는 감소
```

### 12.3 Threshold calibration

`tau_dino`, `tau_clip`, `tau_lpips`는 임의값으로 두지 말고 benign edit 분포에서 정한다.

Benign edit prompts:

```text
identity: a photo of {class_label}
mild edit: make subtle realistic changes while preserving the object identity
viewpoint: viewed from a slightly different angle
appearance: slightly different but still realistic
```

Threshold:

```text
tau_dino = benign DINO similarity 5th percentile
tau_clip = benign CLIP similarity 5th percentile
tau_lpips = benign LPIPS distance 95th percentile
```

### 12.4 Token anchor penalty

Learnable token이 generator prompt manifold 밖으로 크게 벗어나지 않도록 초기 embedding `v0`에 대한 anchor penalty를 추가한다.

```text
L_anchor = ||v - v0||^2
L_total = attack_loss + lambda_sem * semantic_violation + lambda_anchor * L_anchor
```

### 12.5 이미 성공한 샘플을 계속 공격하지 않기

Universal token에서는 일부 샘플이 이미 좋은 성공 상태인데, 나머지 샘플을 맞추려 계속 업데이트하면서 collapse가 생길 수 있다.

Per-sample rule:

```text
if attack_success and semantic_preserved:
    attack_loss = 0
    semantic_loss만 유지
else:
    attack_loss 적용
```

## 13. 재현 명령 예시

### 단일 mixed13 UAP run

```bash
python scripts/run_uap.py \
  --root outputs/uap_mixed13 \
  --imagenet-root /data/imagenet \
  --imagenet-info-root external/imagenet_hierarchy_wordnet \
  --class-mode mixed_13 \
  --victim-name resnet18_mixed13 \
  --victim-checkpoint outputs/victims/mixed13_resnet18/best.pt \
  --train-images-per-class 20 \
  --test-images-per-class 20 \
  --epochs 5 \
  --global-batch-size 16 \
  --generator-batch-size 1 \
  --num-inference-steps 4 \
  --num-tokens 32 \
  --lr 0.1 \
  --objective margin_dino \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3 \
  --wandb-mode disabled \
  --grid-save-policy representative \
  --max-saved-grids 60 \
  --image-format png \
  --image-save-workers 4 \
  --run-name mixed13_resnet18_t32_mdino_ipc20_gb16_custom
```

### e5 checkpoint에서 e10 continuation

```bash
python scripts/run_uap.py \
  --root outputs/uap_mixed13 \
  --imagenet-root /data/imagenet \
  --imagenet-info-root external/imagenet_hierarchy_wordnet \
  --class-mode mixed_13 \
  --victim-name resnet18_mixed13 \
  --victim-checkpoint outputs/victims/mixed13_resnet18/best.pt \
  --train-images-per-class 20 \
  --test-images-per-class 20 \
  --epochs 5 \
  --base-epochs 5 \
  --global-batch-size 16 \
  --generator-batch-size 1 \
  --num-inference-steps 4 \
  --num-tokens 32 \
  --lr 0.1 \
  --objective margin_dino \
  --lambda-sem 0.0 \
  --semantic-loss-weight 3 \
  --init-prompt outputs/uap_mixed13/mixed13_resnet18_t32_mdino_ipc20_gb16_e5/prompt/learned_prompt.pt \
  --wandb-mode disabled \
  --run-name mixed13_resnet18_t32_mdino_ipc20_gb16_e10_custom
```

## 14. Related References

이 프로젝트의 해석에 참고한 대표 논문:

- Universal Adversarial Perturbations: https://arxiv.org/abs/1610.08401
- Geometry of Universal Adversarial Perturbations: https://arxiv.org/abs/1705.09554
- Carlini & Wagner attack / logit-margin style objective: https://arxiv.org/abs/1608.04644
- CLIP: https://arxiv.org/abs/2103.00020
- DINOv2: https://arxiv.org/abs/2304.07193
- Textual Inversion: https://arxiv.org/abs/2208.01618

## 15. Things To Be Careful About

- `scripts/run_uap.py`라는 이름이 남아 있지만 mixed13도 지원한다.
- `clip_img2img`, `dino_img2img`는 현재 attack objective가 아니라 semantic-only objective다.
- `margin_dino`, `margin_clip`의 높은 ASR은 반드시 collapse metric과 함께 해석해야 한다.
- e10은 full scratch 10 epoch이 아니라 e5 checkpoint에서 추가 5 epoch continuation이다.
- 현재 optimizer state는 resume하지 않는다. continuation은 learned token embedding만 이어받고 새 optimizer로 시작한다.
- HTML report는 rerun 결과를 아직 포함하지 않는다.
- `README.md` 등 기존 문서 파일은 사용자가 별도로 관리해왔으므로, 문서 수정 시 git status를 반드시 확인해야 한다.
