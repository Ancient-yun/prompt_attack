# 향후 실험 계획

Updated: 2026-05-28 KST

## 목표

다음 실험의 핵심 질문은 다음입니다.

> 하나의 shared learnable textual-inversion token set으로 ImageNet classifier를 공격하면서, 생성된 이미지가 원본 객체 의미를 유지할 수 있는가?

기존 CR-only 실험은 공격 가능성은 보였지만 이미지 의미가 크게 바뀌는 경우가 많았습니다. 기존 `cr_dino` 실험은 단순 weighted sum에서 CR loss scale이 DINO loss를 압도해 semantic collapse가 발생했습니다.

따라서 다음 단계는 **CE/CR 기반 공격을 계속 세게 미는 방식이 아니라, margin hinge로 decision boundary만 넘기고 image-to-image similarity를 연속 loss로 유지하는 방식**을 비교합니다.

## 1. 핵심 변경점

이전 방식:

```text
attack_loss = relu(true_logit - max_other_logit + attack_margin)
semantic_penalty = relu(semantic_threshold - image_similarity)^2
total_loss = attack_loss + semantic_penalty_weight * semantic_penalty
```

새 방식:

```text
attack_loss = relu(true_logit - max_other_logit + attack_margin)
semantic_loss = 1 - image_similarity(original, adv)
total_loss = attack_loss + semantic_loss_weight * semantic_loss
```

바뀌는 점:

- `semantic_threshold`를 제거합니다.
- semantic success / semantic ASR 같은 binary 지표를 제거합니다.
- DINO, CLIP-I, SSIM은 DreamBooth 계열 논문처럼 연속 metric으로 보고합니다.
- attack 성공 여부는 ASR로 보고, semantic preservation은 평균 similarity와 qualitative image grid로 봅니다.

## 2. 논문 기반 Metric 해석

DINO/CLIP-I 계열 논문들은 보통 hard threshold를 정하지 않고, cosine similarity를 연속 metric으로 보고합니다.

가장 가까운 기준은 DreamBooth 계열 subject fidelity 평가입니다.

- DreamBooth는 subject fidelity를 DINO와 CLIP-I로 평가합니다.
- DINO는 generated image와 real/reference image의 DINO embedding cosine similarity 평균입니다.
- CLIP-I는 generated image와 real/reference image의 CLIP image embedding cosine similarity 평균입니다.
- DreamBooth CVPR 2023 Table 1 기준:
  - Real Images: DINO `0.774`, CLIP-I `0.885`
  - DreamBooth Stable Diffusion: DINO `0.668`, CLIP-I `0.803`
  - Textual Inversion Stable Diffusion: DINO `0.569`, CLIP-I `0.780`

Sources:

- DreamBooth, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/papers/Ruiz_DreamBooth_Fine_Tuning_Text-to-Image_Diffusion_Models_for_Subject-Driven_Generation_CVPR_2023_paper.pdf
- DreamBlend, WACV 2025: https://openaccess.thecvf.com/content/WACV2025/papers/Ram_DreamBlend_Advancing_Personalized_Fine-Tuning_of_Text-to-Image_Diffusion_Models_WACV_2025_paper.pdf

해석:

- DINO `0.85` 같은 cutoff는 논문 기준으로 직접 가져온 값이 아니므로 사용하지 않습니다.
- CLIP-I와 DINO는 서로 다른 embedding space라 절대값을 직접 비교하지 않습니다.
- 같은 metric 안에서 baseline 대비 개선/악화를 봅니다.

## 3. 1차 비교: DINO img2img vs CLIP img2img

고정 설정:

```text
num_tokens: 32
initializer: random_real_tokens 또는 현재 best initializer
attack_margin: 0
lr: 0.1
num_inference_steps: 4
train: fixed10 full train clean-correct subset
test: fixed10 full val clean-correct subset
```

비교할 objective:

```text
margin_dino
margin_clip_img2img
```

Loss:

```text
margin_dino:
  image_similarity = cosine(DINO(original), DINO(adv))
  total_loss = margin_attack_loss + semantic_loss_weight * (1 - image_similarity)

margin_clip_img2img:
  image_similarity = cosine(CLIP_image(original), CLIP_image(adv))
  total_loss = margin_attack_loss + semantic_loss_weight * (1 - image_similarity)
```

초기 비교값:

```text
semantic_loss_weight: 3, 10
```

목적:

- DINO가 원본 객체의 세부 visual identity를 더 잘 유지하는지 확인합니다.
- CLIP image-to-image가 generator의 semantic drift를 더 잘 막는지 확인합니다.
- 둘 중 어느 쪽이 ASR과 semantic preservation의 trade-off가 좋은지 봅니다.

## 4. 2차 비교: Initializer Sweep

1차에서 더 좋은 objective를 고정하고 initializer를 비교합니다.

```text
object
random_real_tokens
fixed10_class_average
```

목적:

- `object`: 기존 textual-inversion 스타일 초기화가 안정적인지 확인
- `random_real_tokens`: 다양한 실제 token embedding에서 시작하는 것이 search에 유리한지 확인
- `fixed10_class_average`: fixed10 클래스 의미를 평균낸 초기화가 semantic preservation에 유리한지 확인

## 5. 3차 비교: Token Count Sweep

best objective와 best initializer를 고정한 뒤 token 수를 비교합니다.

```text
num_tokens: 16, 32, 64
```

이유:

- 기존 CR 결과에서 token32는 공격이 강했지만 semantic drift가 컸습니다.
- token64는 의미 보존은 상대적으로 좋았지만 공격이 약했습니다.
- margin + image-to-image loss에서 token32와 token64 중 어느 쪽이 더 좋은 capacity인지 확인합니다.

## 6. 선택적 후속 실험

CLIP img2img가 DINO보다 좋지만 class 자체가 바뀌는 문제가 남으면, CLIP source-text term을 추가합니다.

```text
source_text = "a photo of {class_label}"
clip_text_sim = cosine(CLIP_image(adv), CLIP_text(source_text))
total_loss = margin_attack_loss
           + image_weight * (1 - CLIP_image_similarity)
           + text_weight * (1 - CLIP_text_similarity)
```

이 실험은 CLIP img2img 단독 결과를 본 뒤에만 진행합니다.

## 7. 성공 기준

기존 B200 결과를 기준선으로 봅니다.

```text
CR token32: ASR은 높지만 DINO similarity가 낮음
CR token64: ASR은 낮지만 DINO/SSIM이 상대적으로 좋음
CR+DINO lambda=0.5: ASR은 높지만 unrelated image로 collapse
```

좋은 설정의 기준:

- 기존 CR token64보다 ASR이 높아야 합니다.
- 기존 CR token32보다 DINO/SSIM 또는 CLIP-I가 좋아야 합니다.
- 하나의 adversarial label로 collapse되면 안 됩니다.
- qualitative example에서 실제 semantic-preserving success가 보여야 합니다.

## 8. 결과 정리 방식

각 run마다 기록할 항목:

- objective
- semantic model
- initializer
- token count
- learning rate
- semantic loss weight
- ASR
- DINO similarity
- CLIP image similarity
- SSIM
- top adversarial label distribution
- success / failure / semantic failure image grids

랩미팅 자료에서는 다음 순서로 정리합니다.

1. 기존 CR 및 CR+DINO 실패 패턴
2. `margin_dino` 결과
3. `margin_clip_img2img` 결과
4. initializer ablation
5. token count ablation
6. best qualitative examples와 remaining failure cases
