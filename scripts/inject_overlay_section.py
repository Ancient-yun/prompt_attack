"""Inject an 'Overlay 진단' section (with embedded images) into the HTML report."""

from __future__ import annotations

import base64
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "docs" / "mixed13_results_summary.html"
RUN = REPO / "outputs/uap_mixed13/mixed13_resnet18_t32_mdino_ipc20_gb16_e10"


def data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def main() -> None:
    mean_resid = data_uri(RUN / "overlay_diagnostic/mean_residual_norm.png")
    grid_boat = data_uri(RUN / "grids/test_boat_ILSVRC2012_val_00000347.png")
    grid_truck = data_uri(RUN / "grids/test_truck_ILSVRC2012_val_00000784.png")

    section = f"""<h2>5. Overlay 진단 — 학습된 토큰의 정체</h2>
<p>§4의 "overlay 가설"을 <strong>이미 저장된 adv 이미지만으로 검증</strong>했다(FLUX 재생성 불필요,
<code>scripts/overlay_diagnostic.py</code>). 세 갈래 증거가 모두 같은 결론을 가리킨다:
<strong>margin_dino 토큰은 semantic edit이 아니라 고정된 universal 적대적 오버레이다.</strong></p>

<div class="flag bad"><strong>증거 1 · Dominant target.</strong> 성공 214건 중 <strong>137건(64%)이 "computer"로 붕괴</strong>하고,
12개 참 클래스 전부의 1순위 오분류가 computer다(fish만 예외). 입력 내용과 무관하게 한 방향으로 미는 UAP의 서명.</div>

<h3>증거 2 · 잔차 결맞음 (residual coherence)</h3>
<p class="muted">서로 다른 클래스 160장의 <code>adv − original</code> 잔차를 비교. overlay라면 잔차들이 같은 방향을 가리키고
(상호 코사인↑), 하나의 공통 패턴이 변화 에너지의 대부분을 설명한다(공통패턴 에너지↑).</p>
<div class="tablewrap"><table>
<thead><tr><th>run</th><th>ASR</th><th>잔차 상호코사인</th><th>공통패턴 에너지</th><th>판정</th></tr></thead>
<tbody>
<tr><td>margin_dino e10</td><td>82.3%</td><td class="bad">+0.724</td><td class="bad">71.8%</td><td><span class="pill high">명백한 overlay</span></td></tr>
<tr><td>margin_clip e10</td><td>80.0%</td><td class="warn">+0.304</td><td class="warn">33.9%</td><td><span class="pill mid">약한/파괴적 overlay</span></td></tr>
<tr><td>dino (보존형) e10</td><td>1.9%</td><td class="good">+0.119</td><td class="good">13.4%</td><td><span class="pill low">overlay 없음 (바닥값)</span></td></tr>
</tbody>
</table></div>
<p class="muted">보존형의 13.4%가 "overlay 없음" 바닥값(FLUX 자체 재구성 노이즈의 약한 공유분). margin_dino는 그 5배 이상.</p>

<h3>증거 3 · 오버레이의 실체를 눈으로</h3>
<div class="charts" style="grid-template-columns:1fr 1fr">
  <div class="chart">
    <h4>평균 잔차 (160장, 정규화)</h4>
    <img src="{mean_resid}" alt="mean residual" style="width:100%;border-radius:8px;display:block">
    <div class="legend" style="display:block">전 픽셀 ~40% 균일 어둡게 + 얼룩진 <strong>청보라 "모니터 유리/스크린" 텍스처</strong>.
    computer가 dominant target인 것과 정확히 일치 — FLUX가 모든 이미지에 이 화면 질감을 덧칠한다.</div>
  </div>
  <div class="chart">
    <h4>실제 붕괴 예시 (원본 | adv)</h4>
    <img src="{grid_boat}" alt="boat to computer" style="width:100%;border-radius:8px;display:block;margin-bottom:8px">
    <img src="{grid_truck}" alt="truck to computer" style="width:100%;border-radius:8px;display:block">
    <div class="legend" style="display:block">boat→computer (SSIM 0.003, DINO 0.893) · truck→computer (SSIM 0.047, DINO 0.907).
    <strong>DINO는 0.9로 버티는데 SSIM은 0에 가깝다</strong> — semantic 인코더가 오버레이에 눈먼 것을 그대로 보여준다.</div>
  </div>
</div>

<div class="flag bad"><strong>함의.</strong> 82% ASR은 "자연스러운 semantic 공격"이 아니라 생성기를 배달수단으로 쓴
<strong>고전적 UAP</strong>다. DINO/CLIP 공간의 제약은 이 오버레이를 원리적으로 못 본다 →
<code>semantic_loss_weight</code> 튜닝은 무의미. 다음 실험은 "가중치 조정"이 아니라
<strong>"오버레이를 깰 수 있는가"</strong>라는 질문이어야 한다.</div>

"""

    html = HTML.read_text(encoding="utf-8")
    anchor = "<h2>5. 다음 방향</h2>"
    if anchor not in html:
        raise SystemExit("anchor '5. 다음 방향' not found; already injected?")
    html = html.replace(anchor, section + "<h2>6. 다음 방향</h2>")
    # bump the trailing footer's section reference if any and fix stat card wording
    HTML.write_text(html, encoding="utf-8")
    print(f"Injected Overlay 진단 section into {HTML}")


if __name__ == "__main__":
    main()
