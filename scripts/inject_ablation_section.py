"""Inject the anti-overlay ablation section (B200 runs) into the HTML report."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HTML = REPO / "docs" / "mixed13_results_summary.html"

SECTION = """<h2>6. Anti-overlay ablation (B200)</h2>
<p>진단으로 overlay 가설을 확정한 뒤, overlay를 깨기 위한 세 개입을 분리 검증했다. 세 run 모두
동일 설정(mixed_13, ipc20, gb16, e5)에서 <strong>margin_dino baseline과 한 요소씩만 다르게</strong>
두고 B200에서 병렬 실행했다. LPIPS = 미분 가능한 픽셀/지각 앵커, EOT = 매 스텝 diffusion seed 재샘플.</p>

<div class="tablewrap"><table>
<thead><tr><th>run</th><th>개입</th><th>ASR</th><th>SSIM</th><th>DINO</th><th>잔차 코사인</th><th>overlay?</th></tr></thead>
<tbody>
<tr><td>baseline</td><td>margin_dino</td><td><span class="pill high">82.3%</span></td><td class="bad">0.025</td><td>0.737</td><td class="bad">0.724</td><td><span class="pill high">있음</span></td></tr>
<tr><td>C</td><td>+ EOT만 (t32)</td><td><span class="pill high">85.0%</span></td><td class="bad">0.029</td><td>0.346</td><td class="bad">0.661</td><td><span class="pill high">그대로</span></td></tr>
<tr><td>B</td><td>+ LPIPS만 (t32)</td><td><span class="pill low">3.1%</span></td><td class="warn">0.550</td><td>0.674</td><td class="good">0.067</td><td><span class="pill low">사라짐</span></td></tr>
<tr><td>A</td><td>+ LPIPS + EOT + t8</td><td><span class="pill low">1.9%</span></td><td class="good">0.851</td><td>0.885</td><td class="good">0.066</td><td><span class="pill low">사라짐</span></td></tr>
</tbody>
</table></div>
<p class="muted">잔차 코사인 = 서로 다른 160장의 <code>adv−original</code> 상호 코사인(높을수록 공통 overlay). baseline·C ≈ 0.7(overlay), A·B ≈ 0.07(overlay 없음, floor 0.13보다도 낮음).</p>

<h3>세 가지 확정 결론</h3>
<div class="flag bad"><strong>1. EOT는 무효.</strong> C는 seed를 매 스텝 랜덤화했는데도 잔차 0.66 · ASR 85% · SSIM 0.03으로
overlay가 그대로다. overlay는 "특정 diffusion 경로 암기"가 아니라 <strong>강건한 universal 텍스처</strong>라
randomization으로 안 깨진다.</div>
<div class="flag"><strong>2. overlay를 깨는 건 LPIPS.</strong> LPIPS가 든 A·B만 잔차 코사인이 0.07로 붕괴(overlay 제거).
픽셀 앵커가 overlay를 확실히 억제한다.</div>
<div class="flag warn"><strong>3. 그러나 overlay를 억제하면 공격도 죽는다.</strong> A·B의 ASR은 2–3%.
<strong>{LPIPS, EOT, 용량축소} 어떤 조합으로도 "보존 + 속이기"를 동시에 달성 못 함</strong> — LPIPS는
반대편 코너(보존됨·공격 안 됨)로 옮겼을 뿐이다.</div>

<div class="flag bad"><strong>함의 (다음 방향의 근거).</strong> 픽셀 제약(LPIPS)은 너무 뭉툭하다 — 나쁜 overlay뿐 아니라
<em>좋은 semantic 편집(포즈·시점 변화, 픽셀은 크게 움직이지만 정체성 유지)까지</em> 전부 막는다. 따라서
"정체성 유지 + classifier 속이기"는 픽셀 제약으로 불가능하며, 변화가 <strong>자연 이미지 manifold 위의
semantic 변형</strong>이어야 하고 정체성은 <strong>별도 oracle</strong>(예: CLIP zero-shot)로 지켜야 한다.
이 ablation은 그 방향을 뒷받침하는 깔끔한 negative result다.</div>

"""


def main() -> None:
    html = HTML.read_text(encoding="utf-8")
    anchor = "<h2>6. 다음 방향</h2>"
    if anchor not in html:
        raise SystemExit("anchor '6. 다음 방향' not found (already injected or renumbered?)")
    html = html.replace(anchor, SECTION + "<h2>7. 다음 방향</h2>")
    HTML.write_text(html, encoding="utf-8")
    print(f"Injected ablation section into {HTML}")


if __name__ == "__main__":
    main()
