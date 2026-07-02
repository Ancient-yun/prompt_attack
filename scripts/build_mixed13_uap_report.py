"""Build an HTML report for mixed13 UAP experiments."""

from __future__ import annotations

import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path("outputs/uap_mixed13")
REPORT_DIR = Path("outputs/uap_mixed13_report")
REPORT_PATH = REPORT_DIR / "mixed13_uap_loss_report.html"


@dataclass(frozen=True)
class RunSpec:
    label: str
    short: str
    run_e5: str
    run_e10: str


RUNS = [
    RunSpec(
        label="Margin + DINO",
        short="margin_dino",
        run_e5="mixed13_resnet18_t32_mdino_ipc20_gb16_e5",
        run_e10="mixed13_resnet18_t32_mdino_ipc20_gb16_e10",
    ),
    RunSpec(
        label="CR",
        short="cr",
        run_e5="mixed13_resnet18_t32_cr_ipc20_gb16_e5",
        run_e10="mixed13_resnet18_t32_cr_ipc20_gb16_e10",
    ),
    RunSpec(
        label="CR + DINO",
        short="cr_dino",
        run_e5="mixed13_resnet18_t32_crdino_lam0p5_ipc20_gb16_e5",
        run_e10="mixed13_resnet18_t32_crdino_lam0p5_ipc20_gb16_e10",
    ),
    RunSpec(
        label="Margin",
        short="margin",
        run_e5="mixed13_resnet18_t32_margin_ipc20_gb16_e5",
        run_e10="mixed13_resnet18_t32_margin_ipc20_gb16_e10",
    ),
    RunSpec(
        label="Margin + CLIP",
        short="margin_clip",
        run_e5="mixed13_resnet18_t32_mclip_ipc20_gb16_e5",
        run_e10="mixed13_resnet18_t32_mclip_ipc20_gb16_e10",
    ),
    RunSpec(
        label="CLIP only",
        short="clip",
        run_e5="mixed13_resnet18_t32_clip_ipc20_gb16_e5",
        run_e10="mixed13_resnet18_t32_clip_ipc20_gb16_e10",
    ),
    RunSpec(
        label="DINO only",
        short="dino",
        run_e5="mixed13_resnet18_t32_dino_ipc20_gb16_e5",
        run_e10="mixed13_resnet18_t32_dino_ipc20_gb16_e10",
    ),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_summary(run: str) -> dict[str, Any]:
    return read_json(ROOT / run / "metrics" / "summary.json")


def read_results(run: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / run / "metrics" / "test_results.csv")
    bool_cols = ["success", "clean_correct", "grid_saved"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1"})
    numeric_cols = [
        "dino_similarity",
        "clip_image_similarity",
        "semantic_similarity",
        "ssim",
        "confidence_drop",
        "margin_drop",
        "adv_true_conf",
        "clean_true_conf",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def pct(value: float | None) -> str:
    if value is None or math.isnan(float(value)):
        return "-"
    return f"{float(value) * 100:.1f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        if math.isnan(float(value)):
            return "-"
    except TypeError:
        return "-"
    return f"{float(value):.{digits}f}"


def minutes(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    return f"{seconds / 60:.1f}m"


def rel(path: str | Path) -> str:
    p = Path(path)
    abs_path = p if p.is_absolute() else (Path.cwd() / p)
    abs_report_dir = Path.cwd() / REPORT_DIR
    return Path(os.path.relpath(abs_path.resolve(), abs_report_dir.resolve())).as_posix()


def run_row(spec: RunSpec, epoch_label: str, summary: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    test = summary["test"]
    dino = pd.to_numeric(df.get("dino_similarity"), errors="coerce")
    ssim = pd.to_numeric(df.get("ssim"), errors="coerce")
    success = df["success"].astype(bool)
    dino_sp70 = float((success & (dino >= 0.70)).mean()) if len(df) else 0.0
    dino_sp80 = float((success & (dino >= 0.80)).mean()) if len(df) else 0.0
    ssim_sp50 = float((success & (ssim >= 0.50)).mean()) if len(df) else 0.0
    return {
        "label": spec.label,
        "short": spec.short,
        "epoch": epoch_label,
        "run_name": summary["run_name"],
        "objective": summary["objective"],
        "asr": float(test["asr"]),
        "success_count": int(test["success_count"]),
        "count": int(test["count"]),
        "dino": float(test.get("mean_dino_similarity") or 0.0),
        "clip": test.get("mean_clip_image_similarity"),
        "ssim": float(test.get("mean_ssim") or 0.0),
        "conf_drop": test.get("mean_confidence_drop"),
        "margin_drop": test.get("mean_decision_logit_gap_drop"),
        "dino_sp70": dino_sp70,
        "dino_sp80": dino_sp80,
        "ssim_sp50": ssim_sp50,
        "elapsed": float(summary.get("elapsed_seconds", 0.0)),
    }


def bar(value: float, *, scale: float = 1.0, color: str = "var(--accent)") -> str:
    width = max(0.0, min(100.0, value / scale * 100.0))
    return (
        f'<div class="bar"><span style="width:{width:.1f}%;background:{color}"></span></div>'
        f'<b>{pct(value)}</b>'
    )


def metric_chart(rows: list[dict[str, Any]], key: str, title: str, color: str) -> str:
    items = []
    for row in rows:
        label = html.escape(row["label"])
        value = float(row[key])
        items.append(
            f"""
            <div class="chart-row">
              <div class="chart-label">{label}</div>
              <div class="chart-bar"><span style="width:{value * 100:.1f}%;background:{color}"></span></div>
              <div class="chart-value">{pct(value)}</div>
            </div>
            """
        )
    return f'<section class="panel"><h2>{html.escape(title)}</h2><div class="chart">{"".join(items)}</div></section>'


def delta_chart(rows5: list[dict[str, Any]], rows10: list[dict[str, Any]]) -> str:
    by_short_5 = {r["short"]: r for r in rows5}
    items = []
    for r10 in rows10:
        r5 = by_short_5[r10["short"]]
        delta = r10["asr"] - r5["asr"]
        cls = "pos" if delta >= 0 else "neg"
        items.append(
            f"""
            <tr>
              <td>{html.escape(r10["label"])}</td>
              <td>{pct(r5["asr"])}</td>
              <td>{pct(r10["asr"])}</td>
              <td class="{cls}">{delta * 100:+.1f} pp</td>
              <td>{num(r5["dino"])} → {num(r10["dino"])}</td>
              <td>{num(r5["ssim"])} → {num(r10["ssim"])}</td>
            </tr>
            """
        )
    return f"""
    <section class="panel">
      <h2>5 epoch → 10 epoch 변화</h2>
      <table>
        <thead><tr><th>Loss</th><th>ASR@5e</th><th>ASR@10e</th><th>ΔASR</th><th>DINO</th><th>SSIM</th></tr></thead>
        <tbody>{''.join(items)}</tbody>
      </table>
    </section>
    """


def scatter_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 720, 380
    pad_l, pad_t, pad_r, pad_b = 74, 36, 24, 62
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def x(v: float) -> float:
        return pad_l + v * plot_w

    def y(v: float) -> float:
        return pad_t + (1.0 - v) * plot_h

    pts = []
    palette = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed", "#475569"]
    for i, row in enumerate(rows):
        cx, cy = x(row["dino"]), y(row["asr"])
        color = palette[i % len(palette)]
        pts.append(
            f'<g><circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{color}" />'
            f'<text x="{cx + 12:.1f}" y="{cy + 4:.1f}">{html.escape(row["label"])}</text></g>'
        )
    return f"""
    <section class="panel">
      <h2>ASR-DINO Trade-off</h2>
      <svg class="scatter" viewBox="0 0 {width} {height}" role="img" aria-label="ASR DINO scatter">
        <line x1="{pad_l}" y1="{pad_t + plot_h}" x2="{pad_l + plot_w}" y2="{pad_t + plot_h}" stroke="#94a3b8" />
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + plot_h}" stroke="#94a3b8" />
        <text x="{width / 2}" y="{height - 16}" text-anchor="middle">Mean DINO similarity, higher is more similar</text>
        <text transform="translate(18 {height / 2}) rotate(-90)" text-anchor="middle">ASR, higher is stronger attack</text>
        <text x="{pad_l}" y="{pad_t + plot_h + 22}">0.0</text>
        <text x="{pad_l + plot_w - 20}" y="{pad_t + plot_h + 22}">1.0</text>
        <text x="{pad_l - 42}" y="{pad_t + plot_h + 4}">0%</text>
        <text x="{pad_l - 50}" y="{pad_t + 4}">100%</text>
        {''.join(pts)}
      </svg>
      <p class="note">오른쪽 위가 이상적입니다. 현재는 높은 ASR run이 왼쪽 또는 낮은 SSIM 쪽으로 이동해 의미 변화가 커지는 경향이 있습니다.</p>
    </section>
    """


def classwise_block(run: str) -> str:
    df = read_results(run)
    grouped = (
        df.groupby("class_label", observed=True)
        .agg(count=("success", "size"), success=("success", "sum"), dino=("dino_similarity", "mean"), ssim=("ssim", "mean"))
        .reset_index()
    )
    grouped["asr"] = grouped["success"] / grouped["count"]
    grouped = grouped.sort_values(["asr", "class_label"], ascending=[False, True])
    rows = []
    for _, r in grouped.iterrows():
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(r["class_label"]))}</td>
              <td>{int(r["success"])}/{int(r["count"])}</td>
              <td>{bar(float(r["asr"]), color="var(--accent-2)")}</td>
              <td>{num(float(r["dino"]))}</td>
              <td>{num(float(r["ssim"]))}</td>
            </tr>
            """
        )
    return f"""
    <details class="classwise">
      <summary>클래스별 ASR 보기</summary>
      <table>
        <thead><tr><th>Class</th><th>Success</th><th>ASR</th><th>DINO</th><th>SSIM</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </details>
    """


def row_path_exists(row: pd.Series) -> bool:
    path = row.get("grid_image_path")
    if not isinstance(path, str) or not path:
        return False
    return Path(path).exists()


def select_examples(df: pd.DataFrame, *, kind: str, n: int = 3) -> pd.DataFrame:
    d = df.copy()
    d = d[d.apply(row_path_exists, axis=1)].copy()
    if d.empty:
        return d
    d["dino_fill"] = d["dino_similarity"].fillna(d["semantic_similarity"]).fillna(0.0)
    d["ssim_fill"] = d["ssim"].fillna(0.0)
    if kind == "preserved_success":
        d = d[d["success"]].copy()
        d["score"] = d["dino_fill"] + 0.5 * d["ssim_fill"]
        return d.sort_values(["score", "confidence_drop"], ascending=[False, False]).head(n)
    if kind == "changed_success":
        d = d[d["success"]].copy()
        d["score"] = d["dino_fill"] + 0.5 * d["ssim_fill"]
        return d.sort_values(["score", "confidence_drop"], ascending=[True, False]).head(n)
    if kind == "failure":
        d = d[~d["success"]].copy()
        d["score"] = d["dino_fill"] + 0.5 * d["ssim_fill"]
        return d.sort_values(["score", "clean_true_conf"], ascending=[False, False]).head(n)
    raise ValueError(kind)


def example_card(row: pd.Series) -> str:
    path = rel(row["grid_image_path"])
    title = f'{row["class_label"]}: {row["clean_pred_label"]} → {row["adv_pred_label"]}'
    status = "success" if bool(row["success"]) else "failure"
    return f"""
    <figure class="example-card {status}">
      <img loading="lazy" src="{html.escape(path)}" alt="{html.escape(title)}">
      <figcaption>
        <b>{html.escape(title)}</b>
        <span>DINO {num(row.get("dino_similarity"))} · SSIM {num(row.get("ssim"))} · conf drop {num(row.get("confidence_drop"))}</span>
      </figcaption>
    </figure>
    """


def examples_block(run: str) -> str:
    df = read_results(run)
    groups = [
        ("보존 성공 후보", "preserved_success"),
        ("의미 변화 성공 후보", "changed_success"),
        ("공격 실패 후보", "failure"),
    ]
    sections = []
    for title, kind in groups:
        ex = select_examples(df, kind=kind)
        if ex.empty:
            body = '<p class="note">해당 조건으로 저장된 grid가 없습니다.</p>'
        else:
            body = '<div class="examples">' + "".join(example_card(row) for _, row in ex.iterrows()) + "</div>"
        sections.append(f"<h4>{title}</h4>{body}")

    grid_dir = ROOT / run / "grids"
    all_imgs = sorted(grid_dir.glob("*.png")) if grid_dir.exists() else []
    gallery = "".join(
        f'<a href="{html.escape(rel(p))}" target="_blank"><img loading="lazy" src="{html.escape(rel(p))}" alt="{html.escape(p.stem)}"></a>'
        for p in all_imgs
    )
    if not gallery:
        gallery = '<p class="note">저장된 grid 이미지가 없습니다.</p>'
    return f"""
    <div class="example-section">
      {''.join(sections)}
      <details>
        <summary>저장된 전체 grid 이미지 {len(all_imgs)}장 보기</summary>
        <div class="gallery">{gallery}</div>
      </details>
    </div>
    """


def run_section(spec: RunSpec, row10: dict[str, Any], row5: dict[str, Any]) -> str:
    return f"""
    <section class="panel run-panel" id="{html.escape(spec.short)}">
      <div class="run-head">
        <div>
          <h2>{html.escape(spec.label)}</h2>
          <p class="mono">{html.escape(row10["run_name"])}</p>
        </div>
        <div class="scorecards small">
          <div><b>{pct(row10["asr"])}</b><span>ASR</span></div>
          <div><b>{num(row10["dino"])}</b><span>DINO ↑</span></div>
          <div><b>{num(row10["ssim"])}</b><span>SSIM ↑</span></div>
        </div>
      </div>
      <table>
        <thead><tr><th>Metric</th><th>5 epoch</th><th>10 epoch</th></tr></thead>
        <tbody>
          <tr><td>ASR</td><td>{pct(row5["asr"])} ({row5["success_count"]}/{row5["count"]})</td><td>{pct(row10["asr"])} ({row10["success_count"]}/{row10["count"]})</td></tr>
          <tr><td>DINO similarity ↑</td><td>{num(row5["dino"])}</td><td>{num(row10["dino"])}</td></tr>
          <tr><td>CLIP image similarity ↑</td><td>{num(row5["clip"])}</td><td>{num(row10["clip"])}</td></tr>
          <tr><td>SSIM ↑</td><td>{num(row5["ssim"])}</td><td>{num(row10["ssim"])}</td></tr>
          <tr><td>DINO-SP ASR@0.70 ↑</td><td>{pct(row5["dino_sp70"])}</td><td>{pct(row10["dino_sp70"])}</td></tr>
          <tr><td>SSIM-preserved ASR@0.50 ↑</td><td>{pct(row5["ssim_sp50"])}</td><td>{pct(row10["ssim_sp50"])}</td></tr>
          <tr><td>Mean confidence drop ↑</td><td>{num(row5["conf_drop"])}</td><td>{num(row10["conf_drop"])}</td></tr>
          <tr><td>Elapsed</td><td>{minutes(row5["elapsed"])}</td><td>{minutes(row10["elapsed"])}</td></tr>
        </tbody>
      </table>
      {classwise_block(spec.run_e10)}
      {examples_block(spec.run_e10)}
    </section>
    """


def summary_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in sorted(rows, key=lambda r: r["asr"], reverse=True):
        body.append(
            f"""
            <tr>
              <td><a href="#{html.escape(row["short"])}">{html.escape(row["label"])}</a></td>
              <td>{html.escape(row["objective"])}</td>
              <td>{row["success_count"]}/{row["count"]}</td>
              <td>{bar(row["asr"], color="var(--danger)")}</td>
              <td>{num(row["dino"])}</td>
              <td>{num(row["clip"])}</td>
              <td>{num(row["ssim"])}</td>
              <td>{pct(row["dino_sp70"])}</td>
              <td>{pct(row["ssim_sp50"])}</td>
            </tr>
            """
        )
    return f"""
    <section class="panel">
      <h2>10 epoch 결과 수치 비교</h2>
      <table>
        <thead>
          <tr>
            <th>Loss</th><th>Objective</th><th>Success</th><th>ASR ↑</th>
            <th>DINO ↑</th><th>CLIP ↑</th><th>SSIM ↑</th><th>DINO-SP ASR@0.70 ↑</th><th>SSIM ASR@0.50 ↑</th>
          </tr>
        </thead>
        <tbody>{''.join(body)}</tbody>
      </table>
      <p class="note">↑는 값이 클수록 원본과 비슷하거나 공격이 강하다는 뜻입니다. ASR과 보존 지표는 동시에 높아야 좋은 결과입니다.</p>
    </section>
    """


def build_report() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows5: list[dict[str, Any]] = []
    rows10: list[dict[str, Any]] = []
    for spec in RUNS:
        s5 = read_summary(spec.run_e5)
        s10 = read_summary(spec.run_e10)
        df5 = read_results(spec.run_e5)
        df10 = read_results(spec.run_e10)
        rows5.append(run_row(spec, "5 epoch", s5, df5))
        rows10.append(run_row(spec, "10 epoch", s10, df10))

    best_asr = max(rows10, key=lambda r: r["asr"])
    best_preserve = max(rows10, key=lambda r: (r["dino"] + r["ssim"]) / 2)
    best_sp = max(rows10, key=lambda r: r["dino_sp70"])
    generated = "2026-07-01"

    sections = []
    by_short_5 = {r["short"]: r for r in rows5}
    for spec in RUNS:
        sections.append(run_section(spec, next(r for r in rows10 if r["short"] == spec.short), by_short_5[spec.short]))

    css = """
    :root {
      --bg: #f8fafc;
      --panel: #ffffff;
      --ink: #0f172a;
      --muted: #64748b;
      --line: #dbe4ee;
      --accent: #2563eb;
      --accent-2: #0891b2;
      --danger: #ef4444;
      --ok: #16a34a;
      --warn: #ca8a04;
      font-family: Inter, Pretendard, "Noto Sans KR", "Malgun Gothic", Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); }
    .hero { padding: 44px 48px 28px; background: linear-gradient(135deg, #0f172a, #1e293b); color: white; }
    .hero h1 { margin: 0 0 12px; font-size: 34px; letter-spacing: 0; }
    .hero p { margin: 0; max-width: 980px; line-height: 1.65; color: #cbd5e1; }
    main { padding: 28px 48px 72px; max-width: 1440px; margin: 0 auto; }
    .scorecards { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin: 22px 0; }
    .scorecards div { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.14); border-radius: 10px; padding: 16px; }
    .scorecards.small { grid-template-columns: repeat(3, 130px); margin: 0; }
    .scorecards.small div { background: #f8fafc; border-color: var(--line); color: var(--ink); }
    .scorecards b { display: block; font-size: 26px; margin-bottom: 4px; }
    .scorecards span { color: inherit; opacity: .72; font-size: 13px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 22px; margin: 20px 0; box-shadow: 0 12px 30px rgba(15, 23, 42, .06); }
    h2 { margin: 0 0 16px; font-size: 22px; }
    h3 { margin: 22px 0 10px; font-size: 18px; }
    h4 { margin: 18px 0 10px; font-size: 15px; color: #334155; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: middle; }
    th { color: #475569; background: #f8fafc; font-weight: 700; }
    tr:hover td { background: #fbfdff; }
    a { color: var(--accent); text-decoration: none; }
    .note { color: var(--muted); line-height: 1.6; margin: 12px 0 0; }
    .mono { font-family: "Cascadia Mono", Consolas, monospace; color: var(--muted); font-size: 12px; margin: 4px 0 0; }
    .bar { display: inline-block; width: 92px; height: 9px; background: #e2e8f0; border-radius: 99px; overflow: hidden; margin-right: 8px; vertical-align: middle; }
    .bar span { display: block; height: 100%; }
    .chart-row { display: grid; grid-template-columns: 170px 1fr 64px; gap: 12px; align-items: center; margin: 10px 0; }
    .chart-label { font-weight: 700; }
    .chart-bar { height: 18px; background: #e2e8f0; border-radius: 999px; overflow: hidden; }
    .chart-bar span { display: block; height: 100%; }
    .chart-value { color: var(--muted); font-variant-numeric: tabular-nums; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .scatter { width: 100%; max-width: 900px; height: auto; }
    .scatter text { font-size: 13px; fill: #334155; }
    .pos { color: var(--ok); font-weight: 700; }
    .neg { color: var(--danger); font-weight: 700; }
    .run-head { display: flex; justify-content: space-between; gap: 24px; align-items: start; border-bottom: 1px solid var(--line); padding-bottom: 16px; margin-bottom: 16px; }
    details { margin-top: 16px; }
    summary { cursor: pointer; font-weight: 700; color: #334155; }
    .examples { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .example-card { margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #fff; }
    .example-card img { display: block; width: 100%; height: auto; background: #e2e8f0; }
    .example-card figcaption { padding: 10px 12px; font-size: 12px; line-height: 1.45; }
    .example-card figcaption b { display: block; margin-bottom: 3px; }
    .example-card figcaption span { color: var(--muted); }
    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 10px; margin-top: 12px; }
    .gallery img { width: 100%; display: block; border-radius: 8px; border: 1px solid var(--line); background: #e2e8f0; }
    .analysis { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
    .analysis article { border: 1px solid var(--line); border-radius: 12px; padding: 16px; background: #fbfdff; }
    .analysis b { display: block; margin-bottom: 8px; }
    .analysis p { color: var(--muted); line-height: 1.55; margin: 0; }
    @media (max-width: 900px) {
      .hero, main { padding-left: 18px; padding-right: 18px; }
      .scorecards, .grid2, .analysis, .examples { grid-template-columns: 1fr; }
      .run-head { flex-direction: column; }
    }
    """

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mixed13 UAP Loss Sweep Report</title>
  <style>{css}</style>
</head>
<body>
  <header class="hero">
    <h1>Mixed13 UAP Loss Sweep 결과 보고서</h1>
    <p>
      FLUX.2-klein-4B learnable textual-inversion token UAP를 mixed13 ResNet-18 victim에 대해 비교했습니다.
      모든 주요 run은 token 32, global batch 16, generator batch 1, num inference steps 4, train/test 각 260장 기준입니다.
      이 보고서는 5 epoch 결과와 10 epoch continuation 결과를 비교하고, 각 공격별 저장된 grid 이미지를 함께 보여줍니다.
    </p>
    <div class="scorecards">
      <div><b>{html.escape(best_asr["label"])}</b><span>최고 Raw ASR: {pct(best_asr["asr"])}</span></div>
      <div><b>{html.escape(best_preserve["label"])}</b><span>최고 평균 보존: DINO {num(best_preserve["dino"])} / SSIM {num(best_preserve["ssim"])}</span></div>
      <div><b>{html.escape(best_sp["label"])}</b><span>최고 DINO-SP ASR@0.70: {pct(best_sp["dino_sp70"])}</span></div>
      <div><b>{generated}</b><span>보고서 생성일</span></div>
    </div>
  </header>
  <main>
    <section class="panel">
      <h2>실험 해석 요약</h2>
      <div class="analysis">
        <article>
          <b>ASR만 보면 margin 계열이 강함</b>
          <p>10 epoch에서 Margin+DINO와 Margin+CLIP은 각각 80% 이상 ASR에 도달했습니다. 다만 보존 지표가 크게 떨어져 공격 성공의 상당 부분은 이미지 의미 변화와 함께 발생했습니다.</p>
        </article>
        <article>
          <b>보존 loss만 쓰면 공격이 거의 안 됨</b>
          <p>CLIP only, DINO only는 DINO/SSIM이 높지만 ASR은 낮습니다. 현재 구조에서는 보존 목적만으로 classifier decision boundary를 넘기 어렵습니다.</p>
        </article>
        <article>
          <b>추가 epoch은 항상 좋은 방향이 아님</b>
          <p>Margin+CLIP은 5→10 epoch에서 ASR이 급증했지만 DINO/SSIM이 붕괴했습니다. 반대로 CR+DINO, DINO only는 보존은 유지되지만 공격은 약해졌습니다.</p>
        </article>
      </div>
    </section>
    {summary_table(rows10)}
    <div class="grid2">
      {metric_chart(rows10, "asr", "10 epoch ASR", "var(--danger)")}
      {metric_chart(rows10, "dino", "10 epoch Mean DINO Similarity", "var(--accent-2)")}
    </div>
    <div class="grid2">
      {metric_chart(rows10, "ssim", "10 epoch Mean SSIM", "var(--ok)")}
      {metric_chart(rows10, "dino_sp70", "DINO-SP ASR@0.70", "var(--warn)")}
    </div>
    {delta_chart(rows5, rows10)}
    {scatter_svg(rows10)}
    {''.join(sections)}
  </main>
</body>
</html>
"""
    REPORT_PATH.write_text(html_doc, encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    build_report()
