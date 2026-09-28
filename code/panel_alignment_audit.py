from __future__ import annotations

import json
from pathlib import Path


def require_matplotlib_panel_alignment(
    fig,
    *,
    json_out,
    overlay_svg,
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
):
    """Audit rendered Matplotlib axes geometry without external helper code."""
    fig.canvas.draw()
    width_pt, height_pt = (float(v) * 72.0 for v in fig.get_size_inches())
    axes = [ax for ax in fig.axes if ax.get_visible()]
    panels = []
    errors = []
    for index, ax in enumerate(axes, start=1):
        box = ax.get_position()
        left, bottom = box.x0 * width_pt, box.y0 * height_pt
        right, top = box.x1 * width_pt, box.y1 * height_pt
        if right <= left or top <= bottom:
            errors.append(f"panel {index} has nonpositive dimensions")
        panels.append({
            "id": f"panel-{index}",
            "bbox_pt": [left, bottom, right, top],
        })

    rows = []
    tolerance = float(tolerance_pt)
    for panel in sorted(panels, key=lambda item: -((item["bbox_pt"][1] + item["bbox_pt"][3]) / 2)):
        _, bottom, _, top = panel["bbox_pt"]
        center = (bottom + top) / 2
        for row in rows:
            row_center = sum((p["bbox_pt"][1] + p["bbox_pt"][3]) / 2 for p in row) / len(row)
            if abs(center - row_center) <= tolerance:
                row.append(panel)
                break
        else:
            rows.append([panel])

    for row_index, row in enumerate(rows, start=1):
        if len(row) < 2:
            continue
        tops = [p["bbox_pt"][3] for p in row]
        bottoms = [p["bbox_pt"][1] for p in row]
        if max(tops) - min(tops) > tolerance or max(bottoms) - min(bottoms) > tolerance:
            errors.append(f"row {row_index} panel plot areas are not vertically aligned")

    overlaps = []
    for i, first in enumerate(panels):
        a1, b1, a2, b2 = first["bbox_pt"]
        for second in panels[i + 1:]:
            c1, d1, c2, d2 = second["bbox_pt"]
            overlap_x = min(a2, c2) - max(a1, c1)
            overlap_y = min(b2, d2) - max(b1, d1)
            if overlap_x > float(gutter_tolerance_pt) and overlap_y > float(gutter_tolerance_pt):
                overlaps.append([first["id"], second["id"]])
    if overlaps:
        errors.append(f"overlapping plot areas: {overlaps}")

    report = {
        "schema_version": 1,
        "figure_size_pt": [width_pt, height_pt],
        "tolerance_pt": float(tolerance_pt),
        "gutter_tolerance_pt": float(gutter_tolerance_pt),
        "panels": panels,
        "horizontal_rows": [[p["id"] for p in row] for row in rows],
        "overlapping_pairs": overlaps,
        "passed": not errors,
        "errors": errors,
    }
    json_path = Path(json_out)
    svg_path = Path(overlay_svg)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rects = []
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for index, panel in enumerate(panels):
        left, bottom, right, top = panel["bbox_pt"]
        y = height_pt - top
        rects.append(
            f'<rect x="{left:.3f}" y="{<local path omitted>" width="{right-left:.3f}" '
            f'height="{top-bottom:.3f}" fill="none" stroke="{colors[index % len(colors)]}" '
            f'stroke-width="0.8"/><text x="{left+2:.3f}" y="{y+9:.3f}" '
            f'font-size="7" fill="{colors[index % len(colors)]}">{panel["id"]}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_pt:.3f}pt" '
        f'height="{height_pt:.3f}pt" viewBox="0 0 {width_pt:.3f} {height_pt:.3f}">'
        f'<rect width="100%" height="100%" fill="white"/>{"".join(rects)}</svg>'
    )
    svg_path.write_text(svg, encoding="utf-8")
    if strict and errors:
        raise RuntimeError("; ".join(errors))
    return report
