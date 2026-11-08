# -*- coding: utf-8 -*-
"""生成双周汇报 Word 文档。

用法（从项目根目录运行）：
    python scripts/gen_biweekly_report.py
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 字体常量
SONG = "宋体"
HEI = "黑体"


# ---------------------------------------------------------------------------
# 样式辅助
# ---------------------------------------------------------------------------
def set_run_font(run, font_name=SONG, size_pt=12, bold=False, color=None):
    """统一设置中文字体（东亚字体需走 w:eastAsia）。"""
    run.font.name = font_name
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), font_name)
    if color is not None:
        run.font.color.rgb = color


def set_paragraph_spacing(p, line_spacing=1.5, space_before=0, space_after=0,
                          alignment=None):
    pf = p.paragraph_format
    pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    pf.line_spacing = line_spacing
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    if alignment is not None:
        p.alignment = alignment


def add_heading(doc, text, level=1):
    """添加黑体标题。level=0 为居中大标题，其余左缩进。"""
    p = doc.add_paragraph()
    if level == 0:
        set_paragraph_spacing(p, 1.5, 6, 6, WD_ALIGN_PARAGRAPH.CENTER)
        run = p.add_run(text)
        set_run_font(run, HEI, 18, bold=True)
    else:
        sizes = {1: 15, 2: 14, 3: 13}
        set_paragraph_spacing(p, 1.5, 6, 3)
        run = p.add_run(text)
        set_run_font(run, HEI, sizes.get(level, 12), bold=True)
    return p


def add_body(doc, text, indent=True):
    """添加宋体小四正文段落。"""
    p = doc.add_paragraph()
    set_paragraph_spacing(p, 1.5, 0, 0, WD_ALIGN_PARAGRAPH.JUSTIFY)
    if indent:
        p.paragraph_format.first_line_indent = Pt(24)  # 首行缩进 2 字符
    run = p.add_run(text)
    set_run_font(run, SONG, 12)
    return p


def _set_cell_border(cell, top=None, bottom=None):
    """设置单元格上下边框（用于三线表）。"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.find(qn("w:tcBorders"))
    if tcBorders is None:
        from docx.oxml import OxmlElement
        tcBorders = OxmlElement("w:tcBorders")
        tcPr.append(tcBorders)

    def make_border(tag, val="single", sz="12"):
        from docx.oxml import OxmlElement
        b = OxmlElement(f"w:{tag}")
        b.set(qn("w:val"), val)
        b.set(qn("w:sz"), sz)
        b.set(qn("w:space"), "0")
        b.set(qn("w:color"), "000000")
        return b

    # 清除已有
    for side in ("top", "bottom", "left", "right"):
        existing = tcBorders.find(qn(f"w:{side}"))
        if existing is not None:
            tcBorders.remove(existing)
        # 默认无边框
        nb = OxmlElement(f"w:{side}")
        nb.set(qn("w:val"), "nil")
        tcBorders.append(nb)

    if top is not None:
        existing = tcBorders.find(qn("w:top"))
        tcBorders.remove(existing)
        tcBorders.append(make_border("top", sz=top))
    if bottom is not None:
        existing = tcBorders.find(qn("w:bottom"))
        tcBorders.remove(existing)
        tcBorders.append(make_border("bottom", sz=bottom))


def _set_cell_text(cell, text, bold=False, alignment=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    set_paragraph_spacing(p, 1.15, 0, 0, alignment)
    run = p.add_run(text)
    set_run_font(run, SONG, 10.5, bold=bold)


def add_three_line_table(doc, header, rows, caption=None):
    """生成三线表。

    header: List[str] 表头
    rows: List[List[str]] 数据行
    caption: 表题文本（置于表格上方，居中）
    """
    if caption:
        cp = doc.add_paragraph()
        set_paragraph_spacing(cp, 1.5, 4, 2, WD_ALIGN_PARAGRAPH.CENTER)
        run = cp.add_run(caption)
        set_run_font(run, HEI, 10.5, bold=True)

    n_cols = len(header)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True

    # 表头
    hdr_cells = table.rows[0].cells
    for j, h in enumerate(header):
        _set_cell_text(hdr_cells[j], h, bold=True)
        _set_cell_border(hdr_cells[j], top="18", bottom="8")  # 顶粗线、表头下细线

    # 数据行
    for i, row in enumerate(rows):
        cells = table.rows[i + 1].cells
        for j, val in enumerate(row):
            _set_cell_text(cells[j], val, bold=False)
            if i == len(rows) - 1:
                _set_cell_border(cells[j], bottom="18")  # 末行底线
            else:
                _set_cell_border(cells[j])

    # 表后留一行空行
    sp = doc.add_paragraph()
    set_paragraph_spacing(sp, 1.0, 0, 0)
    return table


def fmt(x, nd=3):
    """数值格式化为字符串，保留 nd 位有效小数。"""
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


# ---------------------------------------------------------------------------
# 文档构建
# ---------------------------------------------------------------------------
def build_document(run_dir) -> Document:
    """Build only from a completed run's verifiable facts, never invented numbers."""
    import json
    import hashlib
    run_dir = Path(run_dir)
    facts_path = run_dir / "facts.json"
    facts = json.loads(facts_path.read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["stages"]["efa"] not in ("executed", "diagnostic_completed"):
        raise ValueError("Run is incomplete; cannot generate a result report")
    for name in facts["source_files"]:
        if not (run_dir / name).is_file():
            raise ValueError(f"Missing report source: {name}")
    doc = Document()
    add_heading(doc, "双周研究进展汇报", level=0)
    add_body(doc, f"运行编号：{facts['run_id']}；事实来源SHA256：{hashlib.sha256(facts_path.read_bytes()).hexdigest()}")
    add_heading(doc, "一、本期工作概述")
    add_body(doc, "当前研究为疲劳目光行为候选双因子结构。六指标归属为4/2，分布—转移失序不代表完整方向随机性。")
    add_body(doc, f"分析决策：{facts['scientific_decision']}；证据等级：{facts['evidence_level']}；教师验收尚未完成。")
    if manifest.get("diagnostic_only"):
        add_body(doc, "本次为诊断运行，不能用作实验通过或科学验收结论。")
    add_heading(doc, "二、核心研究进展")
    data = facts["data"]
    add_three_line_table(doc, ["指标", "数值", "来源"], [
        ["被试数", str(data["n_subjects"]), "data_audit.json"],
        ["记录数", str(data["n_records"]), "data_audit.json"],
        ["生成窗口", str(data["n_generated"]), "data_audit.json"],
        ["有效窗口", str(data["n_valid"]), "data_audit.json"],
        ["无效率", f"{data['invalid_window_rate']:.2%}", "全部生成窗口作分母"],
        ["配对被试", str(facts["n_pairs"]), "validity_summary.json"],
        ["整人稳定性C5", str(facts["C5_pass"]), "cluster_summary.json"],
    ])
    add_body(doc, "各指标方向、完整载荷和区间见本run的决策报告及数值文件，不能由拟合成功推断机制成立。")
    add_heading(doc, "三、方法与复现")
    config = __import__("yaml").safe_load((run_dir / "config_used.yaml").read_text())
    add_body(doc, f"真非重叠30秒窗/10秒步长相位、迭代PAF、按人/记录等权；I-VT P{config['event_detection']['velocity_percentile']:g}，Hampel窗口{config['preprocess']['hampel_window']}采样。")
    add_body(doc, "参数、代码哈希、输入哈希和依赖版本见manifest与config_used；相位样本、bootstrap失败和配对差值均保留。")
    add_heading(doc, "四、研究文档")
    add_body(doc, "SPEC v5与Plan v5明确科学问题、统计单位、阶段门和负结果分支。")
    add_heading(doc, "五、问题与局限")
    for limitation in facts["limitations"]:
        add_body(doc, limitation)
    add_heading(doc, "六、后续计划")
    add_body(doc, "先完成方法模拟和EFA决策报告验收；CFA、预测、跨数据集和不变性按真实验收顺序执行。")
    for stage in ("cfa", "prediction", "external", "invariance"):
        add_body(doc, f"{stage}：{manifest['stages'][stage]}，无结果不能声称通过。")
    add_heading(doc, "七、需协调事项")
    add_body(doc, "原始信号效度需要EDF和相应复核证据；当前结果不能替代独立测量验证。")
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    today = date.today()
    output = args.output or PROJECT_ROOT / "docs" / "biweekly_report" / f"汇报-邓云溪-{today.month:02d}-{today.day:02d}.docx"
    if output.exists():
        parser.error("Refusing to overwrite an existing historical report")
    doc = build_document(args.run_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)
    print(f"已生成：{output}（{output.stat().st_size} bytes）")


if __name__ == "__main__":
    main()
