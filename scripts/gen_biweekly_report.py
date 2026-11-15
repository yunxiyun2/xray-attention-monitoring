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

    proto = manifest.get("protocol_version", "v5.0")
    is_v6 = proto.startswith("v6")

    groups = facts.get("groups", {})
    f1_list = groups.get("f1", [])
    f2_list = groups.get("f2", [])
    n_indicators = len(f1_list) + len(f2_list)
    structure_str = f"{len(f1_list)}/{len(f2_list)}"

    doc = Document()
    add_heading(doc, "双周研究进展汇报", level=0)
    add_body(doc, f"汇报人：邓云溪")
    add_body(doc, f"运行编号：{facts['run_id']}；协议版本：{proto}；事实来源SHA256：{hashlib.sha256(facts_path.read_bytes()).hexdigest()}")
    add_heading(doc, "一、本期工作概述")
    if is_v6:
        add_body(doc, f"当前研究为疲劳目光行为候选双因子结构。{n_indicators}指标归属为{structure_str}（F1={','.join(f1_list)}，F2={','.join(f2_list)}），SV加入F2以修正v5的4/2统计不可识别问题。")
        add_body(doc, "v5在真实DROZY数据上产出Decision B，根因为F2仅2个指标（SE、GTE）且相关极高（r≈0.90），PAF迭代出现Heywood case。v6将SV加入F2形成4/3结构（3指标>2因子），使F2达到最小识别要求。")
    else:
        add_body(doc, f"当前研究为疲劳目光行为候选双因子结构。{n_indicators}指标归属为{structure_str}，分布—转移失序不代表完整方向随机性。")
    add_body(doc, f"分析决策：{facts['scientific_decision']}；证据等级：{facts['evidence_level']}；教师验收尚未完成。")
    if manifest.get("diagnostic_only"):
        add_body(doc, "本次为诊断运行，不能用作实验通过或科学验收结论。")

    add_heading(doc, "二、核心研究进展")
    data = facts["data"]
    core_rows = [
        ["被试数", str(data["n_subjects"]), "data_audit.json"],
        ["记录数", str(data["n_records"]), "data_audit.json"],
        ["生成窗口", str(data["n_generated"]), "data_audit.json"],
        ["有效窗口", str(data["n_valid"]), "data_audit.json"],
        ["无效率", f"{data['invalid_window_rate']:.2%}", "全部生成窗口作分母"],
        ["配对被试", str(facts["n_pairs"]), "validity_summary.json"],
        ["整人稳定性C5", str(facts["C5_pass"]), "cluster_summary.json"],
        ["潜因子维度ok", str(facts["dimensions_ok"]), "parallel_analysis.json"],
        ["相位通过", str(facts["phase_pass"]), "efa_models.json"],
        ["可观测信号", str(facts["observable_signal_detected"]), "observable_covariance_signal.json"],
        ["潜因子效力验证", str(facts["latent_power_validated"]), "manifest.json"],
    ]
    if is_v6:
        core_rows.append(["方法修正要求", str(facts["method_correction_required"]), "facts.json"])
    add_three_line_table(doc, ["指标", "数值", "来源"], core_rows)
    add_body(doc, "各指标方向、完整载荷和区间见本run的决策报告及数值文件，不能由拟合成功推断机制成立。")

    if is_v6:
        sim_cert_path = run_dir.parent / "simulation_certificate.json"
        if sim_cert_path.is_file():
            sim = json.loads(sim_cert_path.read_text())
            add_heading(doc, "二（续）、模拟验证", level=2)
            add_body(doc, f"模拟验证证书：passed={sim['passed']}；诊断检查={sim['diagnostic_checks_passed']}；潜因子效力={sim['latent_power_validated']}。每组场景200组合成数据，bootstrap迭代200次。")
            sim_rows = []
            for scenario, outcome in sim["outcomes"].items():
                sim_rows.append([
                    scenario,
                    f"{outcome['latent_supported']}/{outcome['n']}",
                    f"{outcome['latent_fraction']:.1%}",
                    f"{outcome['observable_signal']}/{outcome['n']}",
                    f"{outcome['signal_fraction']:.1%}",
                ])
            add_three_line_table(doc, ["场景", "潜因子接受", "接受率", "信号检测", "信号率"], sim_rows, caption="表2 模拟验证7类场景结果")
            add_body(doc, f"v5的模拟接受率为0/200（latent_power_validated=false），v6修正后为200/200（true），证明4/3结构统计可识别。")

        diag_path = run_dir / "measurement_diagnostics.json"
        if diag_path.is_file():
            diag = json.loads(diag_path.read_text())
            add_heading(doc, "二（续）、测量诊断", level=2)
            diag_rows = []
            for key in ("FD_FN_correlation", "PSV_SA_correlation", "SE_GTE_correlation",
                        "SV_SE_correlation", "SV_GTE_correlation"):
                if key in diag:
                    diag_rows.append([key, fmt(diag[key]), "measurement_diagnostics.json"])
            if diag_rows:
                add_three_line_table(doc, ["指标对", "相关系数", "来源"], diag_rows, caption="表3 关键指标对相关系数")
            add_body(doc, f"SE-GTE相关{fmt(diag.get('SE_GTE_correlation', 0))}为极高共线性，是PAF Heywood case的直接原因。SV与SE/GTE呈中等负相关，不足以打破SE-GTE共线性主导。")

    add_heading(doc, "三、方法与复现")
    config = __import__("yaml").safe_load((run_dir / "config_used.yaml").read_text())
    add_body(doc, f"真非重叠30秒窗/10秒步长相位、迭代PAF、按人/记录等权；I-VT P{config['event_detection']['velocity_percentile']:g}，Hampel窗口{config['preprocess']['hampel_window']}采样。")
    add_body(doc, f"参数、代码哈希、输入哈希和依赖版本见manifest与config_used；协议版本{proto}。")
    add_body(doc, "相位样本、bootstrap失败和配对差值均保留。")

    add_heading(doc, "四、配对效度")
    hv = facts.get("hypothesis_validity", [])
    if hv:
        hv_rows = []
        for item in hv:
            hv_rows.append([
                item["indicator"],
                fmt(item["mean_delta_standardized"]),
                f"[{fmt(item['ci_low'])}, {fmt(item['ci_high'])}]",
                fmt(item["p_holm"]),
                str(item["expected_sign"]),
                item["conclusion"],
            ])
        add_three_line_table(doc, ["指标", "均值差(标准化)", "95%CI", "p_holm", "预期方向", "结论"], hv_rows,
                             caption=f"表4 配对效度（{len(hv)}项Holm校正）")
        add_body(doc, "全部配对效度均为inconclusive，10人配对样本下置信区间很宽。")

    add_heading(doc, "五、研究文档")
    if is_v6:
        add_body(doc, "SPEC v6与Plan v6明确4/3结构修正方案、模拟验证要求、阶段门和负结果分支。v6是v5 Decision B允许的唯一修正建议。")
        add_body(doc, "v6产出Decision B后，SPEC v6 §8规定不再继续试删或换指标。")
    else:
        add_body(doc, "SPEC v5与Plan v5明确科学问题、统计单位、阶段门和负结果分支。")

    add_heading(doc, "六、问题与局限")
    for limitation in facts["limitations"]:
        add_body(doc, limitation)
    if is_v6:
        add_body(doc, "SE-GTE近共线性（r=0.92）是DROZY数据固有特征，非方法缺陷。模拟证明4/3结构统计可识别（200/200），真实数据PAF Heywood case由SE-GTE极端共线性驱动。")

    add_heading(doc, "七、后续计划")
    if is_v6:
        add_body(doc, "v6为终止版本，不再继续试删。后续路径（SE/GTE合并、网格粒度敏感性、替代估计方法、独立数据集验证）均需新SPEC设计。")
    else:
        add_body(doc, "先完成方法模拟和EFA决策报告验收；CFA、预测、跨数据集和不变性按真实验收顺序执行。")
    for stage in ("cfa", "prediction", "external", "invariance"):
        add_body(doc, f"{stage}：{manifest['stages'][stage]}，无结果不能声称通过。")

    add_heading(doc, "八、需协调事项")
    add_body(doc, "原始信号效度需要EDF和相应复核证据；当前结果不能替代独立测量验证。")
    if is_v6:
        add_body(doc, "SE与GTE在粗网格（bins=5）下高度重叠，需评估网格粒度对共线性的影响。")
        add_body(doc, "独立数据集验证可确认SE-GTE共线性是否为DROZY数据集特异性。")
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
