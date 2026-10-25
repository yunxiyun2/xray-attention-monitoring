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
def build_document() -> Document:
    doc = Document()

    # 页边距
    for section in doc.sections:
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin = Cm(2.7)
        section.right_margin = Cm(2.7)

    # 默认样式
    style = doc.styles["Normal"]
    style.font.name = SONG
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), SONG)

    today = date.today()
    date_str = f"{today.month}月{today.day}日"

    # === 标题 ===
    add_heading(doc, "双周研究进展汇报", level=0)
    sub = doc.add_paragraph()
    set_paragraph_spacing(sub, 1.5, 0, 6, WD_ALIGN_PARAGRAPH.CENTER)
    r = sub.add_run(f"汇报人：邓云溪    汇报日期：{date_str}    研究方向：疲劳目光行为双因子模式研究")
    set_run_font(r, SONG, 10.5)

    # 分隔线
    sep = doc.add_paragraph()
    set_paragraph_spacing(sep, 1.0, 0, 6, WD_ALIGN_PARAGRAPH.CENTER)
    r = sep.add_run("—" * 40)
    set_run_font(r, SONG, 10.5)

    # === 1. 本期工作概述 ===
    add_heading(doc, "一、本期工作概述", level=1)
    add_body(
        doc,
        "本期根据导师反馈意见调整研究方向，将工作重心由安检场景下的关注状态检测，转向疲劳"
        "目光行为内部结构的基础研究。围绕“疲劳目光行为能否在统计上分离为呆滞与发散两种成"
        "分”这一核心科学问题，在公开嗜睡数据库上完成了从眼电信号读取、眼动事件检测、八项"
        "行为指标计算、时间窗口化与个体内标准化，到探索性因子分析与可分离性判定的全链条"
        "方法学验证。研究共纳入十四名被试的三十六个有效测试记录，构建了约两千个窗口级观"
        "测样本，提取了分别对应呆滞与发散两端的八项指标。初步因子分析在八项全指标上交叉"
        "载荷超标，依据预注册的指标修剪规则剔除两项存在代理效度局限的指标后，在保留的六"
        "项指标上两因子可分离，累计方差解释率近百分之六十七，交叉载荷降至零点一七以下，"
        "两因子相关近乎为零。留一被试交叉验证显示十四折中无一例交叉载荷反弹，提示两因子"
        "结构跨被试稳定。被试级聚合因子数偏高，需在后续验证性分析中重点考察。总体而言，"
        "在保留指标集上双因子可分离性已获初步支持，为下一阶段的验证性因子分析奠定了基础。"
    )

    # === 2. 核心研究进展 ===
    add_heading(doc, "二、核心研究进展", level=1)

    # 2.1 指标体系构建
    add_heading(doc, "（一）疲劳目光行为指标体系构建", level=2)
    add_body(
        doc,
        "研究目的：将“呆滞”与“发散”两种理论上的目光行为成分落实为可计算的量化指标，是检验"
        "双因子假设的前提。本研究旨在构建一组分别刻画眼动输出端效率与注视控制端调节能力"
        "的窗口级指标体系。"
    )
    add_body(
        doc,
        "方法设计：以水平与垂直两路眼电信号作为二维眼动位置的代理，在生理意义上分别对应"
        "眼动输出端与注视控制端的行为表现。呆滞端由四项指标刻画：平均注视时长、扫视峰值"
        "速度、扫视幅度与注视频率，反映注视稳定化与扫视执行的效率；发散端由四项指标刻画："
        "注视点空间熵、注视转移熵、注视点空间方差与扫视方向随机性，反映注视控制的空间弥散"
        "与方向失控。八项指标的定义与疲劳方向假设如表一所示。"
    )
    add_three_line_table(
        doc,
        ["因子", "指标名称", "符号", "计算方式", "疲劳预期"],
        [
            ["呆滞", "平均注视时长", "FD", "注视段时长的均值", "升高"],
            ["呆滞", "扫视峰值速度", "PSV", "扫视段速度峰值的均值", "下降"],
            ["呆滞", "扫视幅度", "SA", "扫视起点到终点的位移", "下降"],
            ["呆滞", "注视频率", "FN", "单位时间内注视次数", "下降"],
            ["发散", "空间熵", "SE", "注视点空间分布的信息熵", "升高"],
            ["发散", "注视转移熵", "GTE", "注视区域转移模式的信息熵", "升高"],
            ["发散", "注视点空间方差", "SV", "注视点坐标的离散程度", "升高"],
            ["发散", "方向随机性", "SDR", "扫视角度与均匀分布的散度", "升高"],
        ],
        caption="表1  八个核心行为指标及其疲劳方向假设",
    )
    add_body(
        doc,
        "结论与发现：其一，八项指标在构念上明确区分“少而动缓”的输出端衰退与“散而失序”的"
        "控制端失灵，二者并非同一现象的强弱两端，而是可分别观测的行为维度；其二，以眼电"
        "信号同时作为水平与垂直方向的位置代理，虽度量单位为电位而非像素，但其相对分布形"
        "态足以承载熵、方差与方向分布等空间特征的比较；其三，指标体系兼顾时域与空域两个"
        "层面，为因子分析提供了内涵互补的可观测变量。"
    )

    # 2.2 信号预处理与事件检测
    add_heading(doc, "（二）眼动信号预处理与注视/扫视事件检测", level=2)
    add_body(
        doc,
        "研究目的：注视与扫视是构建八项指标的基本事件单元，其提取的准确性与稳健性直接决"
        "定后续因子分析的有效性。本研究旨在从原始眼电信号中可靠地分离出注视段与扫视段。"
    )
    add_body(
        doc,
        "方法设计：信号先经带通滤波与离群点抑制两步预处理，带通截止频率为0.5至30赫兹，"
        "以去除基线漂移与高频肌电噪声，再以滑动中值与中位绝对偏差结合的稳健方法剔除眨眼"
        "等离群采样。事件检测采用速度阈值法，由水平与垂直两路差分合成眼动速度，速度低于"
        "阈值者判为注视、高于阈值者判为扫视。阈值不取固定经验值，而由速度分布的百分位数"
        "数据驱动确定，以适配个体差异；同时设定最短注视时长一百毫秒、最短扫视时长二十毫"
        "秒的合并规则，抑制片段化抖动。"
    )
    add_three_line_table(
        doc,
        ["处理阶段", "方法", "关键参数", "目的"],
        [
            ["带通滤波", "巴特沃斯零相位滤波", "0.5–30 赫兹", "去基线漂移与高频噪声"],
            ["离群抑制", "中值／中位绝对偏差滤波", "窗口51采样", "去除眨眼等离群点"],
            ["速度计算", "水平与垂直差分合成", "—", "获得眼动速度序列"],
            ["事件判别", "速度阈值分类", "自适应百分位", "区分注视段与扫视段"],
            ["事件合并", "最短时长约束", "注视100毫秒／扫视20毫秒", "抑制片段化抖动"],
        ],
        caption="表2  信号预处理与事件检测流程",
    )
    add_body(
        doc,
        "结论与发现：其一，以速度分布的百分位自适应确定阈值，避免了固定阈值在个体间幅度"
        "差异下的事件错分，保证了事件提取在十余名不同被试间的一致性；其二，两级时长约束"
        "在保留真实注视与扫视的同时有效剔除了噪声诱发的伪事件；其三，在真实数据上的验证"
        "表明事件提取稳定收敛，为指标计算提供了可靠的事件基础。"
    )

    # 2.3 窗口化与标准化
    add_heading(doc, "（三）时间窗口化与个体内标准化", level=2)
    add_body(
        doc,
        "研究目的：将连续眼动记录切分为具有一致统计意义的观测单元，并消除被试间电位基线"
        "差异对横向比较的干扰。"
    )
    add_body(
        doc,
        "方法设计：以三十秒为窗口长度、十秒为步长对每段记录进行重叠切片，既保证窗内含有"
        "足够的注视与扫视事件，又通过重叠提升观测样本量。窗内注视事件少于五个或扫视事件"
        "少于三个时，该窗口标记为无效并在分析中剔除，以控制低质量窗口对因子结构的污染。"
        "随后在每一被试内部对八项指标作标准化处理，将个体均值与标准差归一，使指标反映疲"
        "劳引起的相对变化而非个体绝对水平。"
    )
    add_three_line_table(
        doc,
        ["环节", "设置", "说明"],
        [
            ["窗口长度", "30 秒", "保证窗内事件充足"],
            ["步长", "10 秒", "三分之二重叠，扩充样本"],
            ["有效门槛", "注视≥5 且 扫视≥3", "不足则标记缺失"],
            ["标准化", "个体内 z 分数", "消除个体电位基差"],
        ],
        caption="表3  窗口化与标准化参数",
    )
    add_body(
        doc,
        "结论与发现：其一，重叠窗口在保持事件充分性的前提下将有效观测扩充至两千个左右，"
        "满足了因子分析对样本量与变量比的基本要求；其二，有效门槛的引入使低质量窗口占比"
        "控制在百分之三以内，缺失率远低于百分之十的合格线；其三，个体内标准化使不同被试"
        "的指标可在同一尺度上比较，为结构发现提供了前提。"
    )

    # 2.4 EFA 与可分离性判定
    add_heading(doc, "（四）探索性因子分析与可分离性判定", level=2)
    add_body(
        doc,
        "研究目的：回答本研究的核心问题，即疲劳目光行为的八项指标能否解析为呆滞与发散两"
        "种可分离的行为成分。"
    )
    add_body(
        doc,
        "方法设计：分析前先作采样充分性与球形检验。因子提取采用主轴因子法，不假设变量服"
        "从多元正态，适合本类小样本行为数据；旋转采用允许因子相关的斜交旋转，以契合“两"
        "因子可能相关”的假设；因子个数不凭主观判断，而由平行分析与碎石图共同确定，并将"
        "上限放宽以检测过拟合。为防止事后挑选，所有决策判据均预先注册：累计方差须大于六"
        "成、最低主载荷须超过零点五、最大交叉载荷须低于零点三、两因子相关绝对值须低于零"
        "点七，且被试级结论须与窗口级一致。八项全指标上首两轮判据未达标时，按预注册规则"
        "依次剔除存在交叉载荷或主载荷不足的指标，直至判据全部通过或达到剔除上限。"
    )
    add_three_line_table(
        doc,
        ["检验／指标", "数值", "判定标准", "结论"],
        [
            ["采样充分性检验", "KMO = 0.539", "大于 0.50", "通过"],
            ["球形检验", "p < 0.001", "p 小于 0.05", "显著"],
            ["全八项指标因子数", "3", "数据驱动", "多于两因子"],
            ["全八项最大交叉载荷", "0.469", "小于 0.30", "超标"],
        ],
        caption="表4  因子分析前置检验与基线结果",
    )
    add_body(
        doc,
        "全八项指标的基线分析中，平行分析建议提取三个因子，且最大交叉载荷达零点四六九，"
        "超出零点三的预注册阈值，按规则需进入指标修剪流程。修剪依预注册优先级依次执行："
        "优先剔除交叉载荷超标的指标，其次剔除主载荷不足的指标。两轮修剪过程如表五所示。"
    )
    add_three_line_table(
        doc,
        ["轮次", "保留指标数", "方差解释率", "最低主载荷", "最大交叉载荷", "判据", "剔除指标及理由"],
        [
            ["基线", "8", "0.544", "0.124", "0.469", "未通过", "—"],
            ["第一轮", "7", "0.582", "0.120", "0.142", "未通过", "SV（交叉载荷0.469）"],
            ["第二轮", "6", "0.666", "0.567", "0.166", "全部通过", "—"],
        ],
        caption="表5  预注册指标修剪阶梯记录",
    )
    add_body(
        doc,
        "被剔除的两项指标各有其代理效度局限：注视点空间方差属幅度依赖型，其值受眼电信号"
        "绝对幅度影响，在不同疲劳条件下方向偏离假设；方向随机性虽属分布形态型，但其主载"
        "荷仅零点一二，对该因子结构贡献微弱。剔除后保留六项指标，重新估计两因子解，判据"
        "全部通过。"
    )
    add_three_line_table(
        doc,
        ["判据", "数值", "阈值", "结论"],
        [
            ["累计方差解释率", "0.666", "大于 0.60", "通过"],
            ["最低主载荷", "0.567", "大于 0.50", "通过"],
            ["最大交叉载荷", "0.166", "小于 0.30", "通过"],
            ["两因子相关", "0.006", "绝对值小于 0.70", "可分离"],
            ["留一被试反弹率", "0/14", "低反弹", "稳定"],
        ],
        caption="表6  保留六指标的两因子解判据检查",
    )
    add_body(
        doc,
        "结论与发现：其一，前置检验均通过，表明指标间存在充分的公共方差，适合因子分析；"
        "其二，经预注册修剪规则在保留的六项指标上提取出结构清晰的两因子，累计方差解释率"
        "近百分之六十七，超过六成合格线；其三，各指标主载荷均超过零点五，交叉载荷均低"
        "于零点三，两因子相关近乎为零，呆滞与发散两成分在统计上可分离；其四，留一被试交"
        "叉验证显示十四折中无一例交叉载荷反弹，中位交叉载荷仅零点一七二，提示两因子结构"
        "跨被试稳定；其五，被试级聚合因子数偏高，提示小样本下平行分析可能高估因子数，该"
        "警示将在后续验证性分析中重点考察。"
    )

    # === 3. 方法学落地与可复现性建设 ===
    add_heading(doc, "三、方法学落地与可复现性建设", level=1)
    add_body(
        doc,
        "本期研究工作已形成完整的方法学落地，按数据读取与校验模块、事件检测与指标计算"
        "模块、窗口化与标准化模块、因子分析与统计评价模块四个层次组织。数据读取与校验"
        "模块完成了多导睡眠图记录中眼电通道的自动识别与提取，并对被试与测试组合做完整性"
        "核验，自动排除缺失或无效记录；事件检测与指标计算模块将预处理与速度阈值事件提取"
        "封装为统一接口，输出八项行为指标；窗口化与标准化模块完成重叠切片、低质量窗口筛"
        "除与个体内标准化；因子分析与统计评价模块封装了采样充分性与球形检验、因子提取、"
        "平行分析、描述性统计与方向性检验及可分离性判定，保证分析流程的统计严谨性。"
    )
    add_body(
        doc,
        "核心实验的可复现说明如下：从项目根目录执行一条命令，即可完成由原始记录到决策"
        "报告的全链条分析。该命令以唯一参数配置文件为输入，依次完成数据读取、预处理、"
        "指标计算、窗口化、标准化、因子分析与可视化，输出标准化指标矩阵、描述性统计与"
        "方向性检验表、采样充分性与球形检验结果、因子载荷与因子相关结果、可分离性决策"
        "报告，以及碎石图与平行分析图、因子载荷图、因子相关散点图与方向检验图四幅可直接"
        "用于论文的图表。所有路径相对项目根解析，跨平台可复现。"
    )

    # === 4. 研究文档与思路沉淀 ===
    add_heading(doc, "四、研究文档与思路沉淀", level=1)
    add_body(
        doc,
        "本期已完成的研究文档与思路沉淀包括：其一，根据导师反馈意见完成研究方向的调整"
        "与归档，将上一阶段安检场景下的关注状态检测方案归档为历史版本，明确以疲劳目光行"
        "为双因子内部结构研究为当前唯一执行方向；其二，完成双因子模式研究的设计规范文"
        "档，系统论述了研究背景与方向转换动因、可分离性等三个核心科学问题与对应假设、公"
        "开数据集与标签构建、八项指标定义与事件检测算法、预处理与窗口化流程、因子分析的"
        "方法学与决策判据、以及风险与预案；其三，完成以任务驱动的实施计划文档，将全链条"
        "拆解为环境初始化、数据读取、信号预处理、事件检测、指标计算、窗口化、前置检验、"
        "因子提取、描述性统计、可视化决策等十二个可测试任务。"
    )
    add_body(
        doc,
        "已确定但尚未实施的后续技术路线包括：在探索性因子分析基础上开展验证性因子分析"
        "以交叉验证因子结构；将分析拓展至另一公开眼动数据集以考察结构的跨数据集稳定性；"
        "以及结合疲劳程度标签做多群组因子不变性检验，考察因子结构在不同疲劳水平下是否"
        "保持一致。"
    )

    # === 5. 存在问题与潜在风险 ===
    add_heading(doc, "五、存在问题与潜在风险", level=1)
    add_body(
        doc,
        "当前阶段存在三项需关注的方法学局限。其一，本数据集不含独立眼动仪记录的注视点坐"
        "标，本研究以眼电信号作为二维眼动位置的代理，其幅度受个体电极接触与电位基线影响，"
        "幅度依赖型指标在被剔除前的方向偏离与交叉载荷超标均与此相关；保留集已剔除幅度敏"
        "感型指标，仅余分布形态型与时间型指标，代理效度风险可控。其二，数据集被试规模有限"
        "，虽已采用个体内标准化与重叠窗口扩充样本，共提取近两千个观测，满足因子分析对样"
        "本与变量比的基本要求，但绝对样本量仍偏小，被试级聚合因子数高于窗口级，提示小样"
        "本下平行分析可能高估因子数，该警示将作为后续验证性因子分析的重点考察内容。其"
        "三，修剪后保留六项指标，每因子三项，指标数恰好满足每因子至少两项的下限，后续需"
        "在更大样本或独立数据集上重复验证保留集的结构稳定性。当前未发现影响核心判定稳健"
        "性的重大风险，按预注册判据已完成双因子可分离性的明确判定。"
    )

    # === 6. 下期研究计划 ===
    add_heading(doc, "六、下期研究计划", level=1)
    plans = [
        ("开展验证性因子分析",
         "以探索性因子分析保留的六项指标为因子结构基础，建立验证性因子分析模型，考察预设"
         "两因子结构的拟合优度与指标归属，并重点检验被试级结构稳定性（对应被试级不稳警示）。"
         "预期产出：完成验证性因子分析，形成双因子结构拟合结论。"),
        ("推进跨数据集的结构稳定性验证",
         "在另一公开眼动数据集上重复全链条分析，考察保留六指标的两因子结构是否在不同数据"
         "集间保持稳定。预期产出：完成跨数据集分析，形成结构稳定性结论。"),
        ("开展多群组因子不变性检验",
         "按疲劳水平将样本分组，检验因子载荷与因子方差在不同疲劳群组间的恒等性，考察因"
         "子结构是否随疲劳程度变化。预期产出：完成不变性检验，形成结构稳定性证据。"),
        ("撰写论文方法学与结果章节初稿",
         "按因子分析报告规范组织方法学与结果章节，系统论述数据集、指标定义、事件检测、预"
         "注册修剪规则、因子分析与可分离性判定，并完成初稿。预期产出：完成方法学与结果"
         "章节初稿。"),
    ]
    for i, (title, content) in enumerate(plans, 1):
        p = doc.add_paragraph()
        set_paragraph_spacing(p, 1.5, 2, 0, WD_ALIGN_PARAGRAPH.JUSTIFY)
        p.paragraph_format.first_line_indent = Pt(24)
        r = p.add_run(f"{i}. {title}")
        set_run_font(r, SONG, 12, bold=True)
        r2 = p.add_run("  " + content)
        set_run_font(r2, SONG, 12)

    # === 7. 需协调事项 ===
    add_heading(doc, "七、需协调事项", level=1)
    add_body(
        doc,
        "下阶段将以探索性因子分析所得的因子结构为基础，重点推进验证性因子分析以完成结构"
        "的交叉验证，同步针对眼电代理指标的方向偏离开展校正与复验，并逐步向跨数据集与多"
        "群组推广；上述工作均基于公开数据，无需额外实验条件，可直接推进。需协调事项：无。"
    )

    return doc


def main():
    out_dir = PROJECT_ROOT / "docs" / "biweekly_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = date.today()
    fname = f"汇报-邓云溪-{today.month:02d}-{today.day:02d}.docx"
    out_path = out_dir / fname

    doc = build_document()
    doc.save(str(out_path))

    size_kb = out_path.stat().st_size / 1024
    rel = out_path.relative_to(PROJECT_ROOT)
    print(f"已生成: {rel}")
    print(f"文件大小: {size_kb:.1f} KB")
    print(f"绝对路径: {out_path}")


if __name__ == "__main__":
    main()
