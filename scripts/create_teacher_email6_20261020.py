"""Preserve teacher's supplied English and provide paragraph-aligned Chinese."""
from pathlib import Path
import json

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path('D:/ntu_related/dissertation/老师邮件/第六封邮件.docx')
QA = Path('D:/ntu_related/dissertation/Zero_shot/reports/teacher_email6_20261020')

PAIRS = [
    ('', 'Dear Zixiao,\nThank you for the detailed update.',
     'Zixiao：\n感谢你提供详细的进展汇报。'),
    ('离线指标与闭环操作',
     'The offline metrics you report, position error dropping from 7.78 to 4.77 cm and rotation error from 12.74 to 12.04 degrees on Bridge, are meaningful improvements but the absolute values are still too large for reliable manipulation. A position error of nearly 5 cm is on the order of the size of most tabletop objects, so the closed-loop failure is expected given these numbers. The key question your Push-T comparison will answer is whether the gap between your adapter frontend and a directly trained diffusion policy is large enough to be the primary explanation, or whether the remaining error is intrinsic to the difficulty of the task distribution and data scale.',
     '你汇报的离线指标表明，在 Bridge 上，位置误差从 7.78 cm 降至 4.77 cm，旋转误差从 12.74° 降至 12.04°。这些改善是有意义的，但误差的绝对值仍然过大，尚不足以支持可靠的操作。接近 5 cm 的位置误差与大多数桌面物体的尺寸处于同一量级，因此，从这些数值来看，闭环执行失败是可以预期的。你的 Push-T 对照将回答一个关键问题：你的 Adapter 前端与直接训练的扩散策略之间的差距，是否大到足以成为主要解释；还是说，剩余误差主要源于任务分布本身的难度和数据规模。'),
    ('视觉表示与空间定位',
     'Regarding likely causes, the most probable explanation at this stage is that CLIP ViT-L/14 features, while semantically rich, are not spatially precise enough for low-level manipulation without additional spatial grounding. CLIP was trained on image-text pairs for recognition, not for pixel-level spatial reasoning, and its visual tokens do not carry reliable spatial coordinates. When the cross-attention adapter queries these tokens for action generation, it receives semantic category information but not precise object location. This is a known limitation in the literature and is one reason why methods like R3M and MVP, which are pre-trained specifically on manipulation video, tend to outperform CLIP for low-level control. Testing whether replacing the CLIP visual encoder with a manipulation-specific visual backbone such as R3M (Nair et al., NeurIPS 2022) or with a spatial-aware backbone such as DINOv2 (Oquab et al., ICLR 2024) improves your offline metrics would be a meaningful contribution in itself and would directly address one of the thesis research questions about what makes the adapter work.',
     '关于可能的原因，现阶段最有可能的解释是：CLIP ViT-L/14 特征虽然包含丰富的语义信息，但如果缺少额外的空间定位机制，其空间精度不足以支持底层操作控制。CLIP 使用图像与文本配对进行训练，目标是识别，而不是像素级的空间推理，其视觉 token 并不携带可靠的空间坐标。当交叉注意力 Adapter 为生成动作而查询这些 token 时，得到的是语义类别信息，而不是精确的物体位置。这是文献中已知的局限，也是 R3M 和 MVP 这类专门在操作视频上预训练的方法，在底层控制中往往优于 CLIP 的原因之一。可以测试：将 CLIP 视觉编码器替换为面向操作任务的视觉骨干网络，例如 R3M（Nair 等，NeurIPS 2022），或具有空间感知能力的骨干网络，例如 DINOv2（Oquab 等，ICLR 2024），是否能够改善离线指标。这项测试本身就可以构成有意义的研究贡献，并且直接回应论文中的一个研究问题：究竟哪些因素使 Adapter 有效。'),
    ('条件注入架构',
     'The second likely cause is the conditioning architecture. A single fused context vector from the cross-attention output may be an information bottleneck for the diffusion decoder. The original Diffusion Policy conditions the U-Net at every resolution level through film-style conditioning, whereas passing only a single context vector to the U-Net input may not give the denoiser enough guidance at each denoising step. I recommend checking whether conditioning the U-Net at multiple levels rather than only at the input improves performance once your Push-T diagnostic is done.',
     '第二个可能的原因是条件注入架构。交叉注意力输出的单个融合上下文向量，可能成为扩散解码器的信息瓶颈。原始 Diffusion Policy 通过 FiLM 风格的条件调制，在 U-Net 的每个分辨率层级注入条件；相比之下，如果只把一个上下文向量传入 U-Net 的输入端，可能无法在每个去噪步骤中给予去噪网络足够的引导。我建议，在完成 Push-T 诊断之后，检查在 U-Net 多个层级注入条件，而不是仅在输入端注入，是否能够改善性能。'),
    ('近期优先事项与后续实验',
     'For your immediate priorities, complete the Push-T adapter comparison first, as you planned, and report the overlap reward for your adapter versus the native Diffusion Policy. This result belongs in your thesis as a controlled ablation isolating the frontend contribution. Once that result is in hand, if your adapter underperforms significantly, the next experiment should be swapping the CLIP visual backbone for DINOv2 or R3M while keeping everything else identical, to test whether the visual representation is the bottleneck. If your adapter performs comparably to the native policy, the bottleneck is more likely in the data scale or task difficulty, and the priority should shift to running the full training pipeline on the larger Open X-Embodiment subsets. In parallel, please begin running your four training conditions on the LIBERO benchmark, since those results are central to the thesis and need sufficient training time. Do not delay the LIBERO experiments further, as the thesis must include complete benchmark results with three random seeds per condition as I described earlier.',
     '关于当前的优先事项，请先按计划完成 Push-T 的 Adapter 对照，并汇报你的 Adapter 与原生 Diffusion Policy 各自的重叠奖励。这项结果应当写入论文，作为一项用于分离前端贡献的受控消融实验。拿到结果后，如果你的 Adapter 表现明显较差，下一项实验应当是在其他设置完全相同的情况下，把 CLIP 视觉骨干网络替换为 DINOv2 或 R3M，以检验视觉表示是否是瓶颈。如果你的 Adapter 与原生策略表现相当，那么瓶颈更可能在于数据规模或任务难度，此时应将重点转向在规模更大的 Open X-Embodiment 子集上运行完整训练流程。同时，请开始在 LIBERO 基准上运行你的四种训练条件，因为这些结果是论文的核心内容，而且需要足够的训练时间。不要再推迟 LIBERO 实验；正如我此前所说，论文必须包含完整的基准测试结果，每种训练条件都要使用三个随机种子。'),
    ('', 'Please update me with the Push-T comparison result as soon as it is available.\nBest regards,\nLipo',
     'Push-T 对照结果出来后，请尽快向我汇报。\n祝好！\nLipo'),
]


def font(style, name, east, size, bold=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), east)


def main():
    if OUT.exists():
        raise FileExistsError('Preserve existing email: ' + str(OUT))
    QA.mkdir(parents=True, exist_ok=True)
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin = section.bottom_margin = Inches(.75)
    section.left_margin = section.right_margin = Inches(.85)
    section.header_distance = section.footer_distance = Inches(.32)
    font(doc.styles['Normal'], 'Calibri', '宋体', 11)
    normal = doc.styles['Normal'].paragraph_format
    normal.line_spacing = 1.14
    normal.space_after = Pt(7)
    normal.widow_control = True
    font(doc.styles['Title'], 'Calibri', '微软雅黑', 19)
    doc.styles['Title'].paragraph_format.space_after = Pt(7)
    font(doc.styles['Subtitle'], 'Calibri', '宋体', 11)
    doc.styles['Subtitle'].font.italic = False
    font(doc.styles['Heading 1'], 'Calibri', '微软雅黑', 12, True)
    doc.styles['Heading 1'].paragraph_format.space_before = Pt(12)
    doc.styles['Heading 1'].paragraph_format.space_after = Pt(6)
    for name in ('EnglishLabel', 'ChineseLabel'):
        style = doc.styles.add_style(name, 1)
        font(style, 'Calibri', '宋体', 10, True)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(4)
        style.paragraph_format.space_after = Pt(3)
    doc.add_paragraph('第六封邮件', 'Title')
    doc.add_paragraph('Lipo 老师关于泛化问题与后续实验的回复', 'Subtitle')
    doc.add_paragraph('英文原文与中文译文按段落对应排列。')
    for heading, en, zh in PAIRS:
        if heading:
            p = doc.add_paragraph(heading, 'Heading 1')
            if heading in ('视觉表示与空间定位', '近期优先事项与后续实验'):
                p.paragraph_format.page_break_before = True
        doc.add_paragraph('英文原文', 'EnglishLabel')
        doc.add_paragraph(en)
        doc.add_paragraph('中文译文', 'ChineseLabel')
        doc.add_paragraph(zh)
    footer = section.footer.paragraphs[0]
    footer.alignment = 2
    run = footer.add_run('第 ')
    run.font.size = Pt(9)
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    footer._p.append(field)
    footer.add_run(' 页').font.size = Pt(9)
    doc.core_properties.title = '第六封邮件'
    doc.core_properties.subject = 'Lipo老师回复的英文原文与中文翻译'
    doc.core_properties.author = 'Zixiao'
    for element in (doc.styles.element, doc.element):
        for border in list(element.iter(qn('w:pBdr'))):
            border.getparent().remove(border)
    doc.save(OUT)
    saved = Document(OUT)
    texts = [p.text for p in saved.paragraphs]
    for _, en, zh in PAIRS:
        assert en in texts and zh in texts
    (QA / 'content_check.json').write_text(json.dumps({
        'output': str(OUT), 'paragraph_pairs': len(PAIRS),
        'english_and_translation_complete': True,
        'four_training_conditions_and_three_seeds_preserved': True,
        'teacher_claims_translated_without_added_analysis': True,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(OUT)


if __name__ == '__main__':
    main()
