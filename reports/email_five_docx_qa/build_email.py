from pathlib import Path
from docx import Document
from docx.shared import Mm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path(r'D:\ntu_related\dissertation\老师邮件\第五封邮件.docx')
EN = [
    'Dear Zixiao,',
    'Thank you for the update. The symptom you describe, training loss decreasing while evaluation error exceeds a no-motion baseline, is a classic sign of overfitting combined with a distribution mismatch between training and evaluation, and it is good that you caught it early through the small-sample fitting test.',
    'Before comparing with the published Diffusion Policy code line by line, I suggest you first check a few specific things that are the most common culprits in this kind of setup. The action normalization is the single most likely cause: if the training actions are normalized to a certain range but the evaluation predictions are not denormalized back before computing the error, or if the normalization statistics were computed on the training split and the evaluation split has a different range, the predictions will look systematically wrong even if the model has learned something reasonable. Please verify that you apply exactly the same normalization and denormalization at training time and evaluation time, using statistics computed on the training data only. The second thing to check is the noise schedule at inference time: during diffusion denoising, if the number of denoising steps or the beta schedule used at inference differs from what was used during training, the output distribution will be wrong. Make sure your inference loop uses the same schedule and the same number of steps as the training objective assumes. The third thing to check is whether your evaluation is conditioning on the correct context: at evaluation time, the adapter must receive the same format of visual observation and language instruction as it saw during training, including the same image normalization applied to the CLIP backbone input. A mismatch in image preprocessing between training and evaluation will cause the visual embeddings to be out of distribution even for seen tasks.',
    'Your plan to resolve these issues before moving to broader experiments is the right one. Please send me a short update once you have identified the root cause, even if it is before the two-week mark, so we can discuss the fix together if needed.',
    'Best regards,\nLipo',
]
ZH = [
    '亲爱的 Zixiao：',
    '感谢你的进度更新。你所描述的现象，即训练损失不断下降，而评估误差却高于不移动基线，是过拟合与训练和评估之间的分布不匹配共同出现时的典型表现。你通过小样本拟合测试及早发现了这一问题，这一点很好。',
    '在逐行对照已发表的 Diffusion Policy 代码之前，我建议你先检查几个具体方面，因为它们是这类设置中最常见的问题来源。动作归一化是最有可能的原因：如果训练动作被归一化到某个范围，但评估时没有先将预测结果反归一化，再计算误差；或者归一化统计量是根据训练集计算的，而评估集的动作范围不同，那么即使模型已经学到了一些合理的信息，预测结果也会呈现出系统性的偏差。请确认训练和评估时采用完全一致的归一化与反归一化方法，并且仅使用训练数据计算统计量。第二个需要检查的方面是推理时的噪声调度：在扩散去噪过程中，如果推理使用的去噪步数或 beta 调度与训练时使用的不同，输出分布就会出现偏差。请确保推理循环使用的调度和步数与训练目标所假定的一致。第三个需要检查的方面是评估是否使用了正确的条件上下文：评估时，适配器必须接收与训练时相同格式的视觉观测和语言指令，包括对 CLIP 主干输入采用相同的图像归一化方法。训练和评估之间的图像预处理不一致，会导致视觉嵌入偏离训练分布，即使评估的是已经见过的任务也会如此。',
    '你计划先解决这些问题，再开展更大范围的实验，这个方向是正确的。一旦确定了根本原因，请给我发送一份简短的进度更新，即使还没有到两周的汇报时间；这样，如有需要，我们可以一起讨论修复方案。',
    '祝好，\nLipo',
]

doc = Document()
section = doc.sections[0]
section.page_width = Mm(210); section.page_height = Mm(297)
section.top_margin = section.bottom_margin = Mm(20)
section.left_margin = section.right_margin = Mm(22)
for name, size in [('Normal',11),('Title',18),('Heading 1',13)]:
    style = doc.styles[name]
    style.font.name = 'Calibri'; style.font.size = Pt(size)
    style.font.color.rgb = RGBColor(0,0,0)
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn('w:eastAsia'),'宋体')
    for border in list(style.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
normal = doc.styles['Normal'].paragraph_format
normal.space_after = Pt(9); normal.line_spacing = 1.12
normal.widow_control = True
doc.styles['Title'].paragraph_format.space_after = Pt(5)
doc.styles['Heading 1'].paragraph_format.space_before = Pt(8)
doc.styles['Heading 1'].paragraph_format.space_after = Pt(10)
doc.add_paragraph('第五封邮件',style='Title')
doc.add_paragraph('Lipo 关于训练泛化问题的回信')
doc.add_paragraph('英文原文',style='Heading 1')
for text in EN:
    doc.add_paragraph(text)
doc.add_page_break()
doc.add_paragraph('中文翻译',style='Heading 1')
for text in ZH:
    p = doc.add_paragraph(text)
    p.paragraph_format.line_spacing = 1.35
    for run in p.runs:
        run.font.size = Pt(11.5)
doc.core_properties.title = '第五封邮件'
doc.core_properties.subject = 'Lipo 关于训练泛化问题的回信 英文原文及中文翻译'
doc.core_properties.author = ''
assert not OUT.exists(), 'Do not overwrite an existing archived email'
doc.save(OUT)
check = Document(OUT)
texts = [p.text for p in check.paragraphs]
assert all(p in texts for p in EN+ZH)
print(OUT)
print('English and Chinese paragraphs verified; bytes:',OUT.stat().st_size)
