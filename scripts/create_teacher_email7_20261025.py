"""Archive the supplied seventh teacher email with paragraph-aligned translation."""
from pathlib import Path
import hashlib
import json
from docx import Document
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
REFERENCE=Path('D:/ntu_related/dissertation/老师邮件/第六封邮件.docx')
OUT=REFERENCE.with_name('第七封邮件.docx')
QA=ROOT/'reports/teacher_email7_20261025'
PAIRS=[
('', 'Dear Zixiao,', 'Zixiao：'),
('主视觉骨干的选择',
 'The gap between frozen CLIP (0.445) and frozen DINOv2 (0.830) on the same adapter and diffusion backend is large enough to be conclusive even with one seed and reused development scenes: the visual backbone is a major bottleneck, and DINOv2 closes most of the gap to the native Diffusion Policy upper bound (0.846). You should adopt frozen DINOv2 as the main setup for all LIBERO experiments.',
 '在使用相同 Adapter 和扩散后端的情况下，冻结 CLIP（0.445）与冻结 DINOv2（0.830）之间的差距足够大，即使只有一个随机种子且重复使用了开发场景，也足以得出结论：视觉骨干网络是一个主要瓶颈，而 DINOv2 已经弥补了与原生 Diffusion Policy 性能上限（0.846）之间的大部分差距。你应当将冻结 DINOv2 作为所有 LIBERO 实验的主设置。'),
('论文呈现与五种实验条件',
 'The CLIP result is not wasted: it becomes a backbone ablation in the thesis, alongside DINOv2, which is exactly the kind of controlled comparison that strengthens the analysis. So the thesis structure should present DINOv2 as the primary backbone and report CLIP as an ablation condition on LIBERO as well, using one seed rather than three since it is a diagnostic rather than a primary result. This means you have five conditions to run on LIBERO in total: the four augmentation conditions (no augmentation, paraphrase only, counterfactual only, and full combined augmentation) with DINOv2 at three seeds each, plus CLIP with the full augmentation at one seed as the backbone ablation. Please also report the Push-T three-way comparison in the thesis as a standalone diagnostic section, since it provides direct evidence for the backbone choice and gives the reader a clear mechanistic explanation.',
 'CLIP 的结果并没有白做：它可以与 DINOv2 一起，构成论文中的视觉骨干消融实验；这种受控比较正有助于加强分析。因此，论文应将 DINOv2 作为主要视觉骨干，同时也在 LIBERO 上报告 CLIP 的消融结果。由于 CLIP 组属于诊断实验，而不是主要结果，使用一个随机种子即可，无需三个。这意味着你总共需要在 LIBERO 上运行五种条件：采用 DINOv2 的四种增强条件（无增强、仅同义改写、仅反事实增强，以及完整的组合增强），每种使用三个随机种子；另外，采用 CLIP 并使用完整增强，以一个随机种子进行视觉骨干消融。还请将 Push-T 的三方对照作为独立的诊断章节写入论文，因为它为视觉骨干的选择提供了直接证据，也能让读者清楚理解背后的机制。'),
('DINOv2 的模型规格',
 'On the DINOv2 variant, use ViT-L/14 from DINOv2 to match the patch size and embedding dimension of the CLIP backbone you have been using, which will make the comparison cleaner and minimize changes to the adapter architecture.',
 '对于 DINOv2 版本，请使用 DINOv2 的 ViT-L/14，使其图像块大小和嵌入维度与你此前使用的 CLIP 骨干网络一致。这样可以让比较更清晰，并尽量减少对 Adapter 架构的改动。'),
('近期优先级与时间要求',
 'Please proceed with the LIBERO runs using frozen DINOv2 as the primary backbone. Given where we are in the project timeline, these runs are now the critical path. Please prioritize getting all five conditions launched over the next two weeks even if the individual runs are still ongoing, and update me on their status at the next check-in.',
 '请继续推进 LIBERO 实验，以冻结 DINOv2 作为主要视觉骨干。考虑到目前的项目进度，这些实验已经成为关键路径。请优先确保在接下来的两周内将全部五种条件启动起来，即使届时个别实验仍在运行也没关系，并在下一次进度交流时向我汇报它们的状态。'),
('', 'Best regards,\nLipo', '祝好！\nLipo'),
]


def main():
    if OUT.exists():raise FileExistsError('Preserve existing email: '+str(OUT))
    QA.mkdir(parents=True,exist_ok=True)
    before=hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    doc=Document(REFERENCE)
    # Preserve the established letter styles, page geometry and page footer.
    body=doc.element.body
    for child in list(body):
        if child.tag!=qn('w:sectPr'):body.remove(child)
    doc.add_paragraph('第七封邮件','Title')
    doc.add_paragraph('Lipo 老师关于 DINOv2 主方案与 LIBERO 实验安排的回复','Subtitle')
    doc.add_paragraph('英文原文与中文译文按段落对应排列。')
    for heading,en,zh in PAIRS:
        if heading:doc.add_paragraph(heading,'Heading 1')
        doc.add_paragraph('英文原文','EnglishLabel')
        doc.add_paragraph(en)
        doc.add_paragraph('中文译文','ChineseLabel')
        translated=doc.add_paragraph(zh)
        if heading=='论文呈现与五种实验条件':translated.paragraph_format.keep_together=True
    doc.core_properties.title='第七封邮件'
    doc.core_properties.subject='Lipo老师关于DINOv2与LIBERO安排的回复及中文翻译'
    doc.save(OUT)
    saved=Document(OUT)
    texts=[p.text for p in saved.paragraphs]
    assert all(en in texts and zh in texts for _,en,zh in PAIRS)
    assert before==hashlib.sha256(REFERENCE.read_bytes()).hexdigest()
    source='\n'.join(en for _,en,_ in PAIRS)
    (QA/'english_source.txt').write_text(source+'\n',encoding='utf-8')
    report=dict(output=str(OUT),paragraph_pairs=len(PAIRS),english_preserved=True,
        english_sha256=hashlib.sha256(source.encode()).hexdigest(),
        frozen_dino_vitl14_primary=True,four_dino_conditions_three_seeds_each=True,
        clip_full_augmentation_one_seed=True,pusht_three_way_thesis_section_preserved=True,
        deadline_launch_all_conditions_within_next_two_weeks_preserved=True,
        teacher_opinions_translated_without_added_experimental_claims=True,
        previous_email_unchanged=True,output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest())
    (QA/'content_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
