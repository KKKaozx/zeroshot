from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

path = Path('D:/ntu_related/dissertation/老师邮件/第六封邮件.docx')
doc = Document(path)
assert doc.core_properties.title == '第六封邮件'
doc.styles['Subtitle'].font.italic = False
for paragraph in doc.paragraphs:
    if paragraph.text in ('视觉表示与空间定位', '近期优先事项与后续实验'):
        paragraph.paragraph_format.page_break_before = True
    if paragraph.text.startswith('子骁：'):
        paragraph.text = paragraph.text.replace('子骁：', 'Zixiao：', 1)
for element in (doc.styles.element, doc.element):
    for border in list(element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
doc.save(path)
print('Refined title border and aligned section pagination')
