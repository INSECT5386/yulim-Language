from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
import pandas as pd

# 엑셀 읽기
df = pd.read_excel("nukil_dictionary.xlsx")

# PDF 설정
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
doc = SimpleDocTemplate("nukil_dictionary.pdf", pagesize=A4, rightMargin=10, leftMargin=10, topMargin=10, bottomMargin=10)

# 스타일 생성
styles = getSampleStyleSheet()
styleN = styles["Normal"]
styleN.fontName = "HYSMyeongJo-Medium"
styleN.fontSize = 8
styleN.leading = 10

# DataFrame을 Paragraph로 변환, 내용 없으면 빼기
data = [df.columns.tolist()]
for i in range(len(df)):
    row = []
    for col in df.columns:
        cell = df.iloc[i][col]
        if pd.isna(cell) or str(cell).strip() == "":
            row.append("")  # 빈 칸으로 처리
        else:
            row.append(Paragraph(str(cell).replace("\n", "<br />"), styleN))
    data.append(row)

# 컬럼 폭 조절
col_widths = [50, 40, 80, 100, 100, 120]

# 테이블 생성
table = Table(data, colWidths=col_widths, repeatRows=1)
table.setStyle(TableStyle([
    ('FONT', (0,0), (-1,-1), 'HYSMyeongJo-Medium'),
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('GRID', (0,0), (-1,-1), 0.3, colors.black),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.lightgrey])
]))

doc.build([table])
print("✅ PDF 생성 완료! 이제 내용 없는 칸은 아예 안 나옴")

