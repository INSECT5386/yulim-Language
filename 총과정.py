import requests
import json
import pandas as pd
!pip install reportlab
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
import pandas as pd

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

# ⬇️ 데이터와 토크나이저 다운로드
download_file('https://huggingface.co/datasets/Yuchan5386/YuLimo/resolve/main/yuchan_dict.json?download=true', 'yuchan_dict.json')

CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def get_chosung(word):
    result = ""
    for char in word:
        if '가' <= char <= '힣':
            code = ord(char) - 0xAC00
            cho = code // 588
            result += CHOSUNG[cho]
        else:
            result += char
    return result

# JSON 불러오기
with open("yuchan_dict.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 뜻 있는 항목 / 없는 항목 나누기
with_meaning = {k:v for k,v in data.items() if "뜻" in v}
without_meaning = {k:v for k,v in data.items() if "뜻" not in v}
sorted_with_meaning = dict(sorted(with_meaning.items(), key=lambda item: get_chosung(item[1]["뜻"])))
sorted_data = {**sorted_with_meaning, **without_meaning}

with open("yuchan_dict_re.json", "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)

# 1. JSON 파일 읽기
with open("yuchan_dict_re.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. 단어별로 표 형태로 변환 (변형/형태 제외)
rows = []
for word, info in data.items():
    품사 = info.get("품사", "")
    뜻 = info.get("뜻", "")
    설명 = info.get("설명", "")
    예문 = "\n".join(info.get("예문", [])) if "예문" in info else ""
    
    # 변형/형태는 완전히 무시하고 포함하지 않음
    rows.append([word, 품사, 뜻, 설명, 예문])

# 3. 데이터프레임 생성
df = pd.DataFrame(rows, columns=["단어", "품사", "뜻", "설명", "예문"])

# 4. 엑셀로 저장
df.to_excel("yuchan_dict.xlsx", index=False)


df = pd.read_excel("yuchan_dict.xlsx")

# PDF 설정
pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
doc = SimpleDocTemplate("yulimo.pdf", pagesize=A4, rightMargin=10, leftMargin=10, topMargin=10, bottomMargin=10)

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

# Install the Hugging Face CLI
!pip install -U "huggingface_hub[cli]"

# Login with your Hugging Face credentials
!hf auth login

# Push your dataset files
!hf upload Yuchan5386/YuLimo . --repo-type=dataset
