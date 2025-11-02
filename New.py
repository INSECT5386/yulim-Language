import requests
import json
import pandas as pd
!pip install reportlab # 주피터 환경이 아닐 경우 이 줄은 주석 처리하거나 제거해야 합니다.
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.enums import TA_CENTER

# 초성 리스트 정의 (정렬 로직에는 직접 사용되지 않음)
CHOSUNG = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']

def download_file(url, save_path):
    """지정된 URL에서 파일을 다운로드합니다."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ 파일 저장됨: {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"❌ 파일 다운로드 실패: {e}")
        # 다운로드 실패 시 빈 JSON 데이터를 생성하여 스크립트 실행이 멈추지 않도록 합니다.
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({}, f)
        print("⚠️ 빈 데이터를 사용하여 계속 진행합니다.")


def get_chosung(word):
    """한글 단어의 초성만을 추출하여 반환합니다."""
    result = ""
    for char in word:
        if '가' <= char <= '힣':
            code = ord(char) - 0xAC00
            cho = code // 588
            result += CHOSUNG[cho]
        else:
            result += char
    return result

# ⬇️ 데이터 다운로드
DOWNLOADED_JSON_FILE = 'yuchan_dict_original.json'
SORTED_JSON_FILE = 'yuchan_dict_re.json'
EXCEL_FILE = 'yuchan_dict.xlsx'
PDF_FILE = 'yulimo.pdf'
DOWNLOAD_URL = 'https://huggingface.co/datasets/Yuchan5386/YuLimo/resolve/main/yuchan_dict_re.json?download=true'

# 데이터와 토크나이저 다운로드
download_file(DOWNLOAD_URL, DOWNLOADED_JSON_FILE)

# 1. JSON 불러오기
try:
    with open(DOWNLOADED_JSON_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError:
    print(f"❌ {DOWNLOADED_JSON_FILE} 파일을 로드할 수 없습니다. 빈 데이터를 사용합니다.")
    data = {}


# --- 정렬 로직 시작 (1순위: 품사, 2순위: 뜻의 ㄱㄴㄷ 순) ---

# 딕셔너리 항목들을 리스트로 변환
items = list(data.items())

def sorting_key(item):
    """
    정렬 기준: (1순위: 품사 문자열, 2순위: '뜻' 문자열)
    1순위로 품사를 묶고, 2순위로 '뜻'을 기준으로 ㄱㄴㄷ 정렬을 수행합니다.
    """
    word = item[0]
    info = item[1]
    
    # 1순위: 품사 문자열. 품사가 없으면 'ZZZZ'를 넣어 가장 뒤로 정렬되게 함.
    pos = info.get("품사", "ZZZZ")
    
    # 2순위: '뜻' 문자열. 뜻이 없으면 'ZZZZZ'를 넣어 해당 그룹 내에서 가장 뒤로 정렬되게 함.
    meaning = info.get("뜻", "ZZZZZ")
    
    # 정렬 키 반환: (품사, 뜻)
    return (pos, meaning)

# 항목 정렬 실행
items.sort(key=sorting_key)

# 정렬된 리스트를 다시 딕셔너리로 변환
sorted_data = dict(items)

# --- 정렬 로직 끝 ---

# 2. 정렬된 데이터를 새로운 JSON 파일로 저장
with open(SORTED_JSON_FILE, "w", encoding="utf-8") as f:
    json.dump(sorted_data, f, ensure_ascii=False, indent=2)
print(f"✅ 정렬된 데이터 저장됨: {SORTED_JSON_FILE}")

# 3. 단어별로 표 형태로 변환 (변형/형태 제외)
rows = []
for word, info in sorted_data.items():
    # 모든 정보가 항상 존재하는 것이 아니므로 .get() 사용
    품사 = info.get("품사", "")
    뜻 = info.get("뜻", "")
    설명 = info.get("설명", "")
    # 예문은 리스트이므로 줄바꿈 처리하여 문자열로 변환
    예문 = "\n".join(info.get("예문", [])) if "예문" in info else ""
    
    rows.append([word, 품사, 뜻, 설명, 예문])

# 4. 데이터프레임 생성
df = pd.DataFrame(rows, columns=["단어", "품사", "뜻", "설명", "예문"])

# 5. 엑셀로 저장
df.to_excel(EXCEL_FILE, index=False)
print(f"✅ 엑셀 파일 저장됨: {EXCEL_FILE}")


# --- PDF 생성 파트 ---

# ReportLab을 위한 폰트 설정
try:
    pdfmetrics.registerFont(UnicodeCIDFont("HYSMyeongJo-Medium"))
    KOREAN_FONT = "HYSMyeongJo-Medium"
except:
    print("⚠️ HYSMyeongJo-Medium 폰트 등록에 실패했습니다. 기본 폰트(Helvetica)를 사용하며 한글이 깨질 수 있습니다.")
    KOREAN_FONT = "Helvetica"


# PDF 문서 설정
doc = SimpleDocTemplate(PDF_FILE,
                        pagesize=A4,
                        rightMargin=10,
                        leftMargin=10,
                        topMargin=20, # 상단 여백 증가
                        bottomMargin=10)

# 스타일 생성 및 수정
styles = getSampleStyleSheet()

# 1. 본문 내용 스타일 (표 내부)
styleN = styles["Normal"]
styleN.fontName = KOREAN_FONT
styleN.fontSize = 8
styleN.leading = 10

# 2. 제목 스타일
styleTitle = styles["Title"]
styleTitle.fontName = KOREAN_FONT
styleTitle.fontSize = 18
styleTitle.alignment = TA_CENTER

# 3. 설명 스타일 (하단용)
styleExplanation = styles["Normal"]
styleExplanation.fontName = KOREAN_FONT
styleExplanation.fontSize = 9
styleExplanation.alignment = TA_CENTER
styleExplanation.leading = 10 # 줄 간격을 좁혀 한 페이지에 더 잘 맞도록 조정

# --- 제목 및 설명 텍스트 준비 ---
total_words = len(sorted_data)
title_text = "YuLimo (유리모) 단어장"

# ⭐⭐ 압축된 설명 텍스트 적용 (페이지 넘김 방지) ⭐⭐
explanation_text = (
    f"--- 조합식 ---<br/>"
    f"&lt;시제&gt; fo'(미래), pa'(과거) | -nus(진행), -pus(완료)<br/>"
    f"&lt;파생&gt; 동사+-utu(피동), -io(명사), -a(형용사), -ia(부사)<br/>"
    f"&lt;기타&gt; 동사+-wen/hen/cen(의도/의무/가능) | 명사/형용사+-vus(~해지다)<br/>"
    f"<br/>"
    f"--- 데이터 요약 ---<br/>"
    f"총 {total_words}개 단어. [품사] → [뜻의 가나다순] 정렬."
)
# ⭐⭐⭐⭐

# Flowables 리스트 (문서에 들어갈 요소들)
flowables = []

# 1. 맨 위에 제목 Paragraph 추가
flowables.append(Paragraph(title_text, styleTitle))
flowables.append(Spacer(1, 10)) # 제목과 테이블 사이 간격 10pt

# DataFrame을 Table 데이터로 변환 (기존 로직)
table_data = [df.columns.tolist()] # 헤더 행
for i in range(len(df)):
    row = []
    for col in df.columns:
        cell_content = str(df.iloc[i][col])
        # NaN 또는 공백은 빈 문자열로 처리
        if pd.isna(cell_content) or cell_content.strip() == "" or cell_content == "nan":
            row.append("")  
        else:
            # ReportLab의 Paragraph를 사용하여 셀 내용 처리 (줄바꿈 허용)
            row.append(Paragraph(cell_content.replace("\n", "<br />"), styleN))
    table_data.append(row)

# 컬럼 폭 조절 (A4 용지 너비 약 595pt. 여백 20pt 제외하고 575pt 활용)
col_widths = [50, 40, 100, 150, 235] # 단어, 품사, 뜻, 설명, 예문 (총 575pt)

# 테이블 생성
table = Table(table_data, colWidths=col_widths, repeatRows=1)
table.setStyle(TableStyle([
    # 전체 폰트 설정
    ('FONT', (0,0), (-1,-1), KOREAN_FONT),
    # 정렬 및 그리드
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('GRID', (0,0), (-1,-1), 0.3, colors.black),
    # 헤더 행 스타일
    ('FONT', (0,0), (-1,0), KOREAN_FONT, 9),
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EFEFEF')),
    # 내용 행 배경색 교차 적용
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F5F5F5')])
]))

# 2. 테이블 객체를 flowables 리스트에 추가
flowables.append(table)

# 3. 맨 아래에 설명 Paragraph 추가
flowables.append(Spacer(1, 15)) # 테이블과 설명글 사이 간격 15pt
flowables.append(Paragraph(explanation_text, styleExplanation))

# PDF 파일 생성 (Flowables 리스트 사용)
try:
    doc.build(flowables)
    print(f"✅ PDF 파일 저장됨: {PDF_FILE}")
except Exception as e:
    print(f"❌ PDF 생성 중 오류 발생: {e}")
    print("ReportLab 설정 또는 한글 폰트 문제일 수 있습니다.")


# --- Hugging Face 업로드 파트 (업로드 명령어는 실제 환경에서 실행해야 함) ---
print("\n--- Hugging Face 업로드 안내 ---")
print("# Hugging Face CLI 설치 및 로그인 명령어는 현재 환경에서 실행되지 않습니다.")
print("# 파일을 업로드하려면 해당 명령어를 직접 실행해야 합니다.")
!pip install -U "huggingface_hub[cli]"
!hf auth login
!hf upload Yuchan5386/YuLimo . --repo-type=dataset
print("---------------------------------")
