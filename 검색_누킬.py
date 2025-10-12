import json

# JSON 파일 불러오기
with open("nukil_dict_re.json", "r", encoding="utf-8") as f:
    nukil_dict = json.load(f)

def search_korean(korean_word):
    korean_word = korean_word.strip()
    found = False

    for nukil_word, entry in nukil_dict.items():
        for key, value in entry.items():
            # value가 문자열일 경우
            if isinstance(value, str) and korean_word in value:
                print(f"🔤 누킬어: {nukil_word}")
                print(f"  {key}: {value}")
                found = True

            # value가 리스트일 경우
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and korean_word in item:
                        print(f"🔤 누킬어: {nukil_word}")
                        print(f"  {key}: {item}")
                        found = True

            # value가 dict일 경우
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, str) and korean_word in subvalue:
                        print(f"🔤 누킬어: {nukil_word}")
                        print(f"  {key} - {subkey}: {subvalue}")
                        found = True

    if not found:
        print("❌ 해당 한국어 뜻에 해당하는 누킬어 단어를 찾을 수 없습니다.")

if __name__ == "__main__":
    while True:
        w = input("\n검색할 한국어 뜻 입력 (q: 종료): ").strip()
        if w.lower() == "q":
            print("👋 프로그램 종료!")
            break
        search_korean(w)
