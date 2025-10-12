import json

# JSON 파일 불러오기
with open("nukil_dict_re.json", "r", encoding="utf-8") as f:
    nukil_dict = json.load(f)

def search_nukil(word):
    word = word.strip().lower()
    entry = nukil_dict.get(word)
    if entry:
        print(f"🔤 단어: {word}")
        for key, value in entry.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for subkey, subvalue in value.items():
                    print(f"  - {subkey}: {subvalue}")
            elif isinstance(value, list):
                print(f"{key}:")
                for item in value:
                    print(f"  - {item}")
            else:
                print(f"{key}: {value}")
    else:
        print("❌ 해당 단어는 사전에 없습니다.")

if __name__ == "__main__":
    while True:
        w = input("\n검색할 누킬어 단어 입력 (q: 종료): ").strip()
        if w.lower() == "q":
            print("👋 프로그램 종료!")
            break
        search_nukil(w)
