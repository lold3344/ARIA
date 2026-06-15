import json, os, re, random, glob

BAD_ROOTS = [
    "хуй", "хуе", "хуя", "хуи", "хул",
    "пизд", "пидор", "пидар", "пидр",
    "еба", "ебу", "ебе", "ебо", "еби", "ебл", "ебн", "ебя",
    "бля", "сук", "жоп", "муд", "говн", "дерьм", "уеб", "выеб", "заеб",
    "долбоеб", "пиздабол", "охуе", "ахуе", "наху",
    "сос", "соса", "соси", "посос", "сосн", "сосу", "сосет", "сосешь",
    "бамп", "перека", "перекач", "дабл", "дубль", "трипл",
]

FORUM_SLANG = ["бамп", "перека", "перекач", "дабл", "дубль", "трипл", "ап", "апн", "апни", "up"]

def has_bad(text):
    t = text.lower()
    for r in BAD_ROOTS:
        if r in t:
            return True
    return False

def has_slang(text):
    t = text.lower()
    for r in FORUM_SLANG:
        if r in t:
            return True
    return False

def is_clean(text):
    if len(text) < 30 or len(text) > 500:
        return False
    if "http" in text or "www." in text or ".ru" in text or ".com" in text:
        return False
    if text.count(">") > 3 or text.count("<") > 3:
        return False
    if has_bad(text):
        return False
    if has_slang(text):
        return False
    # require mostly cyrillic
    letters = sum(1 for c in text if c.isalpha())
    cyr = sum(1 for c in text if 'а' <= c.lower() <= 'я' or c.lower() == 'ё')
    if letters > 0 and cyr / letters < 0.80:
        return False
    return True

files = glob.glob(r"C:\Users\lold\Desktop\LLM\18.6GB\*\*.jsonl")
random.shuffle(files)

out_path = r"C:\Users\lold\Documents\GitHub\ARIA\data base\DataBase_dialog.txt"
out_size = 3 * 1024 * 1024 * 1024  # 3 GB
written = 0
messages = []

with open(out_path, "w", encoding="utf-8") as out:
    for path in files:
        if written >= out_size:
            break
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                    if not isinstance(text, str):
                        continue
                    text = text.replace("\n", " ").strip()
                    if is_clean(text):
                        messages.append(text)
                        if len(messages) >= 7:
                            block = ". ".join(messages) + ".\n"
                            out.write(block)
                            written += len(block.encode("utf-8"))
                            messages = []
                            if written >= out_size:
                                break
                except Exception:
                    continue
        print(f"written {written / 1024 / 1024:.0f} MB from {path}")

if messages:
    block = ". ".join(messages) + ".\n"
    out.write(block)

print(f"Done: {written / 1024 / 1024 / 1024:.2f} GB written to {out_path}")
