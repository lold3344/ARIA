import json, random

# Convert DataBase.txt (groups of 7 messages separated by newlines) into
# a JSONL dialog dataset with roles: Пользователь / Ассистент.
# Each line is one dialog turn pair used for supervised fine-tuning.

in_path = r"C:\Users\lold\Documents\GitHub\ARIA\data base\DataBase.txt"
out_path = r"C:\Users\lold\Documents\GitHub\ARIA\data base\DataBase_roles.jsonl"

def is_clean_line(line):
    s = line.strip()
    if not s:
        return False
    if s.startswith("http") or s.startswith("www") or ".ru" in s or ".com" in s:
        return False
    if len(s) < 5 or len(s) > 300:
        return False
    return True

records = []
with open(in_path, "r", encoding="utf-8") as f:
    for block in f:
        block = block.strip()
        if not block:
            continue
        messages = [m.strip() for m in block.split(".") if is_clean_line(m)]
        if len(messages) < 2:
            continue
        # Build sliding role pairs: even indices = user, odd = assistant.
        for i in range(0, len(messages) - 1, 2):
            user_msg = messages[i]
            assistant_msg = messages[i + 1]
            # Skip if either contains vulgarity roots (cheap pre-filter)
            low = (user_msg + " " + assistant_msg).lower()
            bad = ["хуй", "хуе", "хуя", "пизд", "пидор", "пидар", "еба", "ебу", "ебе", "еби", "ебл", "ебн", "ебя", "бля", "сук", "сос", "соса", "соси", "сосн", "сосу", "сосет", "сосешь"]
            if any(b in low for b in bad):
                continue
            text = f"Пользователь: {user_msg}\nАссистент: {assistant_msg}"
            records.append({"text": text})

random.shuffle(records)

with open(out_path, "w", encoding="utf-8") as out:
    for rec in records:
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")

print(f"Done: {len(records)} dialog records written to {out_path}")
