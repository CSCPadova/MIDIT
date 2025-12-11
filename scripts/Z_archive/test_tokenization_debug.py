from transformers import T5Tokenizer

# Usa lo stesso tokenizer del tuo progetto
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

samples = [
    "140 BPM",
    "at 140 BPM",
    "tempo 140",
    "tempo: 140",
    "one hundred forty BPM",
    "140"
]

for s in samples:
    enc = tokenizer(s, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print("PROMPT:", repr(s))
    print("  token ids:", ids)
    print("  tokens   :", tokens)
    print("-"*60)
