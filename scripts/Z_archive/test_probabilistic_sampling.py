from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# =====================================
# 1️⃣ Setup modello e tokenizer
# =====================================
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
model.eval()

# =====================================
# 2️⃣ Prompt e varianti di test
# =====================================
prompts = [
    "tempo",
    "tempo is",
    "the song has tempo",
    "song tempo",
]

tempi = ["60", "90", "120", "140", "160"]

# =====================================
# 3️⃣ Analisi probabilistica
# =====================================
def token_probabilities(prompt, candidates):
    inputs = tokenizer(prompt, return_tensors="pt")
    decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids
    
    with torch.no_grad():
        outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[0, -1, :]  # solo l’ultimo token predetto
        probs = torch.softmax(logits, dim=-1)

    print(f"\n🔹 PROMPT: '{prompt}'")
    for c in candidates:
        token_id = tokenizer.encode(" " + c, add_special_tokens=False)[0]
        p = probs[token_id].item()
        print(f"  {c:>4} BPM → Prob: {p:.6f}")

# =====================================
# 4️⃣ Esecuzione test
# =====================================
for p in prompts:
    token_probabilities(p, tempi)

