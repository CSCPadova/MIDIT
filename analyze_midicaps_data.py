import os
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

# ===========================
# 🎵 ANALISI ESPLORATIVA MIDICAPS
# ===========================

def analyze_midicaps(data_dir="./data", split="train", max_samples=1000, save_dir="./results/eda_midicaps"):
    """Analizza statisticamente il dataset MidiCaps."""

    print(f"🔍 Caricamento dataset MidiCaps ({split}, max {max_samples} esempi)...")
    dataset = load_dataset("amaai-lab/MidiCaps", split=f"{split}[:{max_samples}]", cache_dir=data_dir)

    print(f"✓ Caricati {len(dataset)} esempi")

    df = pd.DataFrame(dataset)

    # Creazione cartella risultati
    os.makedirs(save_dir, exist_ok=True)

    # ===========================
    # 🔢 Statistiche generali
    # ===========================
    print("\n📊 Statistiche generali:")
    print(df.describe(include="all"))

    # Salva overview CSV
    df.describe(include="all").to_csv(os.path.join(save_dir, "summary.csv"))

    # ===========================
    # 🎚️ Tempo (BPM)
    # ===========================
    if "tempo" in df.columns:
        plt.figure()
        df["tempo"].dropna().astype(float).plot.hist(bins=30)
        plt.title("Distribuzione Tempo (BPM)")
        plt.xlabel("BPM")
        plt.ylabel("Frequenza")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "tempo_distribution.png"))
        plt.close()

    # ===========================
    # ⏱️ Durata
    # ===========================
    if "duration" in df.columns:
        plt.figure()
        df["duration"].dropna().astype(float).plot.hist(bins=30, color='orange')
        plt.title("Distribuzione Durata (secondi)")
        plt.xlabel("Durata (s)")
        plt.ylabel("Frequenza")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "duration_distribution.png"))
        plt.close()

    # ===========================
    # 🎹 Generi musicali
    # ===========================
    if "genre" in df.columns:
        all_genres = [g for sublist in df["genre"].dropna() for g in (sublist if isinstance(sublist, list) else [sublist])]
        genre_series = pd.Series(all_genres)
        top_genres = genre_series.value_counts().head(20)

        plt.figure(figsize=(8,5))
        top_genres.plot(kind='bar')
        plt.title("Top 20 Generi Musicali")
        plt.xlabel("Genere")
        plt.ylabel("Frequenza")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "genres_top20.png"))
        plt.close()

    # ===========================
    # 🎺 Strumenti
    # ===========================
    if "instrument_summary" in df.columns:
        all_instruments = [i for sublist in df["instrument_summary"].dropna() for i in (sublist if isinstance(sublist, list) else [sublist])]
        instr_series = pd.Series(all_instruments)
        top_instr = instr_series.value_counts().head(20)

        plt.figure(figsize=(8,5))
        top_instr.plot(kind='bar', color='green')
        plt.title("Top 20 Strumenti")
        plt.xlabel("Strumento")
        plt.ylabel("Frequenza")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "instruments_top20.png"))
        plt.close()

    # ===========================
    # 😌 Mood
    # ===========================
    if "mood" in df.columns:
        all_moods = [m for sublist in df["mood"].dropna() for m in (sublist if isinstance(sublist, list) else [sublist])]
        mood_series = pd.Series(all_moods)
        top_moods = mood_series.value_counts().head(20)

        plt.figure(figsize=(8,5))
        top_moods.plot(kind='bar', color='purple')
        plt.title("Top 20 Mood")
        plt.xlabel("Mood")
        plt.ylabel("Frequenza")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "moods_top20.png"))
        plt.close()

    # ===========================
    # 🧠 Caption (testo)
    # ===========================
    if "caption" in df.columns:
        df["caption_length"] = df["caption"].dropna().apply(lambda x: len(str(x).split()))
        plt.figure()
        df["caption_length"].plot.hist(bins=30, color='gray')
        plt.title("Distribuzione Lunghezza Caption")
        plt.xlabel("Numero parole")
        plt.ylabel("Frequenza")
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "caption_length.png"))
        plt.close()

    print(f"\n📁 Analisi completata. Grafici salvati in: {save_dir}")

if __name__ == "__main__":
    analyze_midicaps()
