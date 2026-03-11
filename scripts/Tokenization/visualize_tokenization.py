"""
Visualizza la tokenizzazione di un file preprocessato (.pt)
Permette di verificare che la tokenizzazione sia coerente col modello.

Uso:
    python scripts/Tokenization/visualize_tokenization.py
    python scripts/Tokenization/visualize_tokenization.py --file data/preprocessed_phase1/lakh_0001_xxx.pt
    python scripts/Tokenization/visualize_tokenization.py --bars 3
"""

import torch
import argparse
import random
from pathlib import Path
from collections import Counter
from miditok import REMI, TokenizerConfig

# ============================================================================
# Colori ANSI per il terminale
# ============================================================================
COLORS = {
    'pitch':    '\033[94m',   # blu
    'velocity': '\033[92m',   # verde
    'duration': '\033[93m',   # giallo
    'program':  '\033[95m',   # magenta
    'position': '\033[96m',   # ciano
    'bar':      '\033[91m',   # rosso
    'tempo':    '\033[33m',   # arancione
    'timesig':  '\033[35m',   # viola
    'rest':     '\033[36m',   # turchese
    'special':  '\033[90m',   # grigio
    'mask':     '\033[97m\033[41m',  # bianco su sfondo rosso
    'reset':    '\033[0m',
}

TYPE_LABELS = {
    'pitch':    'PTCH',
    'velocity': 'VEL ',
    'duration': 'DUR ',
    'program':  'PROG',
    'position': 'POS ',
    'bar':      'BAR ',
    'tempo':    'TMP ',
    'timesig':  'TSIG',
    'rest':     'REST',
    'special':  'SPEC',
}


def build_tokenizer():
    """Inizializza il tokenizer con la stessa config del training."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_controls=False,
    )
    return REMI(tokenizer_config=config)


def classify_token(token_name):
    """Classifica un token nel suo tipo."""
    if token_name.startswith(('Pitch_', 'PitchDrum_')):
        return 'pitch'
    elif token_name.startswith('Velocity_'):
        return 'velocity'
    elif token_name.startswith('Duration_'):
        return 'duration'
    elif token_name.startswith('Program_'):
        return 'program'
    elif token_name.startswith('Position_'):
        return 'position'
    elif token_name.startswith('Bar_'):
        return 'bar'
    elif token_name.startswith('Tempo_'):
        return 'tempo'
    elif token_name.startswith('TimeSig_'):
        return 'timesig'
    elif token_name.startswith('Rest_'):
        return 'rest'
    else:
        return 'special'


def visualize_sequence(tokens, id_to_token, bar_positions, max_bars=3, use_color=True):
    """Stampa la sequenza token bar per bar."""

    def colorize(text, token_type):
        if not use_color:
            return text
        color = COLORS.get(token_type, '')
        return f"{color}{text}{COLORS['reset']}"

    print("\n" + "=" * 70)
    print("📜 SEQUENZA TOKEN (bar per bar)")
    print("=" * 70)

    # Determina le barre da mostrare
    num_bars = len(bar_positions) - 1
    bars_to_show = min(max_bars, num_bars)
    print(f"   Mostrando {bars_to_show} di {num_bars} bar totali\n")

    for bar_idx in range(bars_to_show):
        start = bar_positions[bar_idx]
        end = bar_positions[bar_idx + 1]
        bar_tokens = tokens[start:end]

        print(f"  ┌─ BAR {bar_idx + 1} (posizioni {start}–{end-1}, {end-start} token)")

        line = "  │  "
        for tok_id in bar_tokens.tolist():
            tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
            tok_type = classify_token(tok_name)
            label = TYPE_LABELS.get(tok_type, '????')
            colored = colorize(f"[{label}:{tok_name}]", tok_type)
            line += colored + " "

            # Vai a capo ogni 3 note (12 token) per leggibilità
            if line.count('[') % 6 == 0 and line.count('[') > 0:
                print(line)
                line = "  │  "

        if line.strip() != "│":
            print(line)

        print("  └" + "─" * 60)
        print()


def show_type_distribution(tokens, id_to_token):
    """Mostra la distribuzione dei tipi di token."""
    print("=" * 70)
    print("📊 DISTRIBUZIONE TIPI DI TOKEN")
    print("=" * 70)

    type_counts = Counter()
    for tok_id in tokens.tolist():
        tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
        tok_type = classify_token(tok_name)
        type_counts[tok_type] += 1

    total = len(tokens)
    bar_width = 30

    for tok_type in ['bar', 'position', 'program', 'pitch', 'velocity', 'duration',
                     'tempo', 'timesig', 'rest', 'special']:
        count = type_counts.get(tok_type, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = '█' * int(pct / 100 * bar_width)
        label = TYPE_LABELS.get(tok_type, '????')
        print(f"  {label}  {bar:<{bar_width}}  {count:>5} token  ({pct:5.1f}%)")

    print(f"\n  {'TOTALE':30}  {total:>5} token")


def show_masking_example(tokens, id_to_token, mask_ratio=0.15, mask_id=3):
    """Mostra come appare la sequenza dopo il masking (primi 60 token)."""
    print("\n" + "=" * 70)
    print(f"🎭 ESEMPIO MASKING ({mask_ratio*100:.0f}% token sostituiti con [MASK])")
    print("=" * 70)

    preview = tokens[:60]
    seq_len = len(preview)
    num_mask = max(1, int(seq_len * mask_ratio))
    mask_positions = set(random.sample(range(seq_len), num_mask))

    print(f"\n  Token originali  (primi {seq_len}):")
    orig_line = "  "
    for i, tok_id in enumerate(preview.tolist()):
        tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
        tok_type = classify_token(tok_name)
        orig_line += f"{TYPE_LABELS.get(tok_type,'????')} "
    print(orig_line)

    print(f"\n  Dopo masking ({num_mask} posizioni mascherate):")
    mask_line = "  "
    for i, tok_id in enumerate(preview.tolist()):
        if i in mask_positions:
            mask_line += f"\033[97m\033[41m[MSK]\033[0m "
        else:
            tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
            tok_type = classify_token(tok_name)
            mask_line += f"{TYPE_LABELS.get(tok_type,'????')} "
    print(mask_line)

    print(f"\n  Posizioni mascherate: {sorted(mask_positions)}")


def verify_consistency(tokens, id_to_token, tokenizer):
    """Verifica la coerenza con le aspettative del modello."""
    print("\n" + "=" * 70)
    print("✅ VERIFICA COERENZA CON IL MODELLO")
    print("=" * 70)

    vocab_size = len(tokenizer.vocab)
    max_id = tokens.max().item()
    min_id = tokens.min().item()
    unknown_ids = [i for i in tokens.tolist() if i not in id_to_token]

    print(f"\n  Vocab size tokenizer:    {vocab_size}")
    print(f"  Token ID range nei dati: [{min_id}, {max_id}]")
    print(f"  Token IDs sconosciuti:   {len(unknown_ids)}")

    if max_id >= vocab_size:
        print(f"\n  ❌ PROBLEMA: token ID {max_id} >= vocab_size {vocab_size}")
    else:
        print(f"\n  ✅ Tutti i token ID sono nel range [0, {vocab_size-1}]")

    if unknown_ids:
        print(f"  ❌ {len(unknown_ids)} token non decodificabili")
    else:
        print(f"  ✅ Tutti i token sono decodificabili")

    # Verifica token speciali
    special_ids = {
        'PAD': tokenizer.vocab.get('PAD_None', None),
        'BOS': tokenizer.vocab.get('BOS_None', None),
        'EOS': tokenizer.vocab.get('EOS_None', None),
        'MASK': tokenizer.vocab.get('MASK_None', None),
    }
    print(f"\n  Token speciali:")
    for name, tok_id in special_ids.items():
        present = tok_id in tokens.tolist() if tok_id is not None else False
        status = "presente" if present else "assente "
        print(f"    {name:4}: ID={tok_id}  ({status})")


def main():
    parser = argparse.ArgumentParser(description="Visualizza tokenizzazione MIDI")
    parser.add_argument('--file', type=str, default=None,
                        help="Percorso file .pt (default: random da data/preprocessed_phase1/)")
    parser.add_argument('--bars', type=int, default=3,
                        help="Numero di bar da visualizzare (default: 3)")
    parser.add_argument('--no-color', action='store_true',
                        help="Disabilita colori ANSI")
    args = parser.parse_args()

    # ── Trova il file ────────────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
    else:
        data_dir = Path("data/preprocessed_phase1")
        all_files = list(data_dir.glob("*.pt"))
        if not all_files:
            print(f"❌ Nessun file .pt trovato in {data_dir}")
            return
        file_path = random.choice(all_files)

    print(f"\n{'='*70}")
    print(f"🎵 VISUALIZZAZIONE TOKENIZZAZIONE MIDI")
    print(f"{'='*70}")
    print(f"\n  File: {file_path.name}")

    # ── Carica dati ─────────────────────────────────────────────────────────
    data = torch.load(file_path, weights_only=False)
    tokens = data['tokens']
    bar_positions = data['bar_positions']
    metadata = data.get('metadata', {})

    print(f"  Token totali:    {len(tokens)}")
    print(f"  Bar totali:      {len(bar_positions) - 1}")
    if metadata:
        print(f"  Sorgente:        {metadata.get('source_file', 'N/A')}")
        print(f"  Tempo iniziale:  {metadata.get('initial_tempo', 'N/A')}")
        print(f"  TimeSig:         {metadata.get('initial_timesig', 'N/A')}")
        print(f"  Programmi:       {metadata.get('num_programs', 'N/A')} strumenti")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = build_tokenizer()
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}

    # ── Visualizzazioni ──────────────────────────────────────────────────────
    use_color = not args.no_color

    visualize_sequence(tokens, id_to_token, bar_positions,
                       max_bars=args.bars, use_color=use_color)

    show_type_distribution(tokens, id_to_token)

    show_masking_example(tokens, id_to_token)

    verify_consistency(tokens, id_to_token, tokenizer)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
