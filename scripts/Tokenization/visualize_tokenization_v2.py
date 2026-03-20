"""
Visualizza la tokenizzazione V2 di un file preprocessato (.pt).
Versione V2: tokenizer config aggiornato, vocab=550, pitch-shift check.

Uso:
    python scripts/Tokenization/visualize_tokenization_v2.py
    python scripts/Tokenization/visualize_tokenization_v2.py --file /path/to/file.pt
    python scripts/Tokenization/visualize_tokenization_v2.py --bars 3
    python scripts/Tokenization/visualize_tokenization_v2.py --no-color
"""

import torch
import argparse
import random
import re
from pathlib import Path
from collections import Counter
from miditok import REMI, TokenizerConfig

# V2 expected vocab size
EXPECTED_VOCAB_SIZE = 550
MASK_ID = 3  # V2: PAD=0 BOS=1 EOS=2 MASK=3

# ============================================================================
# ANSI colors
# ============================================================================
COLORS = {
    'pitch':    '\033[94m',
    'pitchdrum':'\033[34m',
    'velocity': '\033[92m',
    'duration': '\033[93m',
    'program':  '\033[95m',
    'position': '\033[96m',
    'bar':      '\033[91m',
    'tempo':    '\033[33m',
    'timesig':  '\033[35m',
    'rest':     '\033[36m',
    'special':  '\033[90m',
    'reset':    '\033[0m',
}

TYPE_LABELS = {
    'pitch':     'PTCH',
    'pitchdrum': 'DRUM',
    'velocity':  'VEL ',
    'duration':  'DUR ',
    'program':   'PROG',
    'position':  'POS ',
    'bar':       'BAR ',
    'tempo':     'TMP ',
    'timesig':   'TSIG',
    'rest':      'REST',
    'special':   'SPEC',
}


# ============================================================================
# Tokenizer V2
# ============================================================================

def build_tokenizer_v2():
    """Tokenizer con configurazione V2 (beat_res esteso, delete_equal_successive)."""
    config = TokenizerConfig(
        pitch_range=(21, 109),
        beat_res={(0, 4): 8, (4, 12): 4},
        num_velocities=32,
        use_chords=False,
        use_rests=True,
        use_tempos=True,
        use_time_signatures=True,
        use_programs=True,
        special_tokens=["PAD", "BOS", "EOS", "MASK"],
        use_controls=False,
        delete_equal_successive_tempo_changes=True,
        delete_equal_successive_time_sig_changes=True,
    )
    return REMI(tokenizer_config=config)


# ============================================================================
# Token classification
# ============================================================================

def classify_token(token_name):
    if token_name.startswith('PitchDrum_'):
        return 'pitchdrum'
    elif token_name.startswith('Pitch_'):
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


# ============================================================================
# Visualization functions (same API as V1)
# ============================================================================

def colorize(text, token_type, use_color):
    if not use_color:
        return text
    color = COLORS.get(token_type, '')
    return f"{color}{text}{COLORS['reset']}"


def visualize_sequence(tokens, id_to_token, bar_positions, max_bars=3, use_color=True):
    print("\n" + "=" * 70)
    print("SEQUENZA TOKEN (bar per bar)")
    print("=" * 70)

    num_bars = len(bar_positions) - 1
    bars_to_show = min(max_bars, num_bars)
    print(f"   Mostrando {bars_to_show} di {num_bars} bar totali\n")

    for bar_idx in range(bars_to_show):
        start = bar_positions[bar_idx]
        end   = bar_positions[bar_idx + 1]
        bar_tokens = tokens[start:end]

        print(f"  +- BAR {bar_idx + 1} (pos {start}-{end-1}, {end-start} token)")
        line = "  |  "
        count = 0
        for tok_id in bar_tokens.tolist():
            tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
            tok_type = classify_token(tok_name)
            label    = TYPE_LABELS.get(tok_type, '????')
            entry    = colorize(f"[{label}:{tok_name}]", tok_type, use_color)
            line += entry + " "
            count += 1
            if count % 6 == 0:
                print(line)
                line = "  |  "
        if line.strip() not in ("|", ""):
            print(line)
        print("  " + "-" * 60)
        print()


def show_type_distribution(tokens, id_to_token):
    print("=" * 70)
    print("DISTRIBUZIONE TIPI DI TOKEN")
    print("=" * 70)

    type_counts = Counter()
    for tok_id in tokens.tolist():
        tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
        tok_type = classify_token(tok_name)
        type_counts[tok_type] += 1

    total = len(tokens)
    bar_width = 30
    ordered = ['bar', 'position', 'program', 'pitch', 'pitchdrum',
               'velocity', 'duration', 'tempo', 'timesig', 'rest', 'special']
    for tok_type in ordered:
        count = type_counts.get(tok_type, 0)
        pct   = count / total * 100 if total > 0 else 0
        bar   = '#' * int(pct / 100 * bar_width)
        label = TYPE_LABELS.get(tok_type, '????')
        print(f"  {label}  {bar:<{bar_width}}  {count:>5} token  ({pct:5.1f}%)")

    print(f"\n  {'TOTALE':30}  {total:>5} token")


def show_masking_example(tokens, id_to_token, mask_ratio=0.15):
    print("\n" + "=" * 70)
    print(f"ESEMPIO MASKING ({mask_ratio*100:.0f}% token sostituiti con [MASK])")
    print("=" * 70)

    preview   = tokens[:60]
    seq_len   = len(preview)
    num_mask  = max(1, int(seq_len * mask_ratio))
    mask_pos  = set(random.sample(range(seq_len), num_mask))

    print(f"\n  Token originali (primi {seq_len}):")
    orig_line = "  "
    for tok_id in preview.tolist():
        tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
        tok_type = classify_token(tok_name)
        orig_line += f"{TYPE_LABELS.get(tok_type, '????')} "
    print(orig_line)

    print(f"\n  Dopo masking ({num_mask} posizioni):")
    mask_line = "  "
    for i, tok_id in enumerate(preview.tolist()):
        if i in mask_pos:
            mask_line += "\033[97m\033[41m[MSK]\033[0m "
        else:
            tok_name = id_to_token.get(tok_id, f"UNK_{tok_id}")
            tok_type = classify_token(tok_name)
            mask_line += f"{TYPE_LABELS.get(tok_type, '????')} "
    print(mask_line)
    print(f"\n  Posizioni mascherate: {sorted(mask_pos)}")


# ============================================================================
# V2-specific checks
# ============================================================================

def verify_consistency_v2(tokens, id_to_token, tokenizer, metadata):
    """Verifica coerenza V2: vocab=550, range [0,549], tokenizer_ver='v2'."""
    print("\n" + "=" * 70)
    print("VERIFICA COERENZA V2")
    print("=" * 70)

    vocab_size = len(tokenizer.vocab)
    max_id     = tokens.max().item()
    min_id     = tokens.min().item()
    unknown_ids = [i for i in tokens.tolist() if i not in id_to_token]

    print(f"\n  Vocab size tokenizer V2: {vocab_size}")
    print(f"  Atteso:                  {EXPECTED_VOCAB_SIZE}")
    if vocab_size == EXPECTED_VOCAB_SIZE:
        print(f"  [OK] vocab_size corretto")
    else:
        print(f"  [!!] vocab_size NON corrisponde (atteso {EXPECTED_VOCAB_SIZE})")

    print(f"\n  Token ID range nel file: [{min_id}, {max_id}]")
    print(f"  Atteso:                  [0, {EXPECTED_VOCAB_SIZE - 1}]")
    if max_id <= EXPECTED_VOCAB_SIZE - 1:
        print(f"  [OK] tutti i token ID sono nel range [0, {EXPECTED_VOCAB_SIZE-1}]")
    else:
        print(f"  [!!] token ID {max_id} fuori range (max atteso {EXPECTED_VOCAB_SIZE-1})")

    if unknown_ids:
        print(f"\n  [!!] {len(unknown_ids)} token ID non decodificabili")
    else:
        print(f"  [OK] tutti i token sono decodificabili")

    # Special tokens V2: PAD=0 BOS=1 EOS=2 MASK=3
    special_ids = {
        'PAD':  tokenizer.vocab.get('PAD_None', None),
        'BOS':  tokenizer.vocab.get('BOS_None', None),
        'EOS':  tokenizer.vocab.get('EOS_None', None),
        'MASK': tokenizer.vocab.get('MASK_None', None),
    }
    print(f"\n  Token speciali V2 (atteso PAD=0 BOS=1 EOS=2 MASK=3):")
    for name, tok_id in special_ids.items():
        present = (tok_id in tokens.tolist()) if tok_id is not None else False
        status  = "presente" if present else "assente "
        ok_mark = "[OK]" if tok_id == ["PAD","BOS","EOS","MASK"].index(name) else "[!!]"
        print(f"    {name:4}: ID={tok_id}  {ok_mark}  ({status})")

    # Metadata checks
    print(f"\n  Metadata:")
    ver = metadata.get('tokenizer_ver', 'N/A')
    shift = metadata.get('pitch_shift', 'N/A')
    print(f"    tokenizer_ver : {ver}  {'[OK]' if ver == 'v2' else '[!!] atteso v2'}")
    print(f"    pitch_shift   : {shift}")
    print(f"    source_file   : {metadata.get('source_file', 'N/A')}")
    print(f"    num_tokens    : {metadata.get('num_tokens', 'N/A')}")
    print(f"    num_bars      : {metadata.get('num_bars', 'N/A')}")


def show_pitch_shift_comparison(file_path, id_to_token, tokenizer, target_shift=3):
    """
    Confronta i pitch token tra shift=0 e shift=+target_shift dello stesso stem.
    Verifica che ogni Pitch_X nel file shift=0 corrisponda a Pitch_(X+target_shift)
    nel file shiftato (o sia clipped a 108).
    """
    print("\n" + "=" * 70)
    print(f"PITCH SHIFT COMPARISON (shift=0 vs shift=+{target_shift})")
    print("=" * 70)

    data_dir = file_path.parent

    # Determine the stem (strip _shift+N or _shiftN suffix)
    name = file_path.stem  # e.g. "mysong_shift+3" or "mysong"
    base_stem = re.sub(r'_shift[+-]?\d+$', '', name)

    shift0_path    = data_dir / f"{base_stem}.pt"
    shiftN_path    = data_dir / f"{base_stem}_shift+{target_shift}.pt"
    shiftN_neg_path= data_dir / f"{base_stem}_shift{target_shift}.pt"  # fallback without +

    # Prefer +3 notation; if not found try without +
    if not shiftN_path.exists() and shiftN_neg_path.exists():
        shiftN_path = shiftN_neg_path

    if not shift0_path.exists():
        print(f"\n  [SKIP] File shift=0 non trovato: {shift0_path.name}")
        print(f"         (prova con --file che punta al file shift=0 dello stesso stem)")
        return

    if not shiftN_path.exists():
        print(f"\n  [SKIP] File shift+{target_shift} non trovato: {shiftN_path.name}")
        print(f"         (il confronto richiede entrambi i file dello stesso stem)")
        return

    data0 = torch.load(shift0_path,  weights_only=False)
    dataN = torch.load(shiftN_path,  weights_only=False)
    tok0  = data0['tokens']
    tokN  = dataN['tokens']

    print(f"\n  File shift=0 :  {shift0_path.name}  ({len(tok0)} token)")
    print(f"  File shift+{target_shift}:  {shiftN_path.name}  ({len(tokN)} token)")
    print(f"  (Lunghezze identiche: {'[OK]' if len(tok0)==len(tokN) else '[!!] DIVERSE'})")

    # Extract pitch token names and IDs (melodic only, not drums)
    def pitch_ids(tokens_tensor):
        result = []
        for tid in tokens_tensor.tolist():
            name = id_to_token.get(tid, '')
            if name.startswith('Pitch_'):
                result.append((tid, name))
        return result

    pitches0 = pitch_ids(tok0)
    pitchesN = pitch_ids(tokN)

    print(f"\n  Pitch tokens  shift=0 : {len(pitches0)}")
    print(f"  Pitch tokens  shift+{target_shift}: {len(pitchesN)}")

    if len(pitches0) != len(pitchesN):
        print(f"\n  [!!] Numero diverso di pitch token — confronto posizione per posizione non affidabile")
        print(f"       (puo' accadere per clipping a bordo del range [21, 108])")

    # Spot-check first N pitches
    n_check = min(10, len(pitches0), len(pitchesN))
    print(f"\n  Spot-check: prime {n_check} coppie di pitch token")
    print(f"  {'#':>3}  {'shift=0':>18}  {'shift+'+str(target_shift):>18}  {'diff':>6}  status")
    print(f"  {'-'*3}  {'-'*18}  {'-'*18}  {'-'*6}  {'-'*10}")

    PITCH_MAX = 108
    PITCH_MIN = 21
    mismatches = 0

    for i in range(n_check):
        id0, name0 = pitches0[i]
        idN, nameN = pitchesN[i]
        val0 = int(name0.split('_')[1])  # e.g. Pitch_60 -> 60
        valN = int(nameN.split('_')[1])
        diff = valN - val0
        # expected: either +target_shift, or clipped at boundary
        clipped = (val0 + target_shift > PITCH_MAX) or (val0 + target_shift < PITCH_MIN)
        if clipped:
            expected_status = f"clipped -> {max(PITCH_MIN, min(PITCH_MAX, val0+target_shift))}"
            ok = (valN == max(PITCH_MIN, min(PITCH_MAX, val0 + target_shift)))
        else:
            expected_status = f"+{target_shift}"
            ok = (diff == target_shift)

        status = "[OK]" if ok else f"[!!] atteso {expected_status}"
        if not ok:
            mismatches += 1
        print(f"  {i+1:>3}  {name0:>18}  {nameN:>18}  {diff:>+6}  {status}")

    if mismatches == 0:
        print(f"\n  [OK] Tutti {n_check} pitch correttamente shiftati di +{target_shift}")
    else:
        print(f"\n  [!!] {mismatches}/{n_check} pitch non correttamente shiftati")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualizza tokenizzazione MIDI V2")
    parser.add_argument('--file', type=str, default=None,
                        help="File .pt (default: random da preprocessed_v2_5k)")
    parser.add_argument('--dir', type=str,
                        default='/nfsd/tesi/zanettisim/MIDIT/data/preprocessed_v2_5k',
                        help="Directory V2 preprocessed")
    parser.add_argument('--bars', type=int, default=3,
                        help="Numero di bar da visualizzare")
    parser.add_argument('--no-color', action='store_true',
                        help="Disabilita colori ANSI")
    parser.add_argument('--shift', type=int, default=3,
                        help="Shift target per il confronto pitch (default: 3)")
    args = parser.parse_args()

    # ── Trova il file ────────────────────────────────────────────────────────
    if args.file:
        file_path = Path(args.file)
    else:
        data_dir  = Path(args.dir)
        all_files = list(data_dir.glob("*.pt"))
        if not all_files:
            print(f"[ERROR] Nessun file .pt trovato in {data_dir}")
            return
        # Prefer a shift=0 file for cleaner output (pitch comparison works both ways)
        shift0_files = [f for f in all_files if 'shift' not in f.stem]
        file_path = random.choice(shift0_files if shift0_files else all_files)

    print(f"\n{'=' * 70}")
    print(f"VISUALIZZAZIONE TOKENIZZAZIONE MIDI V2")
    print(f"{'=' * 70}")
    print(f"\n  File: {file_path}")

    # ── Carica dati ──────────────────────────────────────────────────────────
    data         = torch.load(file_path, weights_only=False)
    tokens       = data['tokens']
    bar_positions = data['bar_positions']
    metadata     = data.get('metadata', {})

    print(f"  Token totali:     {len(tokens)}")
    print(f"  Bar totali:       {len(bar_positions) - 1}")
    print(f"  ID range:         [{tokens.min().item()}, {tokens.max().item()}]")
    if metadata:
        print(f"  Source file:      {metadata.get('source_file', 'N/A')}")
        print(f"  Pitch shift:      {metadata.get('pitch_shift', 'N/A')}")
        print(f"  Tokenizer ver:    {metadata.get('tokenizer_ver', 'N/A')}")
        print(f"  Num programs:     {metadata.get('num_programs', 'N/A')}")

    # ── Tokenizer V2 ─────────────────────────────────────────────────────────
    tokenizer    = build_tokenizer_v2()
    id_to_token  = {v: k for k, v in tokenizer.vocab.items()}
    use_color    = not args.no_color

    # ── Sezioni ──────────────────────────────────────────────────────────────
    visualize_sequence(tokens, id_to_token, bar_positions,
                       max_bars=args.bars, use_color=use_color)

    show_type_distribution(tokens, id_to_token)

    show_masking_example(tokens, id_to_token)

    verify_consistency_v2(tokens, id_to_token, tokenizer, metadata)

    show_pitch_shift_comparison(file_path, id_to_token, tokenizer,
                                target_shift=args.shift)

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    main()
