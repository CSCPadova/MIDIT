"""
Visualizza la tokenizzazione V2 di un file preprocessato (.pt).
Versione V2: tokenizer config aggiornato, vocab=550, pitch-shift check.

Uso:
    python scripts/preprocessing/visualize_tokenization_v2.py
    python scripts/preprocessing/visualize_tokenization_v2.py --file /path/to/file.pt
    python scripts/preprocessing/visualize_tokenization_v2.py --bars 3
    python scripts/preprocessing/visualize_tokenization_v2.py --no-color
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
# Matplotlib figure — MidiTok-style colored token boxes
# ============================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

BOX_COLORS = {
    'bar':       '#90EE90',  # green
    'position':  '#AED6F1',  # sky blue
    'tempo':     '#FFB347',  # orange
    'timesig':   '#D7BDE2',  # lavender / purple
    'program':   '#76D7C4',  # teal
    'pitch':     '#F1948A',  # salmon / pink-red
    'pitchdrum': '#F1948A',  # same head as pitch
    'velocity':  '#F9E79F',  # pale yellow
    'duration':  '#D2B48C',  # tan / beige-brown
    'rest':      '#BDC3C7',  # silver-grey
    'special':   '#E8E8E8',  # very light grey
}


def midi_to_note(n: int) -> str:
    """Convert MIDI pitch number to note name (e.g. 60 → C4)."""
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def format_display_name(name: str) -> str:
    """Convert an internal MidiTok token name to a two-line display label.

    Examples:
        Bar_None       → 'Bar'
        Position_7     → 'Pos.\\n7'
        Tempo_116      → 'Tempo\\n116'
        TimeSig_4/4    → 'TimeSig\\n4/4'
        Program_0      → 'Prog.\\n0'
        Pitch_60       → 'Pitch\\nC4'
        PitchDrum_36   → 'Drum\\n36'
        Velocity_80    → 'Vel.\\n80'
        Duration_1.0.8 → 'Dur.\\n1.0'
        Rest_0.4.8     → 'Rest\\n0.4'
    """
    if name == 'Bar_None':
        return 'Bar'
    if name.startswith('Position_'):
        return f'Pos.\n{name[9:]}'
    if name.startswith('Tempo_'):
        return f'Tempo\n{name[6:]}'
    if name.startswith('TimeSig_'):
        val = name[8:].split('_')[0]   # drop trailing index if present
        return f'TimeSig\n{val}'
    if name.startswith('Program_'):
        return f'Prog.\n{name[8:]}'
    if name.startswith('Pitch_'):
        try:
            return f'Pitch\n{midi_to_note(int(name[6:]))}'
        except ValueError:
            return f'Pitch\n{name[6:]}'
    if name.startswith('PitchDrum_'):
        return f'Drum\n{name[10:]}'
    if name.startswith('Velocity_'):
        return f'Vel.\n{name[9:]}'
    if name.startswith('Duration_'):
        parts = name[9:].split('.')
        suffix = f'{parts[0]}.{parts[1]}' if len(parts) >= 2 else name[9:]
        return f'Dur.\n{suffix}'
    if name.startswith('Rest_'):
        parts = name[5:].split('.')
        suffix = f'{parts[0]}.{parts[1]}' if len(parts) >= 2 else name[5:]
        return f'Rest\n{suffix}'
    for special in ('PAD_None', 'BOS_None', 'EOS_None', 'MASK_None'):
        if name == special:
            return name.split('_')[0]
    return name.replace('_', '\n', 1)


def render_token_figure(
    tokens,
    id_to_token: dict,
    bar_positions: list,
    max_tokens: int = 28,
    output_path: str = None,
) -> None:
    """Render a MidiTok-style horizontal token-box figure with matplotlib.

    Each token is a rounded colored rectangle with the token name inside.
    Bar boundaries are marked with a thin vertical separator.
    A continuation arrow is drawn at the right end.

    Args:
        tokens:        Token ID tensor.
        id_to_token:   {token_id: token_name} mapping.
        bar_positions: List of bar-boundary token indices.
        max_tokens:    How many tokens to render (default 28).
        output_path:   Save path for PNG.  None → plt.show() interactively.
    """
    try:
        import matplotlib
        if output_path:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] matplotlib not installed.  pip install matplotlib")
        return

    # ── Layout constants (data coordinates) ──────────────────────────────────
    BOX_W = 0.82    # box width
    BOX_H = 2.00    # box height
    GAP   = 0.18    # gap between adjacent boxes
    STEP  = BOX_W + GAP   # horizontal step per token = 1.00
    PAD_L = 0.30    # left margin
    PAD_R = 1.30    # right margin (room for arrow)
    PAD_Y = 0.30    # bottom margin; top margin = PAD_Y + 0.30 for bar labels

    token_ids = tokens[:max_tokens].tolist()
    n = len(token_ids)

    total_w = PAD_L + n * STEP + PAD_R
    total_h = PAD_Y + BOX_H + 0.55     # extra headroom for bar labels

    # Figure size: ~0.58 in per token horizontally, fixed 2.3 in height
    fig_w = max(6.0, PAD_L * 0.58 + n * STEP * 0.58 + PAD_R * 0.58)
    fig_h = 2.30

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    bar_boundary_set = set(bar_positions)
    bar_counter = 1     # first bar label written at the left margin

    y_box = PAD_Y       # bottom-left y of every box

    # "bar 1" label at leftmost edge
    ax.text(
        PAD_L + BOX_W / 2, PAD_Y + BOX_H + 0.10,
        'bar 1',
        ha='center', va='bottom',
        fontsize=6.0, color='#555555',
        fontstyle='italic',
    )

    for i, tok_id in enumerate(token_ids):
        tok_name = id_to_token.get(tok_id, f'UNK_{tok_id}')
        tok_type = classify_token(tok_name)
        color    = BOX_COLORS.get(tok_type, '#FFFFFF')
        display  = format_display_name(tok_name)

        x_box = PAD_L + i * STEP

        # ── Bar separator and label ───────────────────────────────────────────
        if i > 0 and i in bar_boundary_set:
            bar_counter += 1
            sep_x = x_box - GAP / 2
            ax.plot(
                [sep_x, sep_x],
                [PAD_Y - 0.04, PAD_Y + BOX_H + 0.04],
                color='#777777', linewidth=0.85,
            )
            ax.text(
                x_box + BOX_W / 2, PAD_Y + BOX_H + 0.10,
                f'bar {bar_counter}',
                ha='center', va='bottom',
                fontsize=6.0, color='#555555',
                fontstyle='italic',
            )

        # ── Rounded box ───────────────────────────────────────────────────────
        box = mpatches.FancyBboxPatch(
            (x_box, y_box), BOX_W, BOX_H,
            boxstyle='round,pad=0.07',
            facecolor=color,
            edgecolor='#555555',
            linewidth=0.75,
        )
        ax.add_patch(box)

        # ── Token name (two lines) ────────────────────────────────────────────
        ax.text(
            x_box + BOX_W / 2,
            y_box + BOX_H / 2,
            display,
            ha='center', va='center',
            fontsize=7.5, fontweight='bold',
            color='#222222',
            multialignment='center',
            linespacing=1.35,
        )

    # ── Continuation arrow ────────────────────────────────────────────────────
    arrow_x0 = PAD_L + n * STEP - GAP / 2 + 0.12
    arrow_x1 = arrow_x0 + 0.85
    mid_y    = PAD_Y + BOX_H / 2
    ax.annotate(
        '',
        xy=(arrow_x1, mid_y),
        xytext=(arrow_x0, mid_y),
        arrowprops=dict(
            arrowstyle='->', color='#444444',
            lw=1.6, mutation_scale=14,
        ),
    )

    plt.tight_layout(pad=0.1)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"  Figure saved -> {out.resolve()}")
        plt.close(fig)
    else:
        plt.show()


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
    parser.add_argument('--output', type=str, default=None,
                        help="Save matplotlib token-box figure to this PNG path.")
    parser.add_argument('--max_tokens', type=int, default=28,
                        help="Number of tokens to render in the figure (default: 28).")
    parser.add_argument('--figure-only', action='store_true',
                        help="Skip terminal output; only generate the figure (requires --output).")
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

    if not args.figure_only:
        print(f"\n{'=' * 70}")
        print(f"VISUALIZZAZIONE TOKENIZZAZIONE MIDI V2")
        print(f"{'=' * 70}")
        print(f"\n  File: {file_path}")

    # ── Carica dati ──────────────────────────────────────────────────────────
    data          = torch.load(file_path, weights_only=False)
    tokens        = data['tokens']
    bar_positions = data['bar_positions']
    metadata      = data.get('metadata', {})

    if not args.figure_only:
        print(f"  Token totali:     {len(tokens)}")
        print(f"  Bar totali:       {len(bar_positions) - 1}")
        print(f"  ID range:         [{tokens.min().item()}, {tokens.max().item()}]")
        if metadata:
            print(f"  Source file:      {metadata.get('source_file', 'N/A')}")
            print(f"  Pitch shift:      {metadata.get('pitch_shift', 'N/A')}")
            print(f"  Tokenizer ver:    {metadata.get('tokenizer_ver', 'N/A')}")
            print(f"  Num programs:     {metadata.get('num_programs', 'N/A')}")

    # ── Tokenizer V2 ─────────────────────────────────────────────────────────
    tokenizer   = build_tokenizer_v2()
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    use_color   = not args.no_color

    # ── Terminal sections (skipped in figure-only mode) ───────────────────────
    if not args.figure_only:
        visualize_sequence(tokens, id_to_token, bar_positions,
                           max_bars=args.bars, use_color=use_color)

        show_type_distribution(tokens, id_to_token)

        show_masking_example(tokens, id_to_token)

        verify_consistency_v2(tokens, id_to_token, tokenizer, metadata)

        show_pitch_shift_comparison(file_path, id_to_token, tokenizer,
                                    target_shift=args.shift)

        print(f"\n{'=' * 70}\n")

    # ── Matplotlib figure ─────────────────────────────────────────────────────
    if args.output or args.figure_only:
        render_token_figure(
            tokens,
            id_to_token,
            bar_positions,
            max_tokens=args.max_tokens,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
