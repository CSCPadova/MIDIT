"""scripts/inference/inspect_infill.py — Compare original vs MIDIT notes in the masked bars.

Reads original.mid and reconstructed.mid written by infer_from_midi.py (the 8-bar
window, starting at tick 0) and prints, for each onset in the masked bars, the
original and the MIDIT notes, followed by a short summary. Writes no files.

    python scripts/inference/inspect_infill.py --output_dir outputs/example/ --mask_bars 3 4 5

The bars are the window's Bar tokens, not musical bars: a long silence is one Rest
token, so one window bar can span several musical bars, and because MidiTok times a
Rest from the end of the previous note, reconstructed.mid can have a slightly
different timeline from original.mid. The bar boundaries of each file are therefore
taken from its token ids in tokens.json (written by infer_from_midi.py).
"""
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from miditok import TokSequence
from symusic import Score

from midit.infer import build_tokenizer

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_name(pitch):
    return f'{PITCH_CLASSES[pitch % 12]}{pitch // 12 - 1}'


def load_score(path):
    # From bytes: loading from a path makes some symusic builds print debug lines.
    return Score.from_midi(Path(path).read_bytes())


def bar_ticks(ids, tokenizer):
    """Tick of every Bar token in `ids`, plus the tick at which the next bar would start.

    Each tick is read from MidiTok's own decoder: the prefix ending at that Bar token
    is decoded with one marker note at position 0 on an unused program. The marker is
    the last event, so it cannot change any earlier timing.
    """
    vocab = tokenizer.vocab
    id_to_token = {v: k for k, v in vocab.items()}
    bar_id = vocab['Bar_None']
    ids = [int(i) for i in ids]
    used = {int(id_to_token[i].split('_')[1]) for i in ids if id_to_token.get(i, '').startswith('Program_')}
    program = next(p for p in range(127, -1, -1) if p not in used)
    velocity = next(k for k in vocab if k.startswith('Velocity_'))
    duration = min((k for k in vocab if k.startswith('Duration_')), key=vocab.get)
    marker = [vocab['Position_0'], vocab[f'Program_{program}'], vocab['Pitch_60'],
              vocab[velocity], vocab[duration]]
    ticks = []
    for end in [i + 1 for i, t in enumerate(ids) if t == bar_id] + [None]:
        prefix = ids[:end] if end is not None else ids + [bar_id]
        score = tokenizer(TokSequence(ids=prefix + marker))
        ticks.append(next(n.time for t in score.tracks
                          if t.program == program and not t.is_drum for n in t.notes))
    return ticks


def window_grids(out):
    """Bar ticks of original.mid and reconstructed.mid, and a warning if approximated."""
    tokenizer = build_tokenizer()
    tokens_path = out / 'tokens.json'
    if tokens_path.exists():
        data = json.loads(tokens_path.read_text())
        return bar_ticks(data['original'], tokenizer), bar_ticks(data['reconstructed'], tokenizer), None
    # Older output folders: re-tokenizing original.mid reproduces its bars, but the bars
    # of reconstructed.mid can only be approximated by those of original.mid.
    grid = bar_ticks(tokenizer(load_score(out / 'original.mid')).ids, tokenizer)
    return grid, grid, ('tokens.json not found (older infer_from_midi.py output): the bars of '
                        'reconstructed.mid are taken from original.mid, which can be off after a '
                        'long rest that follows a masked note.')


def collect_notes(score, bars, bounds):
    """(bar, offset in bar, pitch, program, is_drum) for every note starting in `bars`."""
    notes = []
    for track in score.tracks:
        for n in track.notes:
            for b in bars:
                if bounds[b] <= n.time < bounds[b + 1]:
                    notes.append((b, n.time - bounds[b], n.pitch, track.program, track.is_drum))
    return notes


def track_names(*scores):
    names = {}
    for score in scores:
        for track in score.tracks:
            names.setdefault('drums' if track.is_drum else track.program, track.name)
    return names


def fmt(note):
    _, _, pitch, program, is_drum = note
    return f'{note_name(pitch)}:{"drums" if is_drum else f"p{program}"}'


def pct(k, n):
    return f'{k}/{n} ({100 * k / n:.1f}%)' if n else f'{k}/{n} (n/a)'


def main():
    parser = argparse.ArgumentParser(
        description='Print original vs MIDIT notes in the masked bars of an '
                    'infer_from_midi.py output folder (writes no files)')
    parser.add_argument('--output_dir', required=True,
                        help='Folder with original.mid, reconstructed.mid and tokens.json from infer_from_midi.py')
    parser.add_argument('--mask_bars', type=int, nargs='+', default=[3, 4, 5],
                        help='Masked bars, 0-indexed within the 8-bar window (default: 3 4 5)')
    args = parser.parse_args()

    out = Path(args.output_dir)
    orig = load_score(out / 'original.mid')
    recon = load_score(out / 'reconstructed.mid')
    bars = sorted(set(args.mask_bars))
    tpq = orig.ticks_per_quarter
    if recon.ticks_per_quarter != tpq:
        sys.exit(f'ticks_per_quarter differs: original {tpq}, reconstructed {recon.ticks_per_quarter}')

    grid_o, grid_r, warning = window_grids(out)
    if min(bars) < 0 or max(bars) + 1 >= len(grid_o):
        sys.exit(f'--mask_bars {bars} out of range for {out / "original.mid"} ({len(grid_o) - 1} bars)')

    orig_notes = collect_notes(orig, bars, grid_o)
    recon_notes = collect_notes(recon, bars, grid_r)
    names = track_names(orig, recon)

    by_onset = defaultdict(lambda: ([], []))
    for n in orig_notes:
        by_onset[(n[0], n[1])][0].append(n)
    for n in recon_notes:
        by_onset[(n[0], n[1])][1].append(n)

    ts = sorted(orig.time_signatures, key=lambda t: t.time)
    ts_str = ', '.join(f'{t.numerator}/{t.denominator}@{t.time}' for t in ts) or '4/4 (default)'
    print(f'Folder         : {out}')
    print(f'Masked bars    : {bars} (0-indexed within the 8-bar window)')
    print(f'Time signatures: {ts_str}  |  tpq {tpq}  |  bar ticks '
          + ', '.join(f'{b}:[{grid_o[b]},{grid_o[b + 1]})' for b in bars))
    if grid_r != grid_o:
        print('                 reconstructed.mid bar ticks '
              + ', '.join(f'{b}:[{grid_r[b]},{grid_r[b + 1]})' for b in bars))
    if warning:
        print(f'Warning        : {warning}')
    print()
    print('Per onset  (note:program, "drums" for drum notes; notes sorted by pitch)')
    beat_ticks = {b: tpq * 4 // next((t.denominator for t in reversed(ts) if t.time <= grid_o[b]), 4)
                  for b in bars}
    exact = pc = pairs = unpaired = 0
    for (bar, offset), (o, r) in sorted(by_onset.items()):
        o.sort(key=lambda n: (n[4], n[2], n[3]))
        r.sort(key=lambda n: (n[4], n[2], n[3]))
        beat = 1 + offset / beat_ticks[bar]
        print(f'  bar {bar} beat {beat:<6g} | orig : {" ".join(fmt(n) for n in o) or "-"}')
        print(f'  {"":17} | MIDIT: {" ".join(fmt(n) for n in r) or "-"}')
        # Pair pitched notes by pitch order within the onset; drums are not compared.
        op = [n[2] for n in o if not n[4]]
        rp = [n[2] for n in r if not n[4]]
        for a, b in zip(op, rp):
            pairs += 1
            exact += a == b
            pc += a % 12 == b % 12
        unpaired += abs(len(op) - len(rp))

    def pitched(notes):
        return [n for n in notes if not n[4]]

    def onsets(notes):
        return {(n[0], n[1]) for n in notes}

    print()
    print('Summary (masked bars only)')
    print(f'  Notes          : original {len(orig_notes)} '
          f'({len(orig_notes) - len(pitched(orig_notes))} drums)  |  '
          f'MIDIT {len(recon_notes)} ({len(recon_notes) - len(pitched(recon_notes))} drums)')
    print(f'  Onsets         : original {len(onsets(orig_notes))}  |  MIDIT {len(onsets(recon_notes))}  |  '
          f'identical sets: {"yes" if onsets(orig_notes) == onsets(recon_notes) else "no"}')
    print(f'  Exact pitch    : {pct(exact, pairs)}')
    print(f'  Pitch class    : {pct(pc, pairs)}')
    print(f'  Unpaired notes : {unpaired}')
    print('  Note: notes are paired by pitch order within each onset (pitched notes only),')
    print('        so these agreement figures are indicative, not the thesis metrics.')

    o_pc = Counter(PITCH_CLASSES[n[2] % 12] for n in pitched(orig_notes))
    r_pc = Counter(PITCH_CLASSES[n[2] % 12] for n in pitched(recon_notes))
    print('  Top-5 pitch classes (pitched notes):')
    print('    original: ' + '  '.join(f'{k} {v}' for k, v in o_pc.most_common(5)))
    print('    MIDIT   : ' + '  '.join(f'{k} {v}' for k, v in r_pc.most_common(5)))

    def n_duplicates(notes):
        keys = [(n[0], n[1], n[2], n[3], n[4]) for n in notes]
        return len(keys) - len(set(keys))

    print(f'  Duplicate notes (same pitch + program + onset): MIDIT {n_duplicates(recon_notes)}  '
          f'(original {n_duplicates(orig_notes)})')

    def prog_key(n):
        return 'drums' if n[4] else n[3]

    o_prog = Counter(prog_key(n) for n in orig_notes)
    r_prog = Counter(prog_key(n) for n in recon_notes)
    print('  Programs        original  MIDIT')
    for key in sorted(set(o_prog) | set(r_prog), key=lambda k: (k == 'drums', k if k != 'drums' else 0)):
        label = 'drums' if key == 'drums' else f'p{key} {names.get(key, "")}'
        print(f'    {label:<26} {o_prog[key]:>4}  {r_prog[key]:>5}')


if __name__ == '__main__':
    main()
