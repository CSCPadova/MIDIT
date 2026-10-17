"""scripts/inference/inspect_infill.py — Compare original vs MIDIT notes in the masked bars.

Reads original.mid and reconstructed.mid written by infer_from_midi.py (the 8-bar
window, starting at tick 0) and prints, for each onset in the masked bars, the
original and the MIDIT notes, followed by a short summary. Writes no files.

    python scripts/inference/inspect_infill.py --output_dir outputs/example/ --mask_bars 3 4 5
"""
import argparse, sys
from collections import Counter, defaultdict
from pathlib import Path

from symusic import Score

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_name(pitch):
    return f'{PITCH_CLASSES[pitch % 12]}{pitch // 12 - 1}'


def load_score(path):
    # From bytes: loading from a path makes some symusic builds print debug lines.
    return Score.from_midi(Path(path).read_bytes())


def bar_boundaries(score, n_bars):
    """Tick boundaries of the first `n_bars` bars, following the file's time signatures."""
    tpq = score.ticks_per_quarter
    time_sigs = sorted(score.time_signatures, key=lambda ts: ts.time)
    bounds, t = [0], 0
    for _ in range(n_bars):
        current = [ts for ts in time_sigs if ts.time <= t]
        num, den = (current[-1].numerator, current[-1].denominator) if current else (4, 4)
        t += tpq * 4 * num // den
        bounds.append(t)
    return bounds


def collect_notes(score, bars, bounds):
    """(bar, onset, pitch, program, is_drum) for every note starting in `bars`."""
    notes = []
    for track in score.tracks:
        for n in track.notes:
            for b in bars:
                if bounds[b] <= n.time < bounds[b + 1]:
                    notes.append((b, n.time, n.pitch, track.program, track.is_drum))
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
                        help='Folder with original.mid and reconstructed.mid from infer_from_midi.py')
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

    bounds = bar_boundaries(orig, max(bars) + 1)
    if min(bars) < 0 or bounds[max(bars)] >= orig.end():
        sys.exit(f'--mask_bars {bars} out of range for {out / "original.mid"}')

    orig_notes = collect_notes(orig, bars, bounds)
    recon_notes = collect_notes(recon, bars, bounds)
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
          + ', '.join(f'{b}:[{bounds[b]},{bounds[b + 1]})' for b in bars))
    print()
    print('Per onset  (note:program, "drums" for drum notes; notes sorted by pitch)')
    beat_ticks = {b: tpq * 4 // next((t.denominator for t in reversed(ts) if t.time <= bounds[b]), 4)
                  for b in bars}
    exact = pc = pairs = unpaired = 0
    for (bar, onset), (o, r) in sorted(by_onset.items()):
        o.sort(key=lambda n: (n[4], n[2], n[3]))
        r.sort(key=lambda n: (n[4], n[2], n[3]))
        beat = 1 + (onset - bounds[bar]) / beat_ticks[bar]
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
        keys = [(n[1], n[2], n[3], n[4]) for n in notes]
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
