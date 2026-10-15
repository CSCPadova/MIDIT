"""scripts/infer_from_midi.py — Run MIDIT bar infilling on a raw .mid file.

Command line:
    python scripts/infer_from_midi.py --midi_path song.mid \
        --checkpoint logs/checkpoints/midit-infilling-epoch05.ckpt \
        --mask_bars 3 4 5 --start_bar 0 --output_dir outputs/my_song/

From Python (e.g. a demo), with the repository root on sys.path:
    from scripts.infer_from_midi import infill_midi
    paths = infill_midi('song.mid', 'logs/checkpoints/midit-infilling-epoch05.ckpt',
                        mask_bars=[3, 4, 5], start_bar=0, output_dir='outputs/my_song')
"""
import argparse, sys, tempfile, torch, io
from pathlib import Path

if __name__ == '__main__':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'Pre_Processing'))

from preprocess_v2 import _init_worker, _process_file
from midit.infer import build_tokenizer, build_vocab_lookups, apply_bar_mask, reconstruct, save_midi
from midit.model.masked_transformer import MaskedTransformerREMI


def _extract_chunk(pt_path, start_bar, bars_per_chunk=8):
    data = torch.load(pt_path, weights_only=False)
    bar_positions = data['bar_positions']
    num_bars = len(bar_positions) - 1
    if num_bars < start_bar + bars_per_chunk:
        raise ValueError(
            f"File has {num_bars} bars -- cannot extract {bars_per_chunk} bars "
            f"starting at bar {start_bar}."
        )
    t0 = bar_positions[start_bar]
    t1 = bar_positions[start_bar + bars_per_chunk]
    return (
        data['tokens'][t0:t1],
        [p - t0 for p in bar_positions[start_bar: start_bar + bars_per_chunk + 1]],
        num_bars,
    )


def infill_midi(midi_path, checkpoint, mask_bars=(3, 4, 5), start_bar=0,
                output_dir='outputs/infilling', skip_filters=False):
    """Infill `mask_bars` (0-indexed within the 8-bar window at `start_bar`) of a MIDI file.

    Writes original.mid, masked.mid and reconstructed.mid to `output_dir` and
    returns their paths as a dict keyed by file name.
    """
    mask_bars = list(mask_bars)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[1/5] Preprocessing MIDI (shift=0)...')
    _init_worker()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _process_file((midi_path, tmpdir, [0], False))
        if result['status'] != 'ok':
            if skip_filters:
                print(f"  Warning: filter {result['status']} bypassed (--skip_filters active)")
            else:
                raise RuntimeError(
                    f"Preprocessing failed ({result['status']}): {result.get('error')}\n"
                    "  F1=sparse(<50 notes), F2=short(<15s), F3=constant velocity"
                )
        pt_path = Path(tmpdir) / f'{Path(midi_path).stem}.pt'
        print(f'[2/5] Extracting 8-bar chunk (start_bar={start_bar})...')
        tokens, bar_positions, total_bars = _extract_chunk(pt_path, start_bar)

    print('[3/5] Loading model...')
    model = MaskedTransformerREMI.load_from_checkpoint(checkpoint)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'  Device: {device}  |  Params: {sum(p.numel() for p in model.parameters()):,}')

    print('[4/5] Applying masking...')
    tokenizer = build_tokenizer()
    lookups   = build_vocab_lookups(tokenizer)
    masked_tokens, mask_labels = apply_bar_mask(tokens, bar_positions, mask_bars, lookups)
    print(f'  Masked {mask_labels.sum().item()} tokens in bars {mask_bars}')

    print('[5/5] Running inference...')
    recon_tokens = reconstruct(model, tokens, masked_tokens, mask_labels, lookups, device)

    for fname, tok in [
        ('original.mid',      tokens),
        ('masked.mid',        masked_tokens),
        ('reconstructed.mid', recon_tokens),
    ]:
        save_midi(tok, tokenizer, output_dir / fname)

    id_to_token = lookups['id_to_token']
    n_notes = sum(
        1 for t in tokens.tolist()
        if id_to_token.get(t, '').startswith(('Pitch_', 'PitchDrum_'))
    )
    print()
    print('=' * 52)
    print('  SUMMARY')
    print(f'  Input        : {Path(midi_path).name}')
    print(f'  Total bars   : {total_bars}  |  Chunk: bars {start_bar}-{start_bar + 7}')
    print(f'  Chunk tokens : {len(tokens)}  |  Notes in chunk: {n_notes}')
    print(f'  Bars masked  : {mask_bars}')
    print(f'  Output dir   : {output_dir.resolve()}')
    for f in ('original.mid', 'masked.mid', 'reconstructed.mid'):
        print(f'    {f}')
    print('=' * 52)

    return {f: output_dir / f for f in ('original.mid', 'masked.mid', 'reconstructed.mid')}


def main():
    parser = argparse.ArgumentParser(description='MIDIT bar infilling from raw MIDI')
    parser.add_argument('--midi_path',  required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mask_bars',  type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--start_bar',  type=int, default=0)
    parser.add_argument('--skip_filters', action='store_true',
                        help='Bypass F1/F2/F3 quality filters from preprocess_v2')
    args = parser.parse_args()

    infill_midi(args.midi_path, args.checkpoint, mask_bars=args.mask_bars,
                start_bar=args.start_bar, output_dir=args.output_dir,
                skip_filters=args.skip_filters)


if __name__ == '__main__':
    main()
