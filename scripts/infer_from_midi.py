"""scripts/infer_from_midi.py — Run MIDIT bar infilling on a raw .mid file."""
import argparse, sys, tempfile, torch, io
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description='MIDIT bar infilling from raw MIDI')
    parser.add_argument('--midi_path',  required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--mask_bars',  type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--start_bar',  type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('[1/5] Preprocessing MIDI (shift=0)...')
    _init_worker()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _process_file((args.midi_path, tmpdir, [0], False))
        if result['status'] != 'ok':
            raise RuntimeError(
                f"Preprocessing failed ({result['status']}): {result.get('error')}\n"
                "  F1=sparse(<50 notes), F2=short(<15s), F3=constant velocity"
            )
        pt_path = Path(tmpdir) / f'{Path(args.midi_path).stem}.pt'
        print(f'[2/5] Extracting 8-bar chunk (start_bar={args.start_bar})...')
        tokens, bar_positions, total_bars = _extract_chunk(pt_path, args.start_bar)

    print('[3/5] Loading model...')
    model = MaskedTransformerREMI.load_from_checkpoint(args.checkpoint)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f'  Device: {device}  |  Params: {sum(p.numel() for p in model.parameters()):,}')

    print('[4/5] Applying masking...')
    tokenizer = build_tokenizer()
    lookups   = build_vocab_lookups(tokenizer)
    masked_tokens, mask_labels = apply_bar_mask(tokens, bar_positions, args.mask_bars, lookups)
    print(f'  Masked {mask_labels.sum().item()} tokens in bars {args.mask_bars}')

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
    print(f'  Input        : {Path(args.midi_path).name}')
    print(f'  Total bars   : {total_bars}  |  Chunk: bars {args.start_bar}-{args.start_bar + 7}')
    print(f'  Chunk tokens : {len(tokens)}  |  Notes in chunk: {n_notes}')
    print(f'  Bars masked  : {args.mask_bars}')
    print(f'  Output dir   : {output_dir.resolve()}')
    for f in ('original.mid', 'masked.mid', 'reconstructed.mid'):
        print(f'    {f}')
    print('=' * 52)


if __name__ == '__main__':
    main()
