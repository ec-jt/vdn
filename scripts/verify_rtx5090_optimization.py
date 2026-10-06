"""CPU-only exact latent and full-media acceptance for a fixed 124-frame run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--reference', default='baseline124-plain-20261005T0734Z')
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = Path('artifacts/rtx5090') / args.label
    reference = Path('artifacts/latents') / (args.reference + '.mp4.latents.pt')
    latents = Path('artifacts/latents') / (args.label + '.mp4.latents.pt')
    a = torch.load(reference, map_location='cpu', weights_only=True)
    b = torch.load(latents, map_location='cpu', weights_only=True)
    assert set(a) == set(b) == {'video', 'audio'}
    exact = {}
    for name in a:
        assert a[name].dtype == b[name].dtype and a[name].shape == b[name].shape
        assert torch.isfinite(b[name]).all()
        exact[name] = torch.equal(a[name].contiguous().view(torch.uint8),
                                  b[name].contiguous().view(torch.uint8))
        assert exact[name], name
    with (directory / ('exact-vs-' + args.reference + '.json')).open('x') as f:
        json.dump(dict(reference=str(reference), candidate=str(latents), bitwise=exact), f, indent=2)
    if not (directory / 'media-check.json').exists():
        subprocess.run([sys.executable, 'scripts/verify_rtx5090_media.py',
                        'results/' + args.label + '.mp4', '--frames', '124',
                        '--latents', str(latents), '--report', str(directory / 'media-check.json'),
                        '--contact', str(directory / 'contact.jpg')], check=True)
    print(json.dumps(dict(label=args.label, reference=args.reference, bitwise=exact)))


if __name__ == '__main__':
    main()
