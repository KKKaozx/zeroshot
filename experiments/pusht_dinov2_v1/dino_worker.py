"""Frozen DINO image worker; the original fixed CLIP text is read byte-for-byte."""
import argparse
import json
from pathlib import Path
import socket

import numpy as np
from wire import receive, send


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fd', type=int, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--text', type=Path, required=True)
    args = parser.parse_args()
    sock = socket.socket(fileno=args.fd)
    sock.settimeout(300)
    try:
        import torch
        from frozen_visual import FrozenDinoVisual, model_sha, file_sha
        p = json.loads(Path(__file__).with_name('protocol.json').read_text())
        if not torch.cuda.is_available():
            raise ValueError('DINO worker requires the existing project CUDA environment')
        if file_sha(args.text) != p['text_sha256']:
            raise ValueError('Original fixed CLIP text bytes changed')
        encoder = FrozenDinoVisual(args.assets)
        before = model_sha(encoder.model)
        if before != p['model_state_sha256']:
            raise ValueError('DINO resource-qualified model state differs')
        with np.load(args.text, allow_pickle=False) as f:
            text = {k: f[k].copy() for k in ('text', 'text_mask')}
        identity = dict(backbone_weights_sha256=p['backbone_weights_sha256'],
                        model_state_sha256=before, text_sha256=p['text_sha256'],
                        preprocessing=p['preprocessing'], all_frozen=True)
        send(sock, dict(status='ready', **identity), text)
        count = 0
        while True:
            metadata, arrays = receive(sock)
            request_id = metadata['request_id']
            operation = metadata['operation']
            if operation == 'close':
                unchanged = model_sha(encoder.model) == before
                frozen = all(not x.requires_grad and x.grad is None for x in encoder.parameters())
                if not unchanged or not frozen:
                    raise ValueError('Frozen DINO weights/gradients changed')
                send(sock, dict(status='closed', request_id=request_id, requests=count,
                    weights_unchanged=True, peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                    **identity))
                break
            if operation != 'encode' or set(arrays) != {'raw_images'}:
                raise ValueError('Unexpected DINO worker request')
            visual = encoder(arrays['raw_images'])
            count += 1
            send(sock, dict(status='ok', request_id=request_id), {'visual': visual})
    except Exception as exc:
        try:
            send(sock, dict(status='error', error=type(exc).__name__+': '+str(exc)))
        finally:
            sock.close()
        raise
    finally:
        sock.close()


if __name__ == '__main__':
    main()
