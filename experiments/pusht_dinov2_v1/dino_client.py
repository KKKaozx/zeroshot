"""Use the qualified numeric IPC transport with a new frozen DINO worker."""
import json
import os
from pathlib import Path
import socket
import subprocess

import numpy as np
from cache_tools import sha
from live_client import CLIPClient, LiveProjectPolicy
from wire import receive


class DINOClient(CLIPClient):
    # encode/close/abort and the actual policy callback remain the old implementation.
    def __init__(self, root, out):
        self.request_id = 0
        self.close_report = None
        self.latencies = []
        bundle = Path(__file__).resolve().parent
        p = json.loads((bundle/'protocol.json').read_text())
        text = root/'runs/pusht-project-cache-213439/text.npz'
        if sha(text) != p['text_sha256']:
            raise ValueError('Original CLIP text bytes changed')
        parent, child = socket.socketpair()
        self.sock = parent
        parent.settimeout(300)
        project = root/'envs/multitask-preflight-v1'
        environment = os.environ.copy()
        environment['LD_LIBRARY_PATH'] = str(project/'lib')
        self.log = (out/'dino_worker.log').open('xb')
        try:
            self.process = subprocess.Popen([str(project/'bin/python'), '-B',
                str(bundle/'dino_worker.py'), '--fd', str(child.fileno()),
                '--assets', str(root/p['assets_directory']), '--text', str(text)],
                pass_fds=(child.fileno(),), env=environment,
                stdout=self.log, stderr=subprocess.STDOUT)
        except Exception:
            parent.close()
            self.log.close()
            raise
        finally:
            child.close()
        try:
            metadata, arrays = receive(parent)
            self.identity = {k: metadata.get(k) for k in ('backbone_weights_sha256',
                'model_state_sha256', 'text_sha256', 'preprocessing', 'all_frozen')}
            expected = {k: p[k] for k in self.identity if k != 'all_frozen'}
            expected['all_frozen'] = True
            if metadata.get('status') != 'ready' or self.identity != expected:
                raise ValueError('DINO worker identity mismatch: '+str(metadata))
            with np.load(text, allow_pickle=False) as f:
                if set(arrays) != {'text', 'text_mask'} or any(
                        not np.array_equal(arrays[k], f[k]) for k in arrays):
                    raise ValueError('Worker text/mask differs from the original cache')
            self.text, self.text_mask = arrays['text'], arrays['text_mask']
        except Exception:
            self.abort()
            raise

    def close(self):
        report = super().close()
        if (any(report.get(k) != v for k, v in self.identity.items())
                or report.get('weights_unchanged') is not True):
            raise ValueError('DINO worker close identity/frozen invariant differs')
        return report
