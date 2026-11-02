"""Process isolation: project CLIP tokens into the original author's policy."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import numpy as np
import torch
from wire import receive, send, raw_from_author_observations


class CLIPClient:
    def __init__(self, root, out):
        self.request_id = 0
        self.close_report = None
        self.latencies = []
        parent, child = socket.socketpair()
        self.sock = parent
        parent.settimeout(120)
        project = root / "envs/multitask-preflight-v1"
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = str(project / "lib")
        self.log = (out / "clip_worker.log").open("xb")
        self.process = subprocess.Popen([str(project / "bin/python"), "-B",
            str(Path(__file__).with_name("clip_worker.py")), "--fd", str(child.fileno()),
            "--assets", str(root / "baseline_setup/clip-vit-large-patch14-v1")],
            pass_fds=(child.fileno(),), env=environment, stdout=self.log, stderr=subprocess.STDOUT)
        child.close()
        try:
            metadata, arrays = receive(parent)
            if metadata.get("status") != "ready" or not metadata.get("all_frozen"):
                raise RuntimeError("CLIP worker did not initialize: " + str(metadata))
            self.text = arrays["text"]
            self.text_mask = arrays["text_mask"]
        except Exception:
            self.abort()
            raise

    def encode(self, raw):
        self.request_id += 1
        tick = time.monotonic()
        send(self.sock, {"operation": "encode", "request_id": self.request_id}, {"raw_images": raw})
        metadata, arrays = receive(self.sock)
        if metadata.get("request_id") != self.request_id or metadata.get("status") != "ok":
            raise RuntimeError("Invalid worker response: " + str(metadata))
        self.latencies.append(time.monotonic() - tick)
        tokens = arrays["visual"]
        if tokens.shape != (len(raw), 257, 1024) or tokens.dtype != np.float32 or not np.isfinite(tokens).all():
            raise ValueError("Bad live CLIP tokens")
        return tokens

    def close(self):
        self.request_id += 1
        send(self.sock, {"operation": "close", "request_id": self.request_id})
        metadata, _ = receive(self.sock)
        if metadata.get("request_id") != self.request_id or metadata.get("status") != "closed":
            raise RuntimeError("Bad worker close: " + str(metadata))
        self.close_report = metadata
        self.sock.close()
        self.process.wait(timeout=20)
        self.log.close()
        if self.process.returncode != 0:
            raise RuntimeError("CLIP worker failed")
        return metadata

    def abort(self):
        self.sock.close()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.log.close()


class LiveProjectPolicy(torch.nn.Module):
    def __init__(self, core, client):
        super().__init__()
        self.core, self.client = core, client

    @property
    def device(self):
        return self.core.device

    @property
    def dtype(self):
        return self.core.dtype

    def reset(self):
        self.core.obs_encoder.cached_tokens = None
        self.core.reset()

    def predict_action(self, obs):
        raw = raw_from_author_observations(obs["image"][:, :2].detach().cpu().numpy())
        tokens = self.client.encode(raw)
        self.core.obs_encoder.set_tokens(torch.from_numpy(tokens).to(self.core.device))
        try:
            return self.core.predict_action(obs)
        finally:
            self.core.obs_encoder.cached_tokens = None
