"""CPU transport/pixel-contract/wrapper tests; no CLIP, simulator or GPU."""
import ast
import json
from pathlib import Path
import struct
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "scripts/cluster/pusht_live_bridge_v1"
sys.path.insert(0, str(BUNDLE))


def reject(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def main():
    for path in BUNDLE.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 9))
    from wire import receive, send, raw_from_author_observations, MAX_BYTES
    from live_client import LiveProjectPolicy
    pixels = np.resize(np.arange(256, dtype=np.float32), (2, 2, 3, 96, 96))
    obs_image = pixels / np.float32(255)
    reconstructed = raw_from_author_observations(obs_image)
    expected = pixels.transpose(0, 1, 3, 4, 2).reshape(4, 96, 96, 3)
    assert np.array_equal(reconstructed, expected)
    reject(lambda: raw_from_author_observations(obs_image / 255))
    reject(lambda: raw_from_author_observations(obs_image * 2 - 1))
    reject(lambda: raw_from_author_observations(obs_image.astype(np.float64)))
    # Fragmented in-memory socket surface avoids Windows loopback policy;
    # actual Linux inherited socket/process behavior is deferred to cluster job.
    class MemorySocket:
        def __init__(self):
            self.incoming = bytearray()

        def sendall(self, data):
            self.peer.incoming.extend(data)

        def recv(self, count):
            count = min(count, 16384, len(self.incoming))
            value = bytes(self.incoming[:count])
            del self.incoming[:count]
            return value

    left, right = MemorySocket(), MemorySocket()
    left.peer, right.peer = right, left
    send(left, {"request_id": 1, "operation": "echo"}, {"pixels": reconstructed})
    metadata, arrays = receive(right)
    assert metadata == {"request_id": 1, "operation": "echo"}
    assert np.array_equal(arrays["pixels"], reconstructed)
    send(right, {"request_id": 1, "status": "ok"}, arrays)
    metadata, arrays = receive(left)
    assert metadata["request_id"] == 1 and np.array_equal(arrays["pixels"], expected)
    reject(lambda: send(left, {}, {"object": np.array([{}], dtype=object)}))
    left.sendall(struct.pack("!IQ", 0, MAX_BYTES + 1))
    reject(lambda: receive(right))

    class Encoder:
        cached_tokens = None

        def set_tokens(self, value):
            self.cached_tokens = value

    class Core(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.obs_encoder = Encoder()
            self.fail = False
            self.device, self.dtype = torch.device("cpu"), torch.float32

        def reset(self):
            pass

        def predict_action(self, obs):
            assert self.obs_encoder.cached_tokens.shape == (4, 257, 1024)
            assert obs["agent_pos"].dtype == torch.float64
            if self.fail:
                raise ValueError("Intentional core failure")
            return {"action": torch.zeros(2, 8, 2)}

    class Client:
        def encode(self, raw):
            assert np.array_equal(raw, expected)
            return np.zeros((len(raw), 257, 1024), dtype=np.float32)

    core = Core()
    policy = LiveProjectPolicy(core, Client())
    obs = {"image": torch.from_numpy(obs_image), "agent_pos": torch.zeros(2, 2, 2, dtype=torch.float64)}
    assert policy.predict_action(obs)["action"].shape == (2, 8, 2)
    assert core.obs_encoder.cached_tokens is None
    core.fail = True
    reject(lambda: policy.predict_action(obs))
    assert core.obs_encoder.cached_tokens is None
    result = {"python39_syntax": True, "all_256_rendered_pixel_values_roundtrip_exact": True,
              "double_normalization_and_wrong_dtype_rejected": True, "fragmented_memory_socket_npz_roundtrip": True,
              "object_and_oversize_packets_rejected": True, "wrapper_clears_cache_on_success_and_failure": True,
              "raw_state_preserved_until_author_normalizer": True,
              "cross_python_gpu_processes_tested": False, "clip_simulation_training_executed": False}
    (ROOT / "reports/pusht_live_bridge_cpu_checks_20261018.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
