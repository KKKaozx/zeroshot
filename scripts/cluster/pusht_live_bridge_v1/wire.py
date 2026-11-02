"""Private inherited socket, JSON plus numeric NPZ. No pickle or listening port."""
import io
import json
import struct
import numpy as np

MAX_BYTES = 64 * 1024 ** 2


def read_exact(sock, count):
    blocks = []
    while count:
        block = sock.recv(min(count, 1024 * 1024))
        if not block:
            raise EOFError("CLIP worker connection closed")
        blocks.append(block)
        count -= len(block)
    return b"".join(blocks)


def send(sock, metadata, arrays=None):
    header = json.dumps(metadata).encode()
    buffer = io.BytesIO()
    if arrays is not None:
        if any(np.asarray(value).dtype.hasobject for value in arrays.values()):
            raise ValueError("Object arrays are forbidden")
        np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    if len(header) > 65536 or len(payload) > MAX_BYTES:
        raise ValueError("IPC packet too large")
    sock.sendall(struct.pack("!IQ", len(header), len(payload)) + header + payload)


def receive(sock):
    header_size, body_size = struct.unpack("!IQ", read_exact(sock, 12))
    if header_size > 65536 or body_size > MAX_BYTES:
        raise ValueError("IPC packet too large")
    metadata = json.loads(read_exact(sock, header_size))
    arrays = {}
    if body_size:
        with np.load(io.BytesIO(read_exact(sock, body_size)), allow_pickle=False) as file:
            arrays = {name: file[name] for name in file.files}
    return metadata, arrays


def raw_from_author_observations(image):
    """Invert the author's uint8->float32/255 observation, not arbitrary floats."""
    image = np.asarray(image)
    if image.dtype != np.float32 or image.ndim != 5 or image.shape[1:] != (2, 3, 96, 96):
        raise ValueError("Expected float32 [B,2,3,96,96] author observation")
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Expected raw author observation [0,1], before policy normalization")
    raw = image.transpose(0, 1, 3, 4, 2).reshape(-1, 96, 96, 3) * np.float32(255)
    rounded = np.rint(raw)
    if np.max(np.abs(raw - rounded)) > 1e-4:
        raise ValueError("Input is not the expected rendered 8-bit pixel grid")
    return rounded  # float32; remove only /255 round-trip rounding error.
