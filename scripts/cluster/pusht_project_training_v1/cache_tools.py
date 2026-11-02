"""Numeric frame cache mechanics, no model or task evaluation."""
import hashlib
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode_frames(images, destination, client, progress=None):
    if destination.shape[0] != len(images):
        raise ValueError("Frame count mismatch")
    written = 0
    for start in range(0, len(images), 32):
        stop = min(start + 32, len(images))
        visual = client.encode(images[start:stop])
        expected = (stop - start,) + destination.shape[1:]
        if visual.shape != expected or visual.dtype != np.float32 or not np.isfinite(visual).all():
            raise ValueError("Invalid frozen CLIP output")
        destination[start:stop] = visual
        written = stop
        if progress and (stop % 512 == 0 or stop == len(images)):
            progress(stop, len(images))
    return written


def compare_reference(cache, frame_ids, reference_ids, expected, atol=1e-4, rtol=1e-5):
    if frame_ids.ndim != 1 or len(frame_ids) != len(cache) or (np.diff(frame_ids) <= 0).any():
        raise ValueError("Frame IDs are not a strictly increasing cache index")
    reference_ids = np.asarray(reference_ids, dtype=np.int64)
    indices = np.searchsorted(frame_ids, reference_ids)
    if (indices >= len(frame_ids)).any() or not np.array_equal(frame_ids[indices], reference_ids):
        raise ValueError("Old reference frame is missing from the training cache")
    maximum = 0.
    for start in range(0, len(reference_ids), 8):
        stop = min(start + 8, len(reference_ids))
        actual = np.asarray(cache[indices[start:stop]])
        maximum = max(maximum, float(np.max(np.abs(actual - expected[start:stop]))))
        if not np.allclose(actual, expected[start:stop], atol=atol, rtol=rtol):
            raise ValueError("Cached CLIP features differ from the established128-frame reference")
    return {"frames": len(reference_ids), "max_absolute_difference": maximum,
            "atol": atol, "rtol": rtol, "passed": True}
