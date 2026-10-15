"""Past-image observation slot only; never construct a future action target."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from dataset import UnifiedRobotDataset, prepare_image


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_previous_images(rows, items, data_root):
    directory = Path(__file__).resolve().parent
    protocol = json.loads((directory / 'protocol.json').read_text())
    audit_file = directory / 'history_inputs.json'
    assert sha(audit_file) == protocol['history_inputs_sha256']
    audit = json.loads(audit_file.read_text())
    assert audit['passed'] and rows == protocol['rows'] and len(items) == len(rows) == 168
    cached = {}; values = []; identities = []
    for i, (row, item, expected) in enumerate(zip(rows, items, audit['rows'])):
        key = (row['shard'], row['record_index']); t = int(row['start_index']); past = max(0, t - 1)
        assert expected['window'] == i and expected['previous_index'] == past
        assert all(expected[k] == row[k] for k in ['shard', 'record_index', 'start_index', 'task'])
        if key not in cached:
            cached[key] = UnifiedRobotDataset._load_tfrecord_example(str(Path(data_root) / key[0]), key[1])
        fields = cached[key].features.feature
        images = fields['steps/observation/image_0'].bytes_list.value
        assert len(images) == expected['episode_steps'] and 0 <= past <= t < len(images)
        for name in ['steps/is_first', 'steps/is_last', 'steps/is_terminal']:
            if name in fields:
                flags = list(fields[name].int64_list.value)
                assert len(flags) == len(images)
                if name == 'steps/is_first': assert not any(flags[1:t + 1])
                if t > 0 and name != 'steps/is_first': assert not flags[past]
        encoded = images[past]; rgb = np.asarray(Image.open(io.BytesIO(encoded)).convert('RGB'))
        assert hashlib.sha256(encoded).hexdigest() == expected['previous_encoded_sha256']
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == expected['previous_rgb_sha256']
        languages = fields['steps/language_instruction'].bytes_list.value
        canonical = lambda s: ' '.join(s.decode('utf-8').lower().split())
        assert canonical(languages[past]) == canonical(languages[t]) == canonical(row['task'].encode())
        value = prepare_image(rgb)
        assert value.shape == item[1].shape == (3, 224, 224) and torch.isfinite(value).all()
        if t == 0:
            assert torch.equal(value, item[1]), 'Boundary history must equal current, within this runtime'
        values.append(value)
        identities.append(dict(window=i, current_index=t, previous_index=past, has_real_history=t > 0,
            encoded_sha256=expected['previous_encoded_sha256'], rgb_sha256=expected['previous_rgb_sha256'],
            normalized_sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest()))
    return torch.stack(values), dict(passed=True, lineage=identities, real_history_train=64,
        real_history_development=30, normalization='Same frozen prepare_image function used by current input',
        cross_runtime_normalized_hash_requires_numeric_review=True,
        future_input_used=False, cross_episode_input_used=False)
