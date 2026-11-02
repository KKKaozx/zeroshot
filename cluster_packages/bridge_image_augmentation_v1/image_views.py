"""One fixed training-only image shift; evaluation uses the original cached view."""
import hashlib
import json
import time


def shifted_images(images, seed=None, offsets=None):
    import torch
    assert images.ndim == 4 and tuple(images.shape[1:]) == (3, 224, 224)
    if offsets is None:
        assert seed is not None
        generator = torch.Generator(device='cpu').manual_seed(seed)
        offsets = torch.randint(17, (len(images), 2), generator=generator)
    assert tuple(offsets.shape) == (len(images), 2)
    assert offsets.dtype == torch.long and int(offsets.min()) >= 0 and int(offsets.max()) <= 16
    padded = torch.nn.functional.pad(images, (8, 8, 8, 8), mode='replicate')
    result = torch.stack([padded[i, :, y:y+224, x:x+224] for i, (y, x) in enumerate(offsets.tolist())])
    return result, offsets


class OnlineViews:
    def __init__(self, model, items, original_tokens, output, report):
        import torch
        self.model, self.items, self.output, self.report = model, items, output, report
        self.calls, self.seconds, self.hash = 0, 0., hashlib.sha256()
        # The original cache is still the sole evaluation path. Check that the new
        # online encoder agrees on four *training* images before any optimizer step.
        model.vision_encoder.cuda().eval()
        indices = [0, 32, 64, 96]
        images = torch.stack([items[i][1] for i in indices])
        centered, _ = shifted_images(images, offsets=torch.full((4, 2), 8, dtype=torch.long))
        assert torch.equal(centered, images)
        encoded = self.encode(centered)
        delta = float((encoded-original_tokens[indices]).abs().max())
        assert torch.allclose(encoded, original_tokens[indices], atol=1e-4, rtol=1e-5), delta
        report.update(center_view_input_exact=True, center_view_token_max_absolute_difference=delta,
                      evaluation_uses_original_tokens=True, augmentation_train_rows_only=True)

    def encode(self, images):
        import torch
        assert not self.model.vision_encoder.training
        with torch.no_grad():
            return torch.cat([self.model.vision_encoder(pixel_values=images[i:i+2].cuda()).last_hidden_state
                              for i in range(0, len(images), 2)]).detach()

    def __call__(self, indices, augmentation_seed, arm):
        import torch
        started = time.monotonic()
        rows = indices.detach().cpu().tolist()
        assert all(0 <= i < 128 for i in rows), 'Held-out images must never be augmented for optimization'
        images = torch.stack([self.items[i][1] for i in rows])
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        augmented, offsets = shifted_images(images, seed=augmentation_seed)
        tokens = self.encode(augmented)
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(gpu_rng, torch.cuda.get_rng_state())
        assert not tokens.requires_grad
        entry = json.dumps(dict(arm=arm, seed=augmentation_seed, rows=rows, offsets=offsets.tolist()), sort_keys=True)+'\n'
        with (self.output/'augmentation_views.jsonl').open('a', encoding='utf-8') as log:
            log.write(entry)
        self.hash.update(entry.encode()); self.calls += 1
        self.seconds += time.monotonic()-started
        self.report.update(augmentation_calls=self.calls, augmentation_trace_sha256=self.hash.hexdigest(),
                           online_encoding_seconds=self.seconds, global_rng_preserved_by_augmentation=True)
        return tokens
