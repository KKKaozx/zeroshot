"""Check augmentation geometry and RNG isolation independently of a neural model."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'cluster_packages/bridge_image_augmentation_v1'))
from image_views import shifted_images

torch.set_num_threads(4)
image = torch.arange(3*224*224, dtype=torch.float32).reshape(1,3,224,224)/150528
offsets = torch.tensor([[0,0],[0,16],[8,8],[16,0],[16,16]])
images = image.expand(5,-1,-1,-1).clone()
before = images.clone()
actual, _ = shifted_images(images, offsets=offsets)
for i, (y, x) in enumerate(offsets.tolist()):
    yy = (torch.arange(224)+y-8).clamp(0,223)
    xx = (torch.arange(224)+x-8).clamp(0,223)
    expected = images[i][:,yy[:,None],xx[None,:]]
    assert torch.equal(actual[i], expected)
assert torch.equal(actual[2], images[2]) and torch.equal(images, before)
torch.manual_seed(712)
rng = torch.get_rng_state().clone()
a, indices = shifted_images(images, seed=910)
assert torch.equal(rng, torch.get_rng_state())
b, replay = shifted_images(images, seed=910)
assert torch.equal(a,b) and torch.equal(indices,replay)
_, changed = shifted_images(images, seed=911)
assert not torch.equal(indices,changed)
mean = torch.tensor([0.48145466,0.4578275,0.40821073]).view(1,3,1,1)
std = torch.tensor([0.26862954,0.26130258,0.27577711]).view(1,3,1,1)
shift_raw, _ = shifted_images(images, offsets=offsets)
shift_normalized, _ = shifted_images((images-mean)/std, offsets=offsets)
assert torch.equal((shift_raw-mean)/std,shift_normalized)
output = ROOT/'reports/bridge_image_augmentation_helper_check_20261016.json'
output.write_text(json.dumps(dict(passed=True, manual_border_and_center_geometry_exact=True,
    input_unchanged=True, normalization_commutes_exactly=True, local_generator_replays=True,
    global_rng_unchanged=True, no_neural_execution=True),indent=2)+'\n',encoding='utf-8')
print('AUGMENTATION INPUT/RNG CONTRACT: PASSED',output)
