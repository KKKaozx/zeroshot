"""Constructor-only restoration of two identity-bound training constants."""
from contextlib import contextmanager


@contextmanager
def training_constant_constructor(model_class, *, contract, profile_path,
                                  learned_state, on_restored):
    from frozen_profile_restore import normalized_config, restore_training_constants, metadata
    import json
    import torch
    original = model_class.__init__
    profile = json.loads(profile_path.read_text())
    events = []

    def initialize(self, config, *args, **kwargs):
        if normalized_config(config) != normalized_config(contract['config']):
            raise ValueError('Actual constructor configuration differs from saved stage1')
        original(self, config, *args, **kwargs)
        learned = learned_state(self)
        actual = {k: metadata(v) for k,v in self.state_dict().items() if k not in learned}
        changed = {k for k in set(actual) | set(profile['tensors'])
                   if actual.get(k) != profile['tensors'].get(k)}
        if not changed <= {'posterior_mean_x0','posterior_mean_xt'}:
            raise ValueError('Resume repair cannot change an encoder or another buffer: '+str(sorted(changed)))
        before = torch.get_rng_state().clone()
        event = restore_training_constants(self, learned, contract, profile_path)
        if not torch.equal(before, torch.get_rng_state()):
            raise RuntimeError('Constant restoration unexpectedly changed CPU RNG')
        events.append(event)
        on_restored(event)

    model_class.__init__ = initialize
    try:
        yield events
    finally:
        model_class.__init__ = original
