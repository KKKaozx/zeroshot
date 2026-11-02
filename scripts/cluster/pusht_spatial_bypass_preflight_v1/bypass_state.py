"""Mode-bound complete snapshots, using the unchanged original state machinery."""
import training_core as core
from spatial_bypass import MODES


def capture(policy, averaged, optimizer, scheduler, ema, completed, base_sha, plan_sha, protocol_sha, mode):
    if mode not in MODES or policy.obs_encoder.mode != mode or averaged.obs_encoder.mode != mode:
        raise ValueError("Bypass mode mismatch")
    state = core.capture(policy, averaged, optimizer, scheduler, ema, completed, base_sha, plan_sha)
    state.update(format="pusht-spatial-bypass-complete-v1", bypass_protocol_sha256=protocol_sha, bypass_mode=mode)
    return state


def restore(payload, policy, averaged, optimizer, scheduler, ema, base_sha, plan_sha, protocol_sha, mode):
    if (payload.get("format") != "pusht-spatial-bypass-complete-v1" or mode not in MODES
            or payload.get("bypass_protocol_sha256") != protocol_sha or payload.get("bypass_mode") != mode
            or policy.obs_encoder.mode != mode or averaged.obs_encoder.mode != mode):
        raise ValueError("Changed or missing bypass checkpoint binding")
    original = dict(payload, format="pusht-native-complete-v1")
    return core.restore(original, policy, averaged, optimizer, scheduler, ema, base_sha, plan_sha)
