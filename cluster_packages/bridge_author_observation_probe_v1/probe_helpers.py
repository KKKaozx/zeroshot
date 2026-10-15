"""Paired observation construction and native RNG replay; no sampler equations."""
import numpy as np

def observation_batch(torch, arrays, indices, device):
    return {k:torch.from_numpy(np.asarray(arrays[k][indices]).copy()).unsqueeze(1).to(device)
            for k in ('image','ee_state','language')}

def replace_observation(torch, original, arrays, indices, donor_map, case):
    assert case in ('original','image_only','state_only','both')
    output={k:v.clone() for k,v in original.items()}
    selected=[(j,donor_map[int(i)]) for j,i in enumerate(indices) if int(i) in donor_map]
    for j,donor in selected:
        assert np.array_equal(arrays['language'][int(indices[j])], arrays['language'][donor])
        for k in ('image','ee_state'):
            if case=='both' or (case=='image_only' and k=='image') or (case=='state_only' and k=='ee_state'):
                value=torch.from_numpy(np.asarray(arrays[k][donor]).copy()).to(original[k].device)
                output[k][j,0]=value
    # Language, original tensors and all unselected rows must remain intact.
    assert torch.equal(output['language'], original['language'])
    selected_indices={j for j,_ in selected}
    for k in ('image','ee_state'):
        untouched=[j for j in range(len(indices)) if j not in selected_indices]
        if untouched:assert torch.equal(output[k][untouched],original[k][untouched])
        if case=='original' or (case=='image_only' and k=='ee_state') or (case=='state_only' and k=='image'):
            assert torch.equal(output[k],original[k])
    return output

def rng_state(torch, device):
    return (torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone() if device.type=='cuda' else None)

def restore_rng(torch, device, state):
    torch.set_rng_state(state[0])
    if state[1] is not None:torch.cuda.set_rng_state(state[1],device)
    assert torch.equal(torch.get_rng_state(),state[0])
    if state[1] is not None:assert torch.equal(torch.cuda.get_rng_state(device),state[1])

def paired_difference(a,b):
    assert a.shape==b.shape and a.shape[-1]==7 and np.isfinite(a).all() and np.isfinite(b).all()
    xyz=np.linalg.norm((a[...,:3].astype(float)-b[...,:3])*10,axis=-1)
    qa=a[...,3:7].astype(float);qb=b[...,3:7].astype(float)
    assert np.linalg.norm(qa,axis=-1).min()>1e-8 and np.linalg.norm(qb,axis=-1).min()>1e-8
    qa/=np.linalg.norm(qa,axis=-1,keepdims=True);qb/=np.linalg.norm(qb,axis=-1,keepdims=True)
    angle=np.rad2deg(2*np.arccos(np.clip(np.abs(np.sum(qa*qb,axis=-1)),0,1)))
    return dict(position_cm=float(xyz.mean()),rotation_deg=float(angle.mean()),
                per_window_position_cm=xyz.mean((0,2)).tolist(),per_window_rotation_deg=angle.mean((0,2)).tolist(),
                per_draw_position_cm=xyz.mean((1,2)).tolist(),per_draw_rotation_deg=angle.mean((1,2)).tolist())
