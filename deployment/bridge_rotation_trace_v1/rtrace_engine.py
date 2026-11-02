"""Read-only instrumentation of the frozen native sample method; no replacement formula."""
import ast
import copy
import hashlib
import time
import numpy as np
import torch


def instrument(path):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RobotAdapterModel')
    fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='sample'))
    loop=next(n for n in fn.body if isinstance(n,ast.For))
    def assigned(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
    out=[];raw_count=0;clean_count=0
    for node in loop.body:
        out.append(node)
        if assigned(node,'predicted_noise'):
            out.append(ast.parse('RTRACE_HOOK.raw(step, actions, predicted_noise)').body[0]);raw_count+=1
        if isinstance(node,ast.If) and ast.unparse(node.test)=='self.clip_denoised':
            nested=[]
            for inner in node.body:
                if assigned(inner,'clean') and isinstance(inner.value,ast.Call) and ast.unparse(inner.value.func)=='torch.cat':
                    nested.append(ast.parse('RTRACE_HOOK.reconstructed(step, clean)').body[0]);clean_count+=1
                nested.append(inner)
                if assigned(inner,'clean') and isinstance(inner.value,ast.Call) and ast.unparse(inner.value.func)=='torch.cat':
                    nested.append(ast.parse('RTRACE_HOOK.clipped(step, clean)').body[0])
            node.body=nested
    assert raw_count==clean_count==1
    loop.body=out;body=[];final_count=0
    for node in fn.body:
        if assigned(node,'quaternion') and isinstance(node.value,ast.Subscript):
            body.append(ast.parse('RTRACE_HOOK.final_raw(actions)').body[0]);final_count+=1
        body.append(node)
    assert final_count==1;fn.body=body
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])))+'\n'


class Recorder:
    def __init__(self,cap=300):
        self.started=time.monotonic();self.cap=cap;self.data={};self.current=None
    def begin(self,seed,role):
        self.current=(seed,role);assert self.current not in self.data
        self.data[self.current]={'raw':[],'latent':[],'reconstructed':[],'clipped':[],'steps':[]}
    def raw(self,step,latent,clean):
        assert time.monotonic()-self.started<self.cap,'Trace cap reached; no automatic retry'
        r=self.data[self.current];assert step==99-len(r['steps'])
        assert latent.shape==clean.shape and clean.shape[1:]==(16,7)
        r['steps'].append(step);r['latent'].append(latent[...,3:7].detach().cpu().numpy().copy())
        r['raw'].append(clean[...,3:7].detach().cpu().numpy().copy())
    def reconstructed(self,step,clean):
        assert self.data[self.current]['steps'][-1]==step
        self.data[self.current]['reconstructed'].append(clean[...,3:7].detach().cpu().numpy().copy())
    def clipped(self,step,clean):
        assert self.data[self.current]['steps'][-1]==step
        self.data[self.current]['clipped'].append(clean[...,3:7].detach().cpu().numpy().copy())
    def final_raw(self,actions):self.data[self.current]['final_raw']=actions.detach().cpu().numpy().copy()
    def arrays(self):
        result={}
        for field in ['raw','latent','reconstructed','clipped']:
            by_seed=[]
            for seed in [1101,1102,1103]:
                # [100,B,H,4] -> [B,100,H,4], then concatenate fixed train/dev rows.
                by_seed.append(np.concatenate([np.stack(self.data[(seed,role)][field],axis=1) for role in ['train','validation']],axis=0))
            result[field]=np.stack(by_seed)
            assert result[field].shape==(3,16,100,16,4)
        result['final_raw']=np.stack([np.concatenate([self.data[(s,r)]['final_raw'] for r in ['train','validation']]) for s in [1101,1102,1103]])
        return result


def advance_sample_rng(device,batches=1):
    """Mirror only the RNG calls of each skipped native B8/100-step sample."""
    for _ in range(batches):
        actions=torch.randn(8,16,7,device=device)
        for step in reversed(range(100)):
            if step>0:torch.randn_like(actions)


def analyze(arrays,targets,masks,final,reference):
    assert np.array_equal(final,reference),'Existing predictions did not replay exactly'
    qtarget=targets[...,3:7].astype(np.float64);valid=masks[...,3:7].all(-1)
    def stats(q,ids):
        v=q[:,ids].astype(np.float64);norm=np.linalg.norm(v,axis=-1)
        mask=np.broadcast_to(valid[ids],norm.shape);n=norm[mask]
        assert np.isfinite(v).all()
        normalized=v/np.maximum(norm[...,None],1e-16)
        truth=qtarget[ids];truth=truth/np.linalg.norm(truth,axis=-1,keepdims=True)
        dot=(normalized*truth).sum(-1)
        angle=np.degrees(2*np.arccos(np.clip(abs(dot),0,1)))
        angular_mask=np.broadcast_to(valid[ids],norm.shape)&(norm>1e-6)
        counts=angular_mask.sum(-1);eligible=counts>0
        per_window=(angle*angular_mask).sum(-1)/np.maximum(counts,1)
        error=((v-qtarget[ids])**2).sum(-1)/4
        truthnorm=np.linalg.norm(qtarget[ids],axis=-1)
        radial=(norm-truthnorm)**2/4
        directional=2*norm*truthnorm*(1-np.clip(dot,-1,1))/4
        assert np.allclose(error,radial+directional,atol=1e-10,rtol=1e-12)
        unit=np.array([0.,0.,0.,1.]);static=((unit-qtarget[ids])**2).sum(-1)/4
        mean=lambda values:float(((values*valid[ids]).sum(-1)/valid[ids].sum(-1)).mean())
        return {'norm_quantiles':np.quantile(n,[0,.01,.5,.99,1]).tolist(),'norm_below0_1_fraction':float((n<.1).mean()),
            'angle_deg_window_equal_nonzero_only':float(per_window[eligible].mean()) if eligible.any() else None,
            'quaternion_component_mse_window_equal':mean(error),
            'quaternion_radial_mse_window_equal':mean(radial),'quaternion_directional_mse_window_equal':mean(directional),
            'identity_component_mse_window_equal':mean(static),
            'negative_dot_fraction':float((dot[mask]<0).mean()),'invalid_norm_count':int((n<=1e-6).sum())}
    result={}
    for role,ids in [('train',list(range(8))),('validation',list(range(8,16)))]:
        rows=[]
        for index in range(100):
            raw=arrays['raw'][:,:,index];recon=arrays['reconstructed'][:,:,index];clip=arrays['clipped'][:,:,index]
            selectedmask=np.broadcast_to(valid[ids][None,...,None],recon[:,ids].shape)
            rows.append({'t':99-index,'raw_x0':stats(raw,ids),'clipped_clean':stats(clip,ids),
                'raw_to_reconstructed_max_abs_all_frames':float(abs(raw[:,ids]-recon[:,ids]).max()),
                'raw_to_reconstructed_max_abs_valid_frames':float(abs(raw[:,ids]-recon[:,ids])[selectedmask].max()),
                'clipped_component_fraction_valid_frames':float((abs(recon[:,ids])[selectedmask]>1).mean())})
        rawfinal=arrays['final_raw'][:,ids,:,3:7].astype(np.float64)
        norm=np.linalg.norm(rawfinal,axis=-1);unit=rawfinal/np.maximum(norm[...,None],1e-16)
        # For nonzero quaternions normalization preserves represented rotation.
        nonzero=norm>1e-6
        result[role]={'windows':8,'all100_steps':rows,'final_before_normalization':stats(arrays['final_raw'][...,3:7],ids),
            'final_identity_fallback_count_all_frames':int((norm<=1e-6).sum()),
            'unit_normalization_max_component_difference_nonzero':float(abs(unit-final[:,ids,:,3:7])[nonzero].max()) if nonzero.any() else None,
            'normalization_angle_comparison_note':'Small nonzero comparison angle can be float32 rounding; exact represented rotation is invariant for nonzero q'}
    return {'roles':result,'existing16window_predictions_replayed_exactly':True,
        'scope':'Native on-policy100-step trace at234100 endpoint,16fixed reused windows,3saved sampling seeds. No teacher-forced five-t probe or new scoring policy.',
        'optimizer_updates':0,'automatic_following_action':False}
