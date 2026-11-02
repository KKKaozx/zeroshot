"""Bounded CPU data/loss/restore/online tests; no real encoder or simulator."""
import copy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ['USE_TF']='0'

import numpy as np
import torch
torch.set_num_threads(1)
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'code'),str(HERE/'vendor/code'),str(HERE/'vendor')]
from bridge_command8_targets import encode_commands,command_chunk
from command8_runtime import read,sha,load_numeric,replace_command_targets
from command8_metrics import summarize
from online_policy import IncrementExecutor,restore_endpoint
from bridge_controller_command_contract import COMMAND_FORMAT


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.adapter=torch.nn.Linear(7,7)
        class Decoder(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.6))
            def forward(self,x,t,c):return self.weight*x+c[:,None,:]
        self.diffusion_decoder=Decoder();self.gripper_head=torch.nn.Linear(7,1)
        self.vision_encoder=torch.nn.Linear(1,1).requires_grad_(False)
        self.text_encoder=torch.nn.Linear(1,1).requires_grad_(False)
        self.diffusion_prediction_type='sample';self.chunk_size=16;self.action_dim=8
        self.decoder_type='diffusion';self.num_diffusion_steps=100;self.separate_gripper_head=True
        self.gripper_loss_weight=.25;self.gripper_target_mode='state'
        self.gripper_change_weight=1.;self.balanced_gripper_loss=False
        self.register_buffer('alpha_bars',torch.linspace(.999,.001,100))
    def get_context_vector(self,image,ids,attention):return self.adapter(image)
    def predict_gripper_logits(self,context,actions,current):return self.gripper_head(context).expand(-1,16)
    from models import RobotAdapterModel
    diffusion_loss=RobotAdapterModel.diffusion_loss


class Env:
    def __init__(self):
        self.position=np.array([.3,0.,.2]);self.tick=0;self.actions=[]
        self.env=SimpleNamespace(control_freq=20)
        self.robots=[SimpleNamespace(controller=SimpleNamespace(control_dim=6,use_delta=True,
            output_max=np.array([.05]*3+[.5]*3),output_min=-np.array([.05]*3+[.5]*3),input_max=1.,input_min=-1.))]
    def get_sim_state(self):return np.r_[self.tick,self.position]
    def obs(self):return {'robot0_eef_pos':self.position.copy(),'robot0_eef_quat':np.array([0.,0.,0.,1.])}
    def step(self,action):
        self.actions.append(action.copy());self.position+=action[:3]*.01;self.tick+=1
        return self.obs(),0.,False,{}
    def check_success(self):return False


def mapping():
    return {'format':'source-target-pose-mapping-v1','source':'Bridge','target_meters_per_source_unit':1.,
        'source_axes_to_target':np.eye(3).tolist(),'source_origin_in_target_m':[0.,0.,0.],
        'right_tool_alignment':np.eye(3).tolist(),'source_eef_to_point_source_units':[0.,0.,0.],
        'target_eef_to_point_m':[0.,0.,0.],'evidence':'Synthetic coincident-point fixture only'}


class Checks(unittest.TestCase):
    def test_source_phase_clipping_fractional_masks_and_episode_tail(self):
        raw=np.array([[.12,0,0,.6,-.5,.3,.25],[.01,.02,0,0,.1,.2,1.],[0,0,0,0,0,0,0.]])
        values,mask,count=encode_commands(raw,[.05]*3+[.25]*3)
        np.testing.assert_allclose(values[0,:3],[.5,0,0])
        np.testing.assert_allclose(Rotation.from_quat(values[0,3:7]).as_matrix(),Rotation.from_euler('xyz',[.25,-.25,.25]).as_matrix(),atol=1e-7)
        self.assertEqual(mask[0,7],0);self.assertEqual(count['motion_components_clipped'],4)
        target,valid=command_chunk(values,mask,1)
        np.testing.assert_array_equal(target[0],values[1]);self.assertTrue((valid[2:]==0).all())
        np.testing.assert_array_equal(target[2:,6],np.ones(14))

    def test_bound_full_material_and_mean_are_source_only(self):
        a,rows=load_numeric();self.assertEqual(a['targets'].shape,(10416,16,8))
        self.assertEqual(len({r['group_id'] for r in rows[:10352]}),1151)
        self.assertEqual(len({r['group_id'] for r in rows[10352:]}),64)
        receipt=read(HERE/'inputs_receipt.json')
        self.assertFalse(receipt['libero_data_used']);self.assertEqual(receipt['images_decoded'],0)
        self.assertTrue(receipt['mean_fit_duplicate_future_windows_counted_once'])

    def test_new_targets_reach_native_loss_and_full_adam_rng_restore(self):
        import train,bcap_core
        from step_training_state import UpdateSchedule
        a,_=load_numeric();slots=np.array([0,1]);torch.manual_seed(42)
        model=ToyModel();port=torch.nn.Linear(18,7,bias=False)
        old={'images':torch.zeros(2,7),'tokens':{'input_ids':None,'attention_mask':None},
             'pose':torch.from_numpy(a['pose_features'][slots]),'current':torch.from_numpy(a['current'][slots]),
             'actions':torch.full((2,16,8),777.),'masks':torch.ones(2,16,8)}
        x=replace_command_targets(old,a,slots)
        np.testing.assert_array_equal(x['actions'].numpy(),a['targets'][slots]);self.assertEqual(float(old['actions'][0,0,0]),777.)
        from motion_scaling import training_coordinates,MotionStandardizer
        x=training_coordinates(x,read(HERE/'motion_normalizer.json'))
        restored=MotionStandardizer(read(HERE/'motion_normalizer.json')).decode(x['actions'][...,:7])
        torch.testing.assert_close(restored,x['command_actions'][...,:7],atol=1e-7,rtol=0)
        groups=[list(model.adapter.parameters())+list(model.diffusion_decoder.parameters()),list(model.gripper_head.parameters()),list(port.parameters())]
        optimizer=torch.optim.AdamW([{'params':g,'lr':1e-4} for g in groups],weight_decay=0.)
        b={'model':model,'port':port,'optimizer':optimizer,'batch':lambda ids:x,
            'api':SimpleNamespace(policy_loss=train.policy_loss,trainable_state_dict=train.trainable_state_dict),
            'cuda':False,'plan':bcap_core.FixedPlan(np.array([[0,1]]*3,np.int64),'toy_new_commands'),
            'schedule':UpdateSchedule(total_updates=3,base_lrs=[1e-4]*3,kind='constant'),
            'contract':{'test_only':True,'effective_action_representation':COMMAND_FORMAT},
            'checkpoint_every':3,'training_cap_seconds':60,'check_frozen':lambda:None}
        guard=bcap_core.zero_guard(b);self.assertTrue(guard['passed']);self.assertFalse(optimizer.state)
        workspace=HERE/'reports';workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as directory:
            extra=bcap_core.train(b,Path(directory),{'history':[],'zero_update_guard':guard})
            saved=torch.load(Path(directory)/'latest_state.pt',map_location='cpu',weights_only=False)
            self.assertEqual(saved['plan']['completed_updates'],3)
            self.assertTrue(all(int(v['step'])==3 for v in saved['optimizer']['state'].values()))
            self.assertEqual(len(saved['optimizer']['param_groups']),3)
            # Reconstruct the same objects then exercise the real resume path.
            restored=ToyModel();restored_port=torch.nn.Linear(18,7,bias=False)
            restored_groups=[list(restored.adapter.parameters())+list(restored.diffusion_decoder.parameters()),list(restored.gripper_head.parameters()),list(restored_port.parameters())]
            b2=dict(b,model=restored,port=restored_port,optimizer=torch.optim.AdamW([{'params':g,'lr':1e-4} for g in restored_groups],weight_decay=0.),
                plan=bcap_core.FixedPlan(np.array([[0,1]]*3,np.int64),'toy_new_commands'))
            received=bcap_core.resume(b2,Path(directory)/'latest_state.pt')
            self.assertEqual(b2['plan'].completed_updates,3);self.assertEqual(len(received['history']),3)
            self.assertTrue(torch.equal(torch.get_rng_state(),saved['rng']['torch_cpu']))

    def test_new_endpoint_strict_restore_and_old_format_rejection(self):
        import train
        from completed_policy_export import model_sha
        model=ToyModel();port=torch.nn.Linear(18,7,bias=False)
        state={'base':copy.deepcopy(train.trainable_state_dict(model)),'pose_port':copy.deepcopy(port.state_dict())}
        p=read(HERE/'protocol.json')
        payload={'format':'bridge-command8-fresh-scaled-policy-v1','completed_updates':8192,'effective_action_representation':COMMAND_FORMAT,
            'source_learned_checkpoint_loaded':False,'fresh_learned_initialization':True,
            'target_supervised':False,'formal_matrix_admitted':False,'contract':{'protocol_sha256':sha(HERE/'protocol.json'),'protocol':p,'motion_normalizer_sha256':sha(HERE/'motion_normalizer.json'),'motion_normalizer':read(HERE/'motion_normalizer.json')},
            'source_stats':read(HERE/'vendor/state_stats.json'),'state':state,'final_identity':{k:model_sha(v) for k,v in state.items()}}
        other=ToyModel();other_port=torch.nn.Linear(18,7,bias=False);restore_endpoint(other,other_port,payload,p)
        self.assertTrue(torch.equal(other_port.weight,port.weight));self.assertEqual(other.max_normalized_position,.5)
        bad=copy.deepcopy(payload);bad['format']='fresh-bridge-capability-policy-v1'
        with self.assertRaisesRegex(ValueError,'old reached-pose'):restore_endpoint(other,other_port,bad,p)
        bad=copy.deepcopy(payload);del bad['state']['pose_port']
        with self.assertRaisesRegex(ValueError,'complete learned'):restore_endpoint(other,other_port,bad,p)

    def test_affine_loss_coordinates_preserve_masks_grip_and_external_roundtrip(self):
        from motion_scaling import MotionStandardizer,training_coordinates
        a,_=load_numeric();norm=MotionStandardizer(read(HERE/'motion_normalizer.json'))
        x={'actions':torch.from_numpy(a['targets'][:2]),'masks':torch.from_numpy(a['masks'][:2])}
        before=x['actions'].clone();y=training_coordinates(x,norm)
        self.assertIs(y['masks'],x['masks']);torch.testing.assert_close(y['actions'][...,7],before[...,7])
        torch.testing.assert_close(norm.decode(y['actions'][...,:7]),before[...,:7],atol=1e-7,rtol=0)
        torch.testing.assert_close(x['actions'],before)
        self.assertEqual(norm.spec['fit_unique_command_rows'],39572)
        self.assertFalse(norm.spec['development_data_fit']);self.assertFalse(norm.spec['libero_data_used'])

    def test_sampling_clips_in_physical_not_standardized_coordinates(self):
        from motion_scaling import MotionStandardizer
        norm=MotionStandardizer(read(HERE/'motion_normalizer.json'))
        z=torch.tensor([[-1000.]*7,[1000.]*7])
        clipped=norm.decode(norm.bound_clean(z))
        low=torch.tensor(norm.low,dtype=torch.float32);high=torch.tensor(norm.high,dtype=torch.float32)
        self.assertTrue((clipped>=low-1e-6).all());self.assertTrue((clipped<=high+1e-6).all())
        self.assertGreater(float(norm.encode(torch.tensor(norm.high,dtype=torch.float32))[0]),1.)

    def test_full_standardized_ddpm_oracle_returns_physical8_and_keeps_state_rng_on_install(self):
        from motion_scaling import MotionStandardizer,install_standardized_sampler
        norm=MotionStandardizer(read(HERE/'motion_normalizer.json'));model=ToyModel()
        target=torch.tensor([.1,-.2,.05,0.,0.,0.,1.]).expand(1,16,7).clone()
        class Oracle(torch.nn.Module):
            def forward(self,x,t,c):return norm.encode(target).expand(len(x),-1,-1)
        model.diffusion_decoder=Oracle()
        schedule=read(HERE/'vendor/profile_sample.json')['schedule_values']
        for name in ('posterior_mean_x0','posterior_mean_xt','posterior_variance'):
            self.assertEqual(len(schedule[name]),100)
            model.register_buffer(name,torch.tensor(schedule[name]))
        rng=torch.get_rng_state().clone();state=copy.deepcopy(model.state_dict())
        install_standardized_sampler(model,norm.spec)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng));self.assertEqual(set(state),set(model.state_dict()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in state.items()))
        result=model.sample(torch.zeros(1,7),torch.zeros(1,2))
        # Original archived float32 DDPM a0=1.000013947..., b0=0, not
        # mathematically exact1/0. Preserve those constants and check their
        # analytic output independently; do not change the sampler to pass.
        self.assertEqual(schedule['posterior_mean_xt'][0],0.)
        physical=target.numpy().astype(np.float64)
        expected=norm.mean+schedule['posterior_mean_x0'][0]*(physical-norm.mean)
        expected[...,3:7]/=np.linalg.norm(expected[...,3:7],axis=-1,keepdims=True)
        self.assertEqual(result.shape,(1,16,8))
        torch.testing.assert_close(result[...,:7],torch.from_numpy(expected).float(),atol=1e-7,rtol=0)
        self.assertLess(float((result[...,:7]-target).abs().max()),5e-6)
        self.assertTrue(torch.isin(result[...,7],torch.tensor([-1.,1.])).all())

    def test_loader_rejects_old_unscaled_endpoint_and_wrong_normalizer(self):
        import train
        from completed_policy_export import model_sha
        model=ToyModel();port=torch.nn.Linear(18,7,bias=False);p=read(HERE/'protocol.json')
        state={'base':copy.deepcopy(train.trainable_state_dict(model)),'pose_port':copy.deepcopy(port.state_dict())}
        payload={'format':'bridge-command8-fresh-scaled-policy-v1','completed_updates':8192,
            'effective_action_representation':COMMAND_FORMAT,'source_learned_checkpoint_loaded':False,'fresh_learned_initialization':True,
            'contract':{'protocol_sha256':sha(HERE/'protocol.json'),'protocol':p,
                'motion_normalizer_sha256':'wrong','motion_normalizer':read(HERE/'motion_normalizer.json')},
            'source_stats':read(HERE/'vendor/state_stats.json'),'state':state,'final_identity':{k:model_sha(v) for k,v in state.items()}}
        with self.assertRaisesRegex(ValueError,'normalizer'):restore_endpoint(model,port,payload,p)
        payload['format']='bridge-command8-source-policy-v1'
        with self.assertRaisesRegex(ValueError,'old reached-pose'):restore_endpoint(model,port,payload,p)

    def test_online_goal_persists_and_old_pose_callback_rejected(self):
        class Policy:
            action_representation=COMMAND_FORMAT
            def __call__(self,*args):return np.tile([.1,0,0,0,0,0,1,1],(16,1))
        env=Env();executor=IncrementExecutor(mapping(),read(HERE/'protocol.json'))
        a=executor.execute(env,Policy(),env.obs(),'fixture',None,native_step_budget=8)
        b=executor.execute(env,Policy(),a['observation'],'fixture',a['current_open_command'],native_step_budget=4)
        np.testing.assert_allclose([v['desired_position'][0] for v in a['trace']],[.31]*4,atol=1e-12)
        np.testing.assert_allclose([v['desired_position'][0] for v in b['trace']],[.32]*4,atol=1e-12)
        self.assertTrue(a['goal_initialized_from_measured_once']);self.assertFalse(b['goal_initialized_from_measured_once'])
        old=Policy();old.action_representation='source_cartesian_delta_pose_open_positive_v1'
        with self.assertRaisesRegex(ValueError,'Reached-pose'):executor.execute(env,old,env.obs(),'fixture',1.,native_step_budget=4)
        self.assertEqual(len(env.actions),8)

    def test_saved_metric_sense_first_command_gate_and_masks(self):
        values,mask,_=encode_commands(np.array([[.01,0,0,.1,0,0,1.],[0,.02,0,0,.12,0,0.]]),[.05]*3+[.25]*3)
        target,valid=command_chunk(values,mask,0)
        target=np.repeat(target[None],4,axis=0);valid=np.repeat(valid[None],4,axis=0)
        pred=np.repeat(target[None],3,axis=0);pred[:,:,2:,:3]=999 # padded slots never scored
        mean=np.array([0,0,0,0,0,0,1,1],np.float32)
        r=summarize(pred,target,valid,mean)
        self.assertTrue(r['source_engineering_review_gate_passed'])
        self.assertAlmostEqual(r['metrics']['model']['first_position_normalized'],0)

    def test_source_reader_pairs_current_image_text_and_command_without_old_target_cache(self):
        import hashlib,io
        import tensorflow as tf
        tf.config.set_visible_devices([],'GPU')
        from PIL import Image
        from dataset import prepare_image
        from command8_inputs import SourceInputs
        colors=[(255,0,0),(0,255,0),(0,0,255)];encoded=[]
        for color in colors:
            buffer=io.BytesIO();Image.new('RGB',(4,4),color).save(buffer,format='PNG');encoded.append(buffer.getvalue())
        example=tf.train.Example(features=tf.train.Features(feature={
            'steps/observation/image_0':tf.train.Feature(bytes_list=tf.train.BytesList(value=encoded)),
            'steps/language_instruction':tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'row0',b'row1',b'row2']))}))
        raw=example.SerializeToString();values,mask,_=encode_commands(np.array([[.01,0,0,0,0,0,1],[.02,0,0,0,0,0,1]]),[.05]*3+[.25]*3)
        pairs=[command_chunk(values,mask,i) for i in (0,1)]
        a={'targets':np.stack([v[0] for v in pairs]),'masks':np.stack([v[1] for v in pairs]),
           'current':np.zeros((2,2),np.float32),'pose_features':np.zeros((2,18),np.float32)}
        class Tokenizer:
            def __call__(self,texts,**kwargs):
                self.texts=texts;return {'input_ids':torch.ones(len(texts),77,dtype=torch.int64),'attention_mask':torch.ones(len(texts),77,dtype=torch.int64)}
        workspace=HERE/'reports';workspace.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=workspace) as folder:
            root=Path(folder);file=root/'data/oxe_nominal_pool_v1/bridge_v2/fixture';file.parent.mkdir(parents=True);file.write_bytes(raw)
            rows=[{'role':'train','window_index':10+i,'input_row':i,'stage_relative_path':'bridge_v2/fixture',
                   'steps':3,'payload_offset':0,'payload_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest()} for i in (0,1)]
            reader=SourceInputs(root,a,rows);tokenizer=Tokenizer();x=reader.pack('train',[11],tokenizer,device='cpu')
            self.assertEqual(tokenizer.texts,['row1'])
            torch.testing.assert_close(x['images'][0],prepare_image(np.array(Image.new('RGB',(4,4),colors[1]))))
            self.assertAlmostEqual(float(x['actions'][0,0,0]),.2,places=6)


if __name__=='__main__':unittest.main(verbosity=2)
