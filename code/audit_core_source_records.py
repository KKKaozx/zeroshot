"""Decode the first full-horizon train record per new core source, without a model."""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('CUDA_VISIBLE_DEVICES','-1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import numpy as np
from PIL import Image
import io
from dataset import UnifiedRobotDataset, POSITION_SCALE_METERS, rt1_relative_gripper_commands
from audit_bridge_multitask_pilot import rotation_quaternion

SOURCES={
    'language_table':(Path('E:/dataset/language_table_0.1.0/0.1.0'),'language_table-train.tfrecord-00000-of-01024','tfrecord_language_table_xy'),
    'bc_z':(Path('D:/ntu_related/dissertation/dataset/OpenX/bc_z_1.0.0/1.0.0'),'bc_z-train.tfrecord-00000-of-00512','tfrecord_bc_z_pose'),
    'fractal':(Path('D:/ntu_related/dissertation/dataset/OpenX/fractal20220817_0.1.0/0.1.0'),'fractal20220817_data-train.tfrecord-00000-of-01024','tfrecord_rt1_pose')}


def rotation_vector(v):
    v=np.asarray(v,dtype=np.float64); angle=np.linalg.norm(v)
    if angle<1e-12:return np.eye(3)
    x,y,z=v/angle
    skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


def main():
    report={'purpose':'one_fixed_train_record_per_core_source_contract_probe','model_loaded':False,
        'trained':False,'test_targets_read':False,'mixed_training_ready':False,'sources':{},
        'limitations':['One record does not establish dataset-wide integrity.',
            'Numerical reconstruction cannot establish physical units or controller execution.',
            'Production decoding and masks checked; full source scanning not checked here.']}
    for name,(root,shard,schema) in SOURCES.items():
        # Eligibility is length only, fixed before inspecting values or decoding images.
        image_key='steps/observation/rgb' if name=='language_table' else 'steps/observation/image'
        for record_index in range(32):
            example=UnifiedRobotDataset._load_tfrecord_example(str(root/shard),record_index)
            if len(example.features.feature[image_key].bytes_list.value)>16:
                break
        else: raise ValueError('No full-horizon record among first 32 train records')
        fields=example.features.feature
        def floats(key,dim):return np.asarray(fields[key].float_list.value,np.float32).reshape(-1,dim)
        if name=='language_table':
            position=np.column_stack((floats('steps/observation/effector_translation',2),np.zeros(len(fields['steps/observation/rgb'].bytes_list.value))))
            matrices=np.repeat(np.eye(3)[None],len(position),axis=0)
            image_key='steps/observation/rgb'; gripper=None
        elif name=='bc_z':
            position=floats('steps/observation/present/xyz',3)
            matrices=np.stack([rotation_vector(v) for v in floats('steps/observation/present/axis_angle',3)])
            image_key='steps/observation/image'; gripper=floats('steps/observation/present/sensed_close',1).reshape(-1)
        else:
            pose=floats('steps/observation/base_pose_tool_reached',7)
            position=pose[:,:3]; matrices=np.stack([rotation_quaternion(q) for q in pose[:,3:7]])
            image_key='steps/observation/image'; gripper=floats('steps/action/gripper_closedness_action',1).reshape(-1)
        n=len(position)
        assert n>16 and np.isfinite(position).all() and np.isfinite(matrices).all()
        images=fields[image_key].bytes_list.value
        assert len(images)==n
        sizes=[]
        for encoded in images:
            with Image.open(io.BytesIO(encoded)) as image:
                image.load();sizes.append(list(image.size))
        probe=UnifiedRobotDataset.__new__(UnifiedRobotDataset)
        probe.chunk_size=16;probe.bcz_target='reached'
        probe.rt1_gripper_policy='relative_scan_v2'
        starts=list(range(0,n-16,4))
        probe.samples=[{'source':schema,'file_path':str(root/shard),'record_index':record_index,'start_index':s,'trajectory_steps':n} for s in starts]
        xyz_error=rotation_error=0.
        command_disagreements=command_views=0
        opened=0; legacy_disagreements=0
        rt1_absolute=rt1_relative_gripper_commands(gripper) if name=='fractal' else None
        for i,start in enumerate(starts):
            language,image,current,target,mask=probe[i]
            action=target.numpy(); supervision=mask.numpy()
            assert image.shape==(3,224,224) and np.isfinite(image.numpy()).all() and np.isfinite(action).all()
            expected=np.ones((16,8),np.float32)
            if name=='language_table':expected[:,2:]=0
            np.testing.assert_array_equal(supervision,expected)
            for j in range(16):
                reached=start+j+1
                reconstructed=position[start]+matrices[start]@(action[j,:3]*POSITION_SCALE_METERS)
                xyz_error=max(xyz_error,float(np.abs(reconstructed-position[reached]).max()))
                if name!='language_table':
                    actual=matrices[start]@rotation_quaternion(action[j,3:7])
                    rotation_error=max(rotation_error,float(np.abs(actual-matrices[reached]).max()))
                    expected_gripper=(1 if gripper[reached]<.5 else -1) if name=='bc_z' else rt1_absolute[reached-1]
                    assert action[j,7]==expected_gripper
                    opened+=int(action[j,7]>0)
                    if name=='fractal':
                        legacy_disagreements+=int(action[j,7]!=(1 if gripper[reached-1]<.5 else -1))
                    if name=='bc_z':
                        feature=fields['steps/action/future/target_close']
                        commands=np.asarray(getattr(feature,feature.WhichOneof('kind')).value).reshape(n,10)
                        command_gripper=1 if commands[reached-1,0]<.5 else -1
                        command_disagreements+=int(command_gripper!=action[j,7]);command_views+=1
            if name=='bc_z':assert current.item()==(1 if floats('steps/observation/present/sensed_close',1)[start,0]<.5 else -1)
        assert xyz_error<1e-5 and rotation_error<1e-5
        report['sources'][name]={'shard':shard,'record_index':record_index,'selection_rule':'first record with more than 16 frames, by length only','steps':n,'windows':len(starts),
            'payload_sha256':hashlib.sha256(example.SerializeToString(deterministic=True)).hexdigest(),
            'decoded_frames':n,'image_sizes':sorted({tuple(s) for s in sizes}),
            'position_max_abs_reconstruction_error':xyz_error,'rotation_matrix_max_abs_error':rotation_error,
            'valid_supervision_dimensions':['x','y'] if name=='language_table' else ['x','y','z','qx','qy','qz','qw','gripper'],
            'open_target_views':opened if name!='language_table' else None,
            'production_gripper_label':'placeholder_masked' if name=='language_table' else 'future_measured_state' if name=='bc_z' else 'preceding_relative_command_reconstructed_absolute',
            'rt1_policy':probe.rt1_gripper_policy if name=='fractal' else None,
            'rt1_legacy_disagreement_target_views':legacy_disagreements if name=='fractal' else None,
            'bcz_measured_label_vs_first_command_disagreements':command_disagreements if name=='bc_z' else None,
            'bcz_compared_target_views':command_views if name=='bc_z' else None,
            'physical_units_independently_verified':False,'numeric_production_reconstruction_passed':True}
        print(name,json.dumps(report['sources'][name]),flush=True)
    path=Path(__file__).resolve().parents[1]/'reports/core_source_fixed_record_checks.json'
    path.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
