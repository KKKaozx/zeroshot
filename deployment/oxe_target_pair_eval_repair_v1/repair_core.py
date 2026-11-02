"""Bounded evaluator fix; never modifies the already trained bundle."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_selection(selection,reference):
    # Use the ORIGINAL published names, no fallback aliases or new defaults.
    assert selection['generation_steps']==reference['config']['model']['num_diffusion_steps']==100
    assert selection['generation_batch_size']==8 and selection['noise_seeds']==[1101,1102,1103]
    assert len(selection['train'])==32 and len(selection['validation'])==88
    rows=selection['train']+selection['validation']
    assert selection['primary_source']=='Bridge' and selection['primary_role']=='validation'
    assert selection['test_or_libero_accessed'] is False
    for role in ['train','validation']:
        for row in selection[role]:
            assert row['role']==role and all(k in row for k in ['source','group_id','episode_key','window_index','input_row','payload_sha256'])
            if role=='train':assert row['prefix_presentations']>0
    return {'windows':len(rows),'generation_steps':100,'generation_batch_size':8,'noise_seeds':selection['noise_seeds']}


def fixed_source(source):
    changes=[
        ('def verify_evaluation(output):','def verify_evaluation(output,evaluation_output=None):'),
        ("output/'evaluation/report.json'","(evaluation_output or output/'evaluation')/'report.json'"),
        ("output/'evaluation/predictions_and_targets.npz'","(evaluation_output or output/'evaluation')/'predictions_and_targets.npz'"),
        ("out=a.output/'evaluation';assert not out.exists();out.mkdir()","out=EVALUATION_OUTPUT;assert not out.exists();out.mkdir()"),
        ("    assert len(rows)==120 and selection['noise_seeds']==[1101,1102,1103] and selection['sampling_steps']==100",
         "    assert validate_selection(selection,reference)['windows']==len(rows)"),
        ('    verify_evaluation(a.output)\n','    verify_evaluation(a.output,out)\n')]
    for before,after in changes:
        assert source.count(before)==1,('Evaluator source differs',before)
        source=source.replace(before,after)
    assert "selection['sampling_steps']" not in source
    compile(source,'<bound-repaired-evaluator>','exec')
    return source


def metadata(base,training):
    contract=read(HERE/'repair_contract.json')
    for name,digest in contract['original_bundle_files'].items():assert sha(base/name)==digest,name
    assert sha(base/'pair_protocol.json')==contract['original_pair_protocol_sha256']
    assert sha(base/'selection.json')==contract['original_selection_sha256']
    selection=validate_selection(read(base/'selection.json'),read(base/'reference_contract.json'))
    assert training.name=='oxe-target-short-231297'
    for target in ['epsilon','sample']:
        assert (training/target/'latest_step.pt').is_file()
        r=read(training/target/'arm_receipt.json')
        assert r['passed'] and r['target']==target and r['prefix_updates']==8192
        assert r['slurm_job_id']=='231297'
    sample=read(training/'sample/arm_receipt.json')
    assert sample['final_learned_sha256']==contract['reported_sample_learned_sha256']
    assert sample['checkpoint_contract_sha256']==contract['reported_sample_contract_sha256']
    assert read(training/'training_pair_verification.json')['passed']
    fixed=fixed_source((base/'evaluate.py').read_text(encoding='utf-8'))
    assert hashlib.sha256(fixed.encode()).hexdigest()==contract['fixed_evaluator_sha256']
    return {'passed':True,'source_training_job':231297,'training_updates':0,'selection':selection,
        'original_bundle_unchanged':True,'scope':'Original metadata/receipts only; full checkpoint verification precedes inference'}


def load_evaluator(base,evaluation_output):
    import sys
    sys.path[:0]=[str(base),str(base/'code')]
    import protocols
    assert protocols.HERE==base.resolve(),'A different training bundle was imported'
    source=fixed_source((base/'evaluate.py').read_text(encoding='utf-8'))
    ns={'__name__':'bound_target_pair_evaluation_repair','__file__':str(base/'evaluate.py'),
        'EVALUATION_OUTPUT':evaluation_output,'validate_selection':validate_selection}
    exec(compile(source,str(base/'evaluate.py'),'exec'),ns)
    return ns,source
