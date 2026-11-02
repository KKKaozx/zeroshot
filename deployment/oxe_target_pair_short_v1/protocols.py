"""Explicit diagnostic admission; original full-training protocol stays intact."""
import copy
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_diagnostic(args,dataset,original_loader):
    pair=read(HERE/'pair_protocol.json');arm=read(args.oxe_formal_protocol)
    assert Path(args.oxe_formal_protocol).resolve()==HERE/f"protocol_{args.diffusion_prediction_type}.json"
    assert arm['format']=='explicit-main-target-prefix-protocol-v1'
    assert arm['target']==args.diffusion_prediction_type in pair['arms']
    assert arm['pair_protocol_sha256']==sha(HERE/'pair_protocol.json')
    assert args.total_updates==65536 and args.stop_after_updates==8192 and args.step_resume is None
    assert args.checkpoint_every_updates==8192 and args.log_interval==256
    reference=read(HERE/'reference_contract.json')
    expected=dict(reference['arguments'],diffusion_prediction_type=arm['target'])
    for key,value in expected.items():assert getattr(args,key)==value,('Diagnostic fixed argument differs',key)
    assert sha(HERE/'source_protocol.json')==pair['original_source_protocol_sha256']
    # Validate the original DATA/source qualification under its original options;
    # this is provenance, not admission of sample training by an epsilon protocol.
    original=copy.copy(args)
    original.diffusion_prediction_type='epsilon';original.stop_after_updates=32768
    original.checkpoint_every_updates=2048
    original.oxe_formal_protocol=str(HERE/'source_protocol.json')
    original.oxe_preflight_binding=str(HERE/'source_binding.json')
    source=original_loader(original,dataset)
    assert source==reference['offline_training_protocol']
    return {'format':'explicit-main-target-prefix-protocol-v1','sha256':sha(args.oxe_formal_protocol),
        'protocol':arm,'cluster_preflight':source['cluster_preflight'],
        'cluster_preflight_binding_sha256':source['cluster_preflight_binding_sha256'],
        'source_reference_admission':source,'pair_protocol_sha256':sha(HERE/'pair_protocol.json'),
        'scope':'Explicit short diagnostic; original historical preflight binds frozen assets only, not target performance'}


def install():
    import oxe_formal_protocol
    old=oxe_formal_protocol.load_and_validate
    def dispatch(args,dataset=None):
        path=Path(args.oxe_formal_protocol).resolve()
        if path in {HERE/'protocol_epsilon.json',HERE/'protocol_sample.json'}:
            return validate_diagnostic(args,dataset,old)
        return old(args,dataset)
    oxe_formal_protocol.load_and_validate=dispatch
    return old
