"""CPU identity-only reproduction: actual frozen assets, no forward or update."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    import torch
    torch.set_num_threads(1);torch.manual_seed(42)
    from models import RobotAdapterModel
    from train import trainable_state_dict
    from step_training_runner import frozen_digest,tokenizer_identity
    from transformers import CLIPTokenizer
    cfg=json.loads((ROOT/'reports/nominal_full_resource_20261025/attempt1/candidate_model_config.json').read_text())
    model=RobotAdapterModel(cfg);saved=trainable_state_dict(model)
    tensors={}
    for name,value in model.state_dict().items():
        if name not in saved:
            tensor=value.detach().cpu().contiguous()
            tensors[name]={'dtype':str(tensor.dtype),'shape':list(tensor.shape),
                'sha256':hashlib.sha256(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
    tokenizer=CLIPTokenizer.from_pretrained(cfg['model']['name'],local_files_only=True)
    out=ROOT/args.output;out.parent.mkdir(parents=True,exist_ok=True)
    report={'pythonhashseed':os.environ.get('PYTHONHASHSEED'),'frozen_sha256':frozen_digest(model,saved),
        'frozen_tensors':tensors,'frozen_key_order':list(tensors),'tokenizer':tokenizer_identity(tokenizer),
        'backend_before_tokenization':json.loads(tokenizer.backend_tokenizer.to_str()),
        'models_on_cpu':True,'forward_calls':0,'optimizer_updates':0}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('frozen_tensors','frozen_key_order','backend_before_tokenization')},indent=2))


if __name__=='__main__':main()
