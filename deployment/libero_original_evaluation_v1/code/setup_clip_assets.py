"""Fetch one pinned safetensors CLIP variant into cluster project storage."""
import hashlib
import json
from pathlib import Path
import subprocess
import importlib.metadata as metadata
from huggingface_hub import HfApi, snapshot_download

ROOT = Path('/projects/Zeroshot')
REVISION = '32bd64288804d66eefd0ccbe215aa642df71cc41'
FILES = ['model.safetensors','config.json','preprocessor_config.json','merges.txt',
         'vocab.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']


def main():
    # Ensure inheritance worked; installing transformers must not replace PyTorch.
    assert metadata.version('torch') == '2.5.1+cu118'
    assert metadata.version('transformers') == '5.17.0'
    info = HfApi().model_info('openai/clip-vit-large-patch14',revision=REVISION,files_metadata=True)
    assert info.sha == REVISION
    sizes = {f.rfilename:f.size for f in info.siblings if f.rfilename in FILES}
    assert set(sizes) == set(FILES) and all(isinstance(n,int) and n>0 for n in sizes.values())
    total = sum(sizes.values())
    assert total < 2*1024**3
    used = int(subprocess.check_output(['du','-sb',str(ROOT.resolve())],text=True).split()[0])
    assert used + total + 2*1024**3 + 15*1024**3 < 200_000_000_000, 'Insufficient project planning headroom'
    destination = ROOT/'baseline_setup/clip-vit-large-patch14-v1'
    print('Pinned assets bytes:',total,'Project apparent used bytes:',used,flush=True)
    snapshot_download('openai/clip-vit-large-patch14',revision=REVISION,allow_patterns=FILES,local_dir=destination)
    hashes={}
    for name,size in sizes.items():
        path=destination/name
        assert path.stat().st_size == size,name
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
        hashes[name]=h.hexdigest()
    from transformers import CLIPModel, CLIPTokenizer
    tokenizer=CLIPTokenizer.from_pretrained(str(destination),local_files_only=True)
    model=CLIPModel.from_pretrained(str(destination),local_files_only=True,use_safetensors=True)
    assert model.vision_model.config.hidden_size == 1024 and model.text_model.config.hidden_size == 768
    assert tokenizer('open the drawer',return_tensors='pt')['input_ids'].shape[0] == 1
    report=dict(passed=True,trained=False,gpu_inference_executed=False,repo='openai/clip-vit-large-patch14',
        revision=REVISION,assets_bytes=total,files_sha256=hashes,project_apparent_used_bytes_before=used,
        live_quota_verified=False,environment='multitask-preflight-v1 inherits bridge-diffusion PyTorch',
        versions={n:metadata.version(n) for n in ('torch','transformers','huggingface-hub','safetensors')})
    (ROOT/'baseline_setup/multitask-setup-report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('MULTITASK ENVIRONMENT AND PINNED CLIP CPU LOAD: PASSED',flush=True)


if __name__ == '__main__': main()
