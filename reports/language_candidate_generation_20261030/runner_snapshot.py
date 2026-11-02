"""Generate one bound eight-instruction batch locally; no training or retries."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_language_candidate_review_20261030 import digest, encode, select_requests


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, default=ROOT / 'reports/language_candidate_requests_20261030')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    payload = (args.requests / 'generation_requests.jsonl').read_bytes()
    bundle = json.loads((args.requests / 'bundle.json').read_bytes())
    if digest(payload) != bundle['requests_sha256']:
        raise ValueError('Bound request bytes changed')
    requests = [json.loads(line) for line in payload.splitlines()]
    queue_path = ROOT / 'reports/project_goal_audit_20261030/training_language_review_queue.jsonl'
    queue_bytes = queue_path.read_bytes()
    if digest(queue_bytes) != bundle['review_queue_sha256']:
        raise ValueError('Training review queue changed')
    expected = select_requests([json.loads(line) for line in queue_bytes.splitlines()], bundle['dataset_identity'])
    if requests != expected or len(requests) != bundle['request_count'] or len(requests) != 8:
        raise ValueError('Only the fixed eight previously ungenerated training requests are allowed')
    if Counter(row['source'] for row in requests) != dict.fromkeys(('Bridge', 'Language Table', 'BC-Z', 'Fractal'), 2):
        raise ValueError('Expected two instructions per source')
    inventory = json.loads((queue_path.parent / 'language_readiness_inventory.json').read_bytes())
    if bundle['dataset_identity'] != inventory['dataset_identity']:
        raise ValueError('Not the bound current training dataset')
    metadata_path = ROOT / 'reports/local_paraphrase_model_assets_20261025.json'
    metadata = json.loads(metadata_path.read_bytes())
    assets = Path(metadata['assets'])
    for name, info in metadata['files'].items():
        path = assets / name
        if path.stat().st_size != info['bytes'] or sha(path) != info['sha256']:
            raise ValueError('Pinned local model asset changed: ' + name)
    gpu_before = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.strip()
    used_mib, utilization = [int(x.strip()) for x in gpu_before.splitlines()[0].split(',')]
    if used_mib > 1024 or utilization > 20:
        raise RuntimeError('Local GPU busy; do not disturb the existing workload')
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 5 * 1024**3:
        raise RuntimeError('Need CUDA and 5 GiB free; no CPU fallback')
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'generation_requests.jsonl').write_bytes(payload)
    outputs = {'format': 'local-paraphrase-batch-outputs-v1',
        'dataset_identity': bundle['dataset_identity'],
        'request_bundle_sha256': bundle['requests_sha256'],
        'model': {'repository': metadata['repository'], 'revision': metadata['revision'], 'local_only': True},
        'outputs': []}
    try:
        load_started = time.monotonic()
        tokenizer = AutoTokenizer.from_pretrained(str(assets), local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(str(assets), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, dtype=torch.float16, attn_implementation='sdpa')
        model.requires_grad_(False).eval().cuda()
        parameter_versions = {name: p._version for name, p in model.named_parameters()}
        load_seconds = time.monotonic() - load_started
        torch.cuda.reset_peak_memory_stats()
        deadline = time.monotonic() + 180
        for index, request in enumerate(requests):
            if time.monotonic() >= deadline:
                raise TimeoutError('Fixed batch generation exceeded 180 seconds; do not retry automatically')
            prompt = tokenizer.apply_chat_template(request['messages'], tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(prompt, return_tensors='pt').to('cuda')
            seed = int(request['request_sha256'][:8], 16)
            torch.cuda.synchronize(); tick = time.monotonic()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
                torch.manual_seed(seed)
                tokens = model.generate(**encoded, max_new_tokens=384, max_time=20,
                    do_sample=False, repetition_penalty=1.05)
            torch.cuda.synchronize()
            generated = tokens[0, encoded['input_ids'].shape[-1]:]
            outputs['outputs'].append({'instruction_id': request['instruction_id'],
                'request_sha256': request['request_sha256'], 'seed': seed,
                'rendered_prompt_sha256': digest(prompt.encode()),
                'input_tokens': int(encoded['input_ids'].numel()), 'output_tokens': int(len(generated)),
                'seconds': time.monotonic() - tick,
                'raw_generation': tokenizer.decode(generated, skip_special_tokens=True)})
            (args.output / 'raw_outputs.json').write_bytes(encode(outputs))
            print('LOCAL_CANDIDATE', index + 1, '/ 8', request['source'],
                  'seconds', round(outputs['outputs'][-1]['seconds'], 2), flush=True)
        if any(p.requires_grad or p.grad is not None or p._version != parameter_versions[name]
               for name, p in model.named_parameters()):
            raise ValueError('Frozen local generation parameter guard failed')
        report = {'complete': True, 'request_count': 8, 'generation_calls': len(outputs['outputs']),
            'dataset_identity': bundle['dataset_identity'], 'request_bundle_sha256': bundle['requests_sha256'],
            'assets_manifest_sha256': sha(metadata_path), 'all_pinned_asset_hashes_checked': True,
            'raw_outputs_sha256': sha(args.output / 'raw_outputs.json'),
            'model_load_seconds': load_seconds, 'elapsed_seconds': time.monotonic() - started,
            'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            'gpu': torch.cuda.get_device_name(0), 'gpu_before_mib_utilization': gpu_before,
            'max_new_tokens': 384, 'max_time_per_call_seconds': 20, 'max_generation_loop_seconds': 180,
            'dtype': 'float16', 'attention': 'sdpa', 'do_sample': False, 'repetition_penalty': 1.05,
            'parameter_versions_unchanged_and_no_gradients': True,
            'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'safetensors')},
            'training_updates': 0, 'new_cluster_jobs': 0, 'network_used_for_generation': False,
            'packages_installed': False, 'semantic_approvals': 0, 'training_bank_created': False,
            'automatic_retries': 0, 'automatic_bulk_generation': False}
        (args.output / 'report.json').write_bytes(encode(report))
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        (args.output / 'failure.json').write_bytes(encode({'complete': False,
            'completed_generation_calls': len(outputs['outputs']), 'error': type(exc).__name__ + ': ' + str(exc),
            'traceback': traceback.format_exc(), 'automatic_retry': False}))
        raise


if __name__ == '__main__':
    main()
