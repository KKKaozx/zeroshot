"""One local 7B/schema qualification batch; no retries, training or remote inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_language_candidate_review_20261030 import encode, digest, ingest_outputs


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def gpu_memory():
    return subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True).stdout.strip()


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    config_path = ROOT / 'configs/local_7b_language_qualification_20261030.json'
    config = json.loads(config_path.read_bytes())
    payload = (ROOT / config['input_requests']).read_bytes()
    if digest(payload) != config['input_requests_sha256']:
        raise ValueError('Qualification request bytes changed')
    requests = [json.loads(line) for line in payload.splitlines()]
    if len(requests) != 8:
        raise ValueError('Only the fixed eight qualification requests are allowed')
    bundle = json.loads((ROOT / config['input_requests']).with_name('bundle.json').read_bytes())
    for request in requests:
        if digest(encode({k: v for k, v in request.items() if k != 'request_sha256'})) != request['request_sha256']:
            raise ValueError('Individual request changed')
    manifest = json.loads((ROOT / 'reports/local_7b_generator_20261030/asset_manifest.json').read_bytes())
    if (manifest['model_revision'] != config['model_revision'] or manifest['runtime_tag'] != config['runtime_tag']
            or config['model_revision'] != 'bb5d59e06d9551d752d08b292a50eb208b07ab1f'):
        raise ValueError('Wrong qualification model or runtime')
    for item in manifest['files'] + manifest['runtime_files']:
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Acquired asset changed: ' + Path(item['path']).name)
    assets = Path(manifest['assets'])
    executable = assets / 'runtime/llama-server.exe'
    model = assets / 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf'
    before_stats = {i['path']: [Path(i['path']).stat().st_size, Path(i['path']).stat().st_mtime_ns]
                    for i in manifest['files']}
    before_gpu = gpu_memory()
    used, util = [int(v.strip()) for v in before_gpu.splitlines()[0].split(',')]
    if used > 1024 or util > 20:
        raise RuntimeError('Local GPU busy; leave the existing workload untouched')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'protocol.json').write_bytes(config_path.read_bytes())
    (args.output / 'generation_requests.jsonl').write_bytes(payload)
    (args.output / 'responses').mkdir()
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    command = [str(executable), '-m', str(model), '--device', config['runtime_device'],
        '--n-gpu-layers', '99', '--fit', 'off', '--ctx-size', '2048', '--parallel', '1',
        '--threads', '4', '--batch-size', '256', '--ubatch-size', '128', '--no-warmup',
        '--host', '127.0.0.1', '--port', str(port), '--offline', '--no-webui']
    environment = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    environment['LLAMA_ARG_OFFLINE'] = '1'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = 'http://127.0.0.1:' + str(port)
    outputs = {'format': 'local-paraphrase-batch-outputs-v1', 'dataset_identity': bundle['dataset_identity'],
        'request_bundle_sha256': bundle['requests_sha256'],
        'model': {'repository': config['model_repository'], 'revision': config['model_revision'],
                  'quantization': 'Q4_K_M', 'runtime': config['runtime_tag'], 'local_only': True}, 'outputs': []}
    process, report = None, None
    stdout_path, stderr_path = args.output / 'server.stdout.log', args.output / 'server.stderr.log'
    with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
        try:
            process = subprocess.Popen(command, cwd=executable.parent, stdout=stdout, stderr=stderr,
                env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
            (args.output / 'server_process.json').write_bytes(encode({'pid': process.pid, 'command': command}))
            deadline = time.monotonic() + config['limits']['model_load_seconds']
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Local server exited during model loading; see saved stderr')
                try:
                    with opener.open(base_url + '/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok':
                            break
                except urllib.error.URLError as exc:
                    if getattr(exc.reason, 'winerror', None) == 10013:
                        raise RuntimeError('Windows sandbox denied loopback access before any completion request') from exc
                except TimeoutError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Local model load exceeded its fixed budget')
                time.sleep(.5)
            load_seconds = time.monotonic() - started
            print('LOCAL_7B_READY', round(load_seconds, 2), config['runtime_device'], flush=True)
            generation_deadline = time.monotonic() + config['limits']['generation_loop_seconds']
            observed_gpu = [gpu_memory()]
            for index, request in enumerate(requests):
                if time.monotonic() > generation_deadline:
                    raise TimeoutError('Qualification generation loop exceeded fixed budget')
                body = {'model': config['model_repository'], 'messages': request['messages'],
                    'temperature': 0, 'seed': 42, 'repeat_penalty': 1.05,
                    'max_tokens': config['generation']['max_tokens'], 'stream': False,
                    'response_format': {'type': 'json_schema', 'schema': config['generation']['json_schema']}}
                (args.output / 'responses' / ('%02d_request.json' % index)).write_bytes(encode(body))
                tick = time.monotonic()
                post = urllib.request.Request(base_url + '/v1/chat/completions', data=encode(body),
                    headers={'Content-Type': 'application/json'})
                try:
                    with opener.open(post, timeout=config['limits']['seconds_per_completion']) as response:
                        native_bytes = response.read()
                except urllib.error.HTTPError as exc:
                    (args.output / 'responses' / ('%02d_http_error.txt' % index)).write_bytes(exc.read())
                    raise
                (args.output / 'responses' / ('%02d_response.json' % index)).write_bytes(native_bytes)
                native = json.loads(native_bytes)
                content = native['choices'][0]['message']['content']
                if not isinstance(content, str):
                    raise ValueError('Expected a plain generated string')
                outputs['outputs'].append({'instruction_id': request['instruction_id'],
                    'request_sha256': request['request_sha256'], 'raw_generation': content,
                    'seconds': time.monotonic() - tick, 'usage': native.get('usage'),
                    'finish_reason': native['choices'][0].get('finish_reason')})
                (args.output / 'raw_outputs.json').write_bytes(encode(outputs))
                observed_gpu.append(gpu_memory())
                print('LOCAL_7B_OUTPUT', index + 1, '/ 8', request['source'],
                    'seconds', round(outputs['outputs'][-1]['seconds'], 2), flush=True)
            tasks = ingest_outputs(bundle, requests, outputs)
            (args.output / 'candidate_review_tasks.jsonl').write_bytes(b''.join(encode(row) for row in tasks))
            after_stats = {i['path']: [Path(i['path']).stat().st_size, Path(i['path']).stat().st_mtime_ns]
                           for i in manifest['files']}
            if before_stats != after_stats:
                raise ValueError('Asset file size/mtime changed during inference')
            report = {'complete': True, 'generation_calls': len(outputs['outputs']),
                'structural_passed': sum(t['structural_checks_passed'] for t in tasks),
                'request_bundle_sha256': bundle['requests_sha256'], 'raw_outputs_sha256': sha(args.output / 'raw_outputs.json'),
                'protocol_sha256': sha(config_path), 'asset_manifest_sha256': sha(ROOT / 'reports/local_7b_generator_20261030/asset_manifest.json'),
                'all_asset_hashes_verified_before_run': True, 'asset_size_mtime_unchanged': True,
                'server_command': command, 'model_ready_seconds_including_hashes': load_seconds,
                'observed_gpu_memory_mib_utilization': observed_gpu,
                'robot_training_updates': 0, 'new_cluster_jobs': 0, 'automatic_retries': 0,
                'automatic_bulk_generation': False, 'external_project_text_transmission': False,
                'semantic_review_pending': True, 'training_bank_created': False,
                'limits': ['New model, quantization, runtime and schema together; not a capacity-only causal comparison.',
                           'Native runtime file stat guard is not a model-tensor version audit.',
                           'GPU memory snapshots are observed usage, not a continuous peak measurement.']}
        except Exception as exc:
            (args.output / 'failure.json').write_bytes(encode({'complete': False,
                'generation_outputs_saved': len(outputs['outputs']), 'error': type(exc).__name__ + ': ' + str(exc),
                'traceback': traceback.format_exc(), 'automatic_retry': False}))
            raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
    if report is not None:
        report.update(server_stopped=process.poll() is not None, elapsed_seconds=time.monotonic() - started)
        (args.output / 'report.json').write_bytes(encode(report))
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
