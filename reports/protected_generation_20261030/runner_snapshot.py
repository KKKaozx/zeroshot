"""One offline native-GBNF generation batch from frozen protected source frames."""
import argparse
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
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'tools'))
from protected_language_frames import ProtectedLanguageFrames
from protected_language_generation import build_grammar, validate_generated_group
from prepare_language_candidate_review_20261030 import encode, digest, ingest_outputs
from run_local_7b_qualification_20261030 import sha, gpu_memory


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    completed = ROOT / 'reports/protected_generation_20261030/report.json'
    if completed.exists() and json.loads(completed.read_bytes()).get('complete'):
        raise ValueError('Protected generation already completed; reuse saved outputs')
    started = time.monotonic()
    protocol_path = ROOT / 'configs/protected_language_generation_20261030.json'
    config = json.loads(protocol_path.read_bytes())
    if config['qualification_variant'] != '7b-protected-gbnf-v1' or config['preparation_revision'] != 2:
        raise ValueError('Wrong protected-generation protocol')
    for name, expected in config['implementation_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Protected source-frame implementation changed')
    request_path = ROOT / config['input_requests']
    payload = request_path.read_bytes()
    bundle = json.loads(request_path.with_name('bundle.json').read_bytes())
    if digest(payload) != config['input_requests_sha256'] or digest(payload) != bundle['requests_sha256']:
        raise ValueError('Protected request bundle changed')
    requests = [json.loads(line) for line in payload.splitlines()]
    if len(requests) != 8:
        raise ValueError('Only eight bound protected requests are allowed')
    checker = ProtectedLanguageFrames(ROOT / config['protected_profile'], bundle['dataset_identity'],
        {r['instruction_id']: r['original_instruction'] for r in requests})
    if checker.contract != config['protected_frame_contract']:
        raise ValueError('Protected frame contract changed')
    built_grammars = {}
    for request in requests:
        if digest(encode({k: v for k, v in request.items() if k != 'request_sha256'})) != request['request_sha256']:
            raise ValueError('Protected request changed')
        built = build_grammar(checker.frames[request['instruction_id']], checker)
        grammar = (ROOT / request['grammar_path']).read_text(encoding='utf-8')
        if built['grammar'] != grammar or digest(grammar.encode()) != request['grammar_sha256']:
            raise ValueError('Native grammar differs from offline-checked protected language')
        built_grammars[request['instruction_id']] = built
    manifest_path = ROOT / 'reports/local_7b_generator_20261030/asset_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['model_revision'] != config['model_revision'] or manifest['runtime_tag'] != config['runtime_tag']:
        raise ValueError('Wrong local model/runtime')
    for item in manifest['files'] + manifest['runtime_files']:
        if sha(Path(item['path'])) != item['sha256']:
            raise ValueError('Pinned asset changed')
    before_stats = {row['path']: [Path(row['path']).stat().st_size, Path(row['path']).stat().st_mtime_ns]
                    for row in manifest['files']}
    used, util = [int(s.strip()) for s in gpu_memory().splitlines()[0].split(',')]
    if used > 1024 or util > 20:
        raise RuntimeError('GPU busy; preserve existing workload')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'protocol.json').write_bytes(protocol_path.read_bytes())
    (args.output / 'generation_requests.jsonl').write_bytes(payload)
    (args.output / 'responses').mkdir()
    assets = Path(manifest['assets']); executable = assets / 'runtime/llama-server.exe'
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0)); port = probe.getsockname()[1]
    command = [str(executable), '-m', str(assets / 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf'),
        '--device', config['runtime_device'], '--n-gpu-layers', '99', '--fit', 'off',
        '--ctx-size', '2048', '--parallel', '1', '--threads', '4', '--batch-size', '256',
        '--ubatch-size', '128', '--no-warmup', '--host', '127.0.0.1', '--port', str(port),
        '--offline', '--no-webui']
    environment = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    environment['LLAMA_ARG_OFFLINE'] = '1'
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    local = 'http://127.0.0.1:' + str(port)
    outputs = {'format': 'local-paraphrase-batch-outputs-v1', 'dataset_identity': bundle['dataset_identity'],
        'request_bundle_sha256': bundle['requests_sha256'],
        'model': {'repository': config['model_repository'], 'revision': config['model_revision'],
            'quantization': 'Q4_K_M', 'runtime': config['runtime_tag'], 'mechanism': 'protected-gbnf-v1', 'local_only': True},
        'outputs': []}
    process, result, checks, memory = None, None, [], []
    with (args.output / 'server.stdout.log').open('wb') as stdout, (args.output / 'server.stderr.log').open('wb') as stderr:
        try:
            process = subprocess.Popen(command, cwd=executable.parent, env=environment,
                stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            (args.output / 'server_process.json').write_bytes(encode({'pid': process.pid, 'command': command}))
            deadline = time.monotonic() + config['limits']['model_load_seconds']
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Native server exited; inspect saved stderr')
                try:
                    with opener.open(local + '/health', timeout=2) as response:
                        if json.load(response).get('status') == 'ok': break
                except urllib.error.URLError as exc:
                    if getattr(exc.reason, 'winerror', None) == 10013:
                        raise RuntimeError('Windows denied loopback access before any generation') from exc
                except TimeoutError:
                    pass
                if time.monotonic() > deadline: raise TimeoutError('Fixed local readiness budget exceeded')
                time.sleep(.5)
            ready = time.monotonic() - started
            print('PROTECTED_READY', round(ready, 2), flush=True)
            deadline = time.monotonic() + config['limits']['generation_loop_seconds']
            for index, request in enumerate(requests):
                if time.monotonic() > deadline: raise TimeoutError('Fixed generation budget exceeded')
                built = built_grammars[request['instruction_id']]
                body = {'model': config['model_repository'], 'messages': request['messages'],
                    'temperature': 0, 'seed': 42, 'repeat_penalty': 1.05, 'max_tokens': 384,
                    'stream': False, 'grammar': built['grammar']}
                (args.output / 'responses' / ('%02d_request.json' % index)).write_bytes(encode(body))
                post = urllib.request.Request(local + '/v1/chat/completions', data=encode(body),
                    headers={'Content-Type': 'application/json'})
                tick = time.monotonic()
                try:
                    with opener.open(post, timeout=config['limits']['seconds_per_completion']) as response:
                        native_bytes = response.read()
                except urllib.error.HTTPError as exc:
                    (args.output / 'responses' / ('%02d_http_error.txt' % index)).write_bytes(exc.read()); raise
                (args.output / 'responses' / ('%02d_response.json' % index)).write_bytes(native_bytes)
                native = json.loads(native_bytes); content = native['choices'][0]['message']['content']
                outputs['outputs'].append({'instruction_id': request['instruction_id'],
                    'request_sha256': request['request_sha256'], 'raw_generation': content,
                    'seconds': time.monotonic() - tick, 'usage': native.get('usage'),
                    'finish_reason': native['choices'][0].get('finish_reason')})
                (args.output / 'raw_outputs.json').write_bytes(encode(outputs))
                generated = json.loads(content)
                if set(generated) == {'abstain_reason'}:
                    checks.append({'instruction_id': request['instruction_id'], 'abstained': True, 'usable_for_training': False})
                else:
                    check = validate_generated_group(checker.frames[request['instruction_id']], built,
                        generated['paraphrases'], checker)
                    checks.append({'instruction_id': request['instruction_id'], 'abstained': False, **check})
                memory.append(gpu_memory())
                print('PROTECTED_OUTPUT', index + 1, '/8', 'abstained', checks[-1]['abstained'], flush=True)
            tasks = ingest_outputs(bundle, requests, outputs, checker)
            (args.output / 'candidate_review_tasks.jsonl').write_bytes(b''.join(encode(row) for row in tasks))
            (args.output / 'constraint_checks.json').write_bytes(encode(checks))
            after_stats = {row['path']: [Path(row['path']).stat().st_size, Path(row['path']).stat().st_mtime_ns]
                           for row in manifest['files']}
            if before_stats != after_stats: raise ValueError('Model asset file size/mtime changed')
            result = {'complete': True, 'generation_calls': len(outputs['outputs']),
                'structural_passed': sum(task['structural_checks_passed'] for task in tasks),
                'protected_group_passed': sum(not c['abstained'] for c in checks),
                'abstentions': sum(c['abstained'] for c in checks),
                'raw_outputs_sha256': sha(args.output / 'raw_outputs.json'), 'request_bundle_sha256': bundle['requests_sha256'],
                'protocol_sha256': sha(protocol_path), 'protected_frame_contract': checker.contract,
                'all_pinned_assets_verified_before_run': True, 'model_asset_size_mtime_unchanged': True,
                'ready_seconds_including_hashes': ready, 'observed_gpu_mib_utilization': memory,
                'server_command': command, 'automatic_retries': 0, 'unconstrained_fallback': False,
                'external_project_text_transmission': False, 'training_updates': 0, 'new_cluster_jobs': 0,
                'automatic_bulk_generation': False, 'semantic_review_pending': True, 'training_bank_created': False,
                'scope': 'Eight development frames with human-reviewed finite forms; no independent or full-pool language-generalization claim'}
        except Exception as exc:
            (args.output / 'failure.json').write_bytes(encode({'complete': False,
                'generation_outputs_saved': len(outputs['outputs']), 'error': type(exc).__name__ + ': ' + str(exc),
                'traceback': traceback.format_exc(), 'automatic_retry': False})); raise
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
    if result is not None:
        result.update(server_stopped=process.poll() is not None, elapsed_seconds=time.monotonic() - started)
        (args.output / 'report.json').write_bytes(encode(result))
        print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
