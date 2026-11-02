"""Static review of actual main-runner wiring; never imports a model or data."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/oxe_runner_readiness_20261020'


def keyset(node):
    return {key.value for key in node.keys if isinstance(key,ast.Constant) and isinstance(key.value,str)}


def main():
    path=ROOT/'code/train.py'
    source=path.read_text(encoding='utf-8')
    tree=ast.parse(source)
    functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
    checkpoint_nodes=[x for x in ast.walk(tree) if isinstance(x,ast.Dict)
                      and {'trainable_state_dict','optimizer_state_dict','scheduler_state_dict','epoch'}.issubset(keyset(x))]
    assert len(checkpoint_nodes)==1
    checkpoint=checkpoint_nodes[0]
    checkpoint_keys=keyset(checkpoint)
    config_model=next(x for x in ast.walk(functions['build_config']) if isinstance(x,ast.Dict) and 'decoder_type' in keyset(x))
    cli_flags=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
            cli_flags.extend(x.value for x in node.args if isinstance(x,ast.Constant) and isinstance(x.value,str) and x.value.startswith('--'))
    calls=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call):
            calls.append(ast.unparse(node.func))
    assert '--max-steps-per-epoch' in cli_flags
    assert not any(x in cli_flags for x in ['--max-updates','--max-training-updates','--max-steps'])
    assert 'diffusion_architecture' not in keyset(config_model)
    assert not any(x.endswith(('get_rng_state','set_rng_state','get_rng_state_all','set_rng_state_all','getstate','setstate','get_state','set_state')) for x in calls)
    old=json.loads((ROOT/'reports/oxe_pose_source_support_20261020/report.json').read_text())
    current_hashes={name:hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest() for name in old['production_source_sha256']}
    assert current_hashes==old['production_source_sha256']
    OUT.mkdir(exist_ok=False)
    result={
        'review_complete':True,'scope':'Static AST and source review of main train.py only; no runtime qualification',
        'production_source_sha256':current_hashes,
        'main_checkpoint_dictionary_line':checkpoint.lineno,
        'main_checkpoint_literal_keys':sorted(checkpoint_keys),
        'model_optimizer_scheduler_saved':True,
        'python_numpy_torch_rng_capture_restore_calls_found':False,
        'training_generator_recreated_from_seed_on_resume':True,
        'sampling_generator_state_and_batch_cursor_saved':False,
        'resume_boundary':'epoch+1; no mid-epoch cursor',
        'strict_global_update_limit_cli_present':False,
        'epoch_limit_and_per_epoch_batch_limit_present':True,
        'schedule_unit':'epoch',
        'fresh_build_config_sets_diffusion_architecture':False,
        'models_supports_multiscale':True,
        'fresh_main_model_architecture_default':'compact',
        'balanced_sampling_available':'equal probability by schema, replacement; num_samples=len(train_indices)',
        'four_source_training_weights_and_loss_accounting_frozen':False,
        'required_before_formal_submission':[
            'Explicitly freeze architecture/objective and source sampling/valid-dimension loss accounting in the actual runner config.',
            'Use a fixed total optimizer-update budget and an immutable schedule horizon shared across all conditions/seeds.',
            'Checkpoint and restore Python/NumPy/Torch CPU/CUDA RNG plus exact sampling-generator state, current schedule and batch cursor.',
            'Verify resumed versus uninterrupted sample IDs, noise/timesteps, learning rates, optimizer state, parameters and predictions at a common update.',
            'Consume the existing frozen episode roles via an admitted window manifest after physical data contracts are supported.',
        ],
        'not_a_historical_generalization_root_cause_verdict':True,
        'production_code_changed':False,'model_or_data_imported':False,
        'training_or_inference_run':False,'training_ready':False,
    }
    (OUT/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['review_complete','model_optimizer_scheduler_saved','python_numpy_torch_rng_capture_restore_calls_found','strict_global_update_limit_cli_present','fresh_main_model_architecture_default','training_ready']},indent=2))


if __name__=='__main__':main()
