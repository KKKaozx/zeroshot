"""Prepare a score-blind LIBERO protocol/task roster from pinned local sources.

Only public task text and raw asset checksums are read. No initial-state values,
HDF5 examples, models, simulator, success predicates or scores are evaluated.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'training_cache/libero_author_8f1084e'
LIBERO = AUTHOR / 'libero/libero'
OUT = ROOT / 'reports/libero_evaluation_protocol_20261026'
SUITES = ('libero_spatial','libero_object','libero_goal','libero_10')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def seeds(task_key, episode_slot):
    def derive(channel, modulo):
        value = json.dumps([20261026,task_key,episode_slot,channel],separators=(',',':'))
        return int(hashlib.sha256(value.encode()).hexdigest()[:16],16) % modulo
    return {'environment_seed':derive('environment',2**31-1),
            'diffusion_episode_seed':derive('diffusion',2**63-1)}


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol_candidate.json').exists(), 'Preserve existing candidate rather than overwrite'
    commit = subprocess.check_output(['git','-C',str(AUTHOR),'rev-parse','HEAD'],text=True).strip()
    assert commit == '8f1084e3132a39270c3a13ebe37270a43ece2a01'
    mapping_file = LIBERO/'benchmark/libero_suite_task_map.py'
    tree = ast.parse(mapping_file.read_text())
    mapping = ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='libero_task_map' for t in n.targets)))
    benchmark_file = LIBERO/'benchmark/__init__.py'
    language_function = next(n for n in ast.parse(benchmark_file.read_text()).body
                            if isinstance(n,ast.FunctionDef) and n.name=='grab_language_from_filename')
    ns = {}
    exec(compile(ast.Module(body=[language_function],type_ignores=[]),'official_language_function','exec'),ns)
    excluded = json.loads((ROOT/'configs/libero_five_condition_plan_20261025.json').read_text(encoding='utf-8'))['engineering_task_excluded_from_final_scores']
    roster,slots = [],[]
    for suite in SUITES:
        assert len(mapping[suite])==10
        for index,name in enumerate(mapping[suite]):
            key = suite+'/'+name
            assert key != excluded
            bddl = LIBERO/'bddl_files'/suite/(name+'.bddl')
            init = LIBERO/'init_files'/suite/(name+'.pruned_init')
            assert bddl.is_file() and init.is_file()
            language = ns['grab_language_from_filename'](name+'.bddl')
            assert language.strip()
            roster.append({'suite':suite,'canonical_task_index':index,'task':name,'task_key':key,
                'official_language':language,'bddl_file':bddl.relative_to(AUTHOR).as_posix(),
                'bddl_sha256':sha(bddl),'initial_states_file':init.relative_to(AUTHOR).as_posix(),
                'initial_states_file_bytes':init.stat().st_size,'initial_states_file_sha256':sha(init),
                'initial_state_vectors_decoded':False})
            for slot in range(100):
                slots.append({'task_key':key,'episode_slot':slot,
                    'official_initial_state_index_rule':'episode_slot % number_of_official_pruned_initial_states',
                    **seeds(key,slot)})
    assert len(roster)==40 and len(slots)==4000
    assert len({row['task_key'] for row in roster})==40
    (OUT/'task_roster.json').write_text(json.dumps(roster,indent=2)+'\n')
    (OUT/'episode_slots_candidate100.json').write_text(json.dumps(slots,indent=2)+'\n')
    execution = json.loads((ROOT/'configs/libero_nominal_execution_candidate_20261025.json').read_text())
    protocol = {
        'format':'libero-score-blind-evaluation-candidate-v1',
        'status':'common_rules_prepared_rollout_count_question_pending',
        'formal_task_scoring_admitted':False,'score_generated':False,
        'teacher_basis':{'training_conditions':'Teacher seventh email:four DINO augmentation conditions x3 training seeds, CLIP combined x1',
            'episode_quote':'100 episodes per task split',
            'count_ambiguity_not_hidden':True,
            'primary_suite_scope_is_implementation_proposal_not_teacher_quote':True},
        'official_author_commit':commit,
        'primary_suites':list(SUITES),'primary_tasks':40,
        'primary_suite_basis':'Pinned official lifelong/evaluate.py accepts exactly spatial/object/goal/10',
        'libero90_role':'Not part of the proposed primary40; no claim of evaluating all130 tasks. Existing single engineering task stays excluded.',
        'engineering_task_excluded_from_final_scores':excluded,
        'task_roster_sha256':sha(OUT/'task_roster.json'),
        'candidate100_slots_sha256':sha(OUT/'episode_slots_candidate100.json'),
        'rollouts':{'selection':None,'recommended_candidate':'100 per individual task',
            'options':{'per_task100':{'episodes_per_task':100,'episodes_per_suite':1000,'original_instruction_episodes_per_model':4000},
                'bounded_per_suite100':{'episodes_per_task':10,'episodes_per_suite':100,'original_instruction_episodes_per_model':400,
                    'scope':'Bounded evaluation only, not automatically the final thesis requirement'}},
            'counts_apply_to_original_instruction_only':True},
        'initial_state_rule':{'source':'Pinned official .pruned_init assets; no demonstrations or synthesized states',
            'index':'episode_slot % number_of_official_pruned_initial_states',
            'duplicates':'Repeat counts/unique initial-state indices must be reported;100 trials need not mean100 distinct physical states',
            'seeds':'SHA256([20261026,task_key,episode_slot,channel]), first64 bits mod2^31-1 for environment or2^63-1 for sampler',
            'paired_across':'Same task/slot/seeds for all model conditions and training seeds; seeds exclude condition, training seed and phrasing',
            'reset_policy_per_episode':'Reset callback call counter and use fixed per-episode sampler seed; never reload/train the model between trials'},
        'language':{'primary':'Official benchmark task.language from grab_language_from_filename; never substitute BDDL wording silently',
            'evaluation_paraphrases':'Five semantically reviewed phrasings, retained outside training; freeze text before scoring those variants',
            'evaluation_paraphrase_bank_ready':False,
            'paraphrase_rollout_budget':None,
            'training_languages':'No LIBERO task language/demo/init is added to OXE training or augmentation banks'},
        'execution':{'native_control_hz':20,'image_height':128,'image_width':128,
            'policy_inputs':{'rgb':'Only upright agentview RGB through the training prepare_image path',
                'language':'Frozen CLIP text encoding of official benchmark language',
                'proprioception':'Last explicit gripper command plus availability; no joint/EEF/object coordinates passed to the learned network',
                'eef_reference':'Measured EEF pose is used only by the fixed native OSC conversion, not as an added learned feature'},
            'camera':'agentview_image','image_convention':'opengl',
            'settle_native_steps':5,'settle_action':[0,0,0,0,0,0,0],
            'settle_gripper_history':'Zero native gripper is neutral; keep command history unknown, not falsely known open/closed',
            'policy_native_step_budget':600,'policy_simulated_seconds':30,
            'policy_decisions_max':150,'native_ticks_per_decision':4,
            'checkpoint':'Immutable final65536 endpoint; no intermediate/best/target-outcome checkpoint selection',
            'ddpm_steps':100,'prediction_type':'epsilon','schedule':'squaredcos_cap_v2',
            'frozen_state':'Exact training digest; use explicit verified original coefficient profile on a different runtime',
            'control':execution['constructor_kwargs'],
            'source_units_and_tcp_calibrated':False,
            'transfer_assumption':'Evaluate the already-declared uncalibrated shared-Cartesian convention as an explicit experimental setting, not as proven calibration'},
        'success':{'predicate':'Official native BDDL goal predicate as returned by LIBERO step done, with goal status recorded',
            'decision':'At least one success during at most600 actual policy control ticks; stop at first success',
            'no_surrogates':'Position loss, reward proxies, arm motion and8-decision engineering connection do not count as task success'},
        'baseline':{'kind':'Native seven-dimensional zero-action policy',
            'initial_states_seeds_and_step_budget':'Identical to learned policies',
            'reuse':'One fixed baseline result per task/slot; reuse across model/training seeds and language variants rather than rerun identical trials'},
        'statistics':{'per_task':'Success count/planned completed trials; never silently omit technical errors',
            'per_suite':'Equal-weight mean over10 task success rates',
            'overall':'Equal-weight mean over4 suite means; also publish each suite separately',
            'training_seeds':'Report mean and sample SD(ddof1) over42/43/44 when all three available; single-seed results have no invented training-seed SD',
            'paired_comparisons':'Match task/slot and training seed; scene resampling clusters by official initial-state index',
            'clip_ablation':'Single training seed; do not treat episode bootstrap as three-seed evidence'},
        'errors_and_resume':{'technical_errors':'Report evaluation incomplete; retry the identical task/slot/seeds only, no resampled easier episodes',
            'native_horizon_failure':'A valid rollout with no goal by600 ticks is a task failure',
            'resume':'Completed episode identities are immutable, skipped only after checkpoint/protocol/asset hashes match',
            'selection':'No algorithm/control/data/budget/checkpoint changes based on final scores; a changed method gets a new exploratory protocol'},
        'storage':{'episode_reports':'Compact per-episode results and trace digests, not per-episode full simulator dumps',
            'videos':'If enabled, choose task/episode identifiers before outcomes, never select successes only'},
        'remaining_before_scoring':['Resolve rollout-count interpretation','Bind actual evaluator code and per-run checkpoint/profile identities','Specify separate paraphrase robustness budget before those scores'],
        'metadata_only':True,'initial_state_vectors_decoded':0,'model_forwards':0,'physics_steps':0,
        'source_sha256':{str(p.relative_to(AUTHOR)):sha(p) for p in (
            benchmark_file,mapping_file,AUTHOR/'libero/lifelong/evaluate.py',AUTHOR/'libero/lifelong/metric.py',
            AUTHOR/'libero/configs/eval/default.yaml',LIBERO/'envs/env_wrapper.py',LIBERO/'envs/bddl_base_domain.py')},
    }
    protocol['candidate_identity']=digest(protocol)
    (OUT/'protocol_candidate.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'candidate_identity':protocol['candidate_identity'],'primary_tasks':40,
        'proposed_original_episode_options':[4000,400],'scoring_admitted':False,
        'raw_asset_hashes_pinned':80,'final_init_values_decoded':0,'simulator_steps':0},indent=2))


if __name__=='__main__':
    main()
