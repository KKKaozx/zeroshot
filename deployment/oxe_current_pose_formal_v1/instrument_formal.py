"""Promote the exact passed run_arm by declared mechanical edits only."""
import hashlib
import inspect
from binding import HERE,read,sha

TAIL='''    import formal_state
    from completed_policy_export import model_sha
    initial={'source_optimizer_sha256':source_optimizer,'source_rng_sha256':source_rng,
        'initial_port_sha256':port_initial,'source_learned_sha256':reference['source_learned_sha256'],
        'zero_endpoint_replay':replay}
    segments=[os.environ['SLURM_JOB_ID']]
    if resume_path is not None:
        restored=formal_state.restore(resume_path,model,port,optimizer,plan,schedule,new_contract,api.trainable_state_dict)
        assert {k:restored['initial'][k] for k in initial if k!='zero_endpoint_replay'}=={k:initial[k] for k in initial if k!='zero_endpoint_replay'}
        rows=restored['rows'];noise.events=restored['noise_stream'];initial=restored['initial']
        segments=restored['training_jobs']+[os.environ['SLURM_JOB_ID']]
    start_update=plan.completed_updates
    assert len(rows)==len(noise.events)==plan.completed_updates-8192
    for oldrow in rows:
        j=oldrow['update']-8193
        assert oldrow['indices']==suffix['batches'][j].tolist()
    while plan.completed_updates<10240:
        row=step();rows.append(row)
        if row['update']<=8194:
            expected=QUALIFIED_PREFLIGHT_REPORT['groups'][role]['rows'][row['update']-8193]
            assert all(row[k]==v for k,v in expected.items() if k!='measured_update_seconds'),'Passed first two updates changed'
        if plan.completed_updates%1024==0 or plan.completed_updates==10240:
            assert frozen_digest(model,saved_keys)==reference['frozen_sha256']
            formal_state.save(output/'latest_state.pt',model,port,optimizer,plan,schedule,new_contract,api.trainable_state_dict,
                {'rows':rows,'noise_stream':noise.events,'initial':initial,'training_jobs':segments})
    assert len(rows)==len(noise.events)==2048 and frozen_digest(model,saved_keys)==reference['frozen_sha256']
    if role=='mask_control':assert torch.count_nonzero(port.weight[:,:9])==0
    else:assert torch.count_nonzero(port.weight[:,:9])>0
    result={'passed':True,'role':role,'completed_update':10240,'additional_updates':2048,
        'lr_horizon':65536,'initial':initial,'training_jobs':segments,'first_two_updates_match_preflight':True,
        'base_learned_sha256':model_sha(api.trainable_state_dict(model)),'port_sha256':digest(port.state_dict()),
        'contract_sha256':__import__('completed_policy_export').contract_sha(new_contract),
        'frozen_state_sha256':frozen_digest(model,saved_keys),'frozen_restore_events':events,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/1024**3,'segment_new_updates':10240-start_update,
        'full_core_port_adam_plan_rng_retained':True,'not_completed65536_horizon':True,
        'formal_contract_sha256':FORMAL_CONTRACT_SHA,'performance_verdict':None,'automatic_following_action':False}
    (output/'arm_receipt.json').write_text(json.dumps(result,indent=2)+'\\n')
    print('CURRENT_POSE_FORMAL_ARM_COMPLETE',role,json.dumps(result),flush=True)
    return result
'''


def source(pg):
    s=inspect.getsource(pg.run_arm)
    assert hashlib.sha256(s.encode()).hexdigest()==read(HERE/'formal_contract.json')['passed_run_arm_sha256']
    replacements=[
        ('def run_arm(root,role,payload,data,output,reference,stats):','def run_arm(root,role,payload,data,output,reference,stats,resume_path=None):'),
        ("if role=='mask_control':replay=replay_and_features", "if role=='mask_control' and resume_path is None:replay=replay_and_features"),
        ("'format':'current-pose-preflight-augmentation-v1'","'format':'current-pose-bounded-formal-augmentation-v1','formal_contract_sha256':FORMAL_CONTRACT_SHA,'preflight_report_sha256':QUALIFIED_PREFLIGHT_REPORT_SHA"),
        ("'logical_steps':2,'physical_optimizer_calls_including_restore_replay':3","'logical_steps':2048,'physical_optimizer_calls':2048"),
        ('        torch.cuda.synchronize();tick=time.monotonic()','        rates_used=[g[\'lr\'] for g in optimizer.param_groups]\n        torch.cuda.synchronize();tick=time.monotonic()'),
        ("row={'update':plan.completed_updates,'indices':ix,","row={'update':plan.completed_updates,'indices':ix,'lr_used':rates_used,'valid_dimensions':masks.sum((0,1)).cpu().tolist(),"),
        ("        print('CURRENT_POSE_ENGINEERING_UPDATE',role,json.dumps(row),flush=True);return row",
         "        if plan.completed_updates<=8194 or plan.completed_updates%256==0 or plan.completed_updates==10240:\n            print('CURRENT_POSE_FORMAL_UPDATE',role,json.dumps(row),flush=True)\n        return row")]
    for old,new in replacements:
        assert s.count(old)==1,old;s=s.replace(old,new)
    cut=s.index('    rows.append(step())\n');s=s[:cut]+TAIL
    compile(s,'<qualified-pose-formal-arm>','exec');return s


def install(pg,preflight_report,preflight_report_sha):
    s=source(pg);pg.__dict__.update(FORMAL_CONTRACT_SHA=sha(HERE/'formal_contract.json'),QUALIFIED_PREFLIGHT_REPORT=preflight_report,
        QUALIFIED_PREFLIGHT_REPORT_SHA=preflight_report_sha)
    exec(compile(s,'<qualified-pose-formal-arm>','exec'),pg.__dict__)
    return pg.run_arm,s


def prepare_source(pg):
    s=inspect.getsource(pg.replay_and_features)
    assert hashlib.sha256(s.encode()).hexdigest()==read(HERE/'formal_contract.json')['passed_replay_source_sha256']
    old='def replay_and_features(root,data,model,port,tokenizer,api,stats,selection):'
    assert s.count(old)==1;s=s.replace(old,'def prepare_inputs(root,data,model,port,tokenizer,api,stats,selection):')
    marker='        contexts=[]\n';assert s.count(marker)==1;s=s[:s.index(marker)]
    s+='''        target=np.stack([x[3].numpy() for x in items]);masks=np.stack([x[4].numpy() for x in items])
        with np.load(root/'runs/oxe-target-short-eval-fixed-232091/predictions_and_targets.npz') as z:
            assert np.array_equal(target,z['targets']) and np.array_equal(masks,z['masks'])
        return {'visual':visual,'language':language,'attention':mask,'features':state,
            'current':torch.stack([x[2] for x in items]).cuda(),'targets':target,'masks':masks}
'''
    compile(s,'<qualified-fixed-evaluation-inputs>','exec');return s
