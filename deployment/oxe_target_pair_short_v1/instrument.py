"""Declared source instrumentation; original files stay byte preserved."""
import ast
from protocols import HERE


def loop_source():
    path=HERE/'code/step_training_runner.py';source=path.read_text()
    changes=[
        ('    json.dumps(contract, sort_keys=True, allow_nan=False)\n','    DIAG_HOOK.bind(model,api,contract,plan)\n    json.dumps(contract, sort_keys=True, allow_nan=False)\n'),
        ('        indices = plan.begin_batch()\n','        indices = plan.begin_batch()\n        DIAG_HOOK.begin(indices,plan.completed_updates)\n'),
        ('        loss, pose, gripper = api.policy_loss(model, output, actions, criterion,',
         '        DIAG_HOOK.observe(model,api,output,actions,current,masks,criterion)\n        loss, pose, gripper = api.policy_loss(model, output, actions, criterion,'),
        ('        optimizer.zero_grad(set_to_none=True)\n        loss.backward()\n',
         '        optimizer.zero_grad(set_to_none=True)\n        DIAG_HOOK.audit_first(model,optimizer,named)\n        loss.backward()\n'),
        ('        plan.commit_batch()\n','        plan.commit_batch()\n        DIAG_HOOK.commit(plan.completed_updates)\n'),
        ("'language_counts': dict(language_counts)})","'language_counts': dict(language_counts),'diagnostic_target_state':DIAG_HOOK.state()})"),
        ("    (out / 'step_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')",
         "    report['diagnostic_target_state']=DIAG_HOOK.state()\n    report['diagnostic_prefix_completed']=(plan.completed_updates==8192)\n    report['diagnostic_training_admitted']=report.pop('formal_training_admitted')\n    (out / 'step_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')")]
    for old,new in changes:
        assert source.count(old)==1,old;source=source.replace(old,new)
    return path,source


def method_source():
    path=HERE/'code/models.py';tree=ast.parse(path.read_text(encoding='utf-8'))
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='diffusion_loss')
    found=0
    for i,node in enumerate(fn.body):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='noisy_actions' for t in node.targets):
            call=ast.parse('DIAG_HOOK.capture_noise(self,noise,timesteps,noisy_actions)').body[0]
            fn.body.insert(i+1,call);found+=1;break
    assert found==1
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    return path,ast.unparse(module)+'\n'
