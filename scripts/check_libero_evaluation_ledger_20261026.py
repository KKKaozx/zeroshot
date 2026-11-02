"""New scoring/resume failure boundaries only; no LIBERO/model execution."""
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deployment/libero_original_evaluation_v1'))
from evaluation_core import Ledger, exclusive_run, rollout_episode


class FakeEnv:
    def __init__(self, goal_tick=None, disagree=False):
        self.steps=0
        self.goal_tick=goal_tick
        self.disagree=disagree
    def step(self, action):
        self.steps+=1
        return {}, 0., self.check_success() != self.disagree, {}
    def check_success(self):
        return self.goal_tick is not None and self.steps>=self.goal_tick


commands=[]
def chunk(env, policy, obs, instruction, command, contract, *, native_step_budget):
    commands.append(command)
    n=0
    for _ in range(4):
        obs,_,done,_=env.step([0.]*7)
        n+=1
        if done:break
    return {'observation':obs,'last_open_command':1.,'native_done':done,
            'native_steps':n,'policy_calls':1,
            'trace':[{'native_command_saturated':False} for _ in range(n)]}


def rejected(f):
    try:f()
    except (ValueError,OSError,BlockingIOError):return True
    return False


def main():
    checks={}
    env=FakeEnv(goal_tick=12)
    r=rollout_episode(env,object(),{},'fixed',None,chunk,0,50)
    checks['stop_on_native_goal_with_partial_prefix']=r['success'] and r['policy_native_steps']==7 and env.steps==12
    checks['unknown_command_then_executed_history']=commands==[None,1.]
    env=FakeEnv()
    r=rollout_episode(env,None,{},'fixed',None,chunk,0,50)
    checks['native_timeout_is_failure_and_settle_excluded']=not r['success'] and r['policy_native_steps']==600 and env.steps==605
    checks['done_goal_disagreement_is_technical_error']=rejected(lambda:rollout_episode(FakeEnv(disagree=True),None,{},'fixed',None,chunk,0,50))
    with tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='libero_eval_checks_') as tmp:
        out=Path(tmp)
        binding={'checkpoint':'fixed','protocol':'fixed','runtime':'fixed'}
        roster=[{'task_key':'suite/task','suite':'suite'}]
        slots=[{'task_key':'suite/task','episode_slot':i,'environment_seed':i+1} for i in range(2)]
        with exclusive_run(out):
            ledger=Ledger(out,binding)
            checks['simultaneous_worker_rejected']=rejected(lambda:exclusive_run(out).__enter__())
            ledger.error(slots[0],RuntimeError('render failure'))
            checks['technical_error_not_completed_or_resampled']=ledger.read(slots[0]) is None
            ledger.complete(slots[0],r)
            checks['partial_not_published_as_final']=ledger.summary(slots,roster)['overall_success_rate'] is None
            checks['duplicate_completed_slot_rejected']=rejected(lambda:ledger.complete(slots[0],r))
        with exclusive_run(out):
            resumed=Ledger(out,binding)
            checks['resume_skips_only_completed_fixed_slot']=[s for s in slots if resumed.read(s) is None]==[slots[1]]
            good=dict(r,success=True)
            resumed.complete(slots[1],good)
            checks['complete_denominator_includes_timeout_failure']=resumed.summary(slots,roster)['overall_success_rate']==.5
            checks['changed_identity_rejected']=rejected(lambda:Ledger(out,dict(binding,runtime='changed')))
            p=resumed.path(slots[1])
            row=json.loads(p.read_text());row['result']['success']=False
            p.write_text(json.dumps(row))
            checks['tampered_completed_record_rejected']=rejected(lambda:resumed.read(slots[1]))
    report={'passed':all(checks.values()),'checks':checks,'training_updates':0,
            'real_policy_forwards':0,'real_physics_steps':0,'formal_episodes_completed':0}
    target=ROOT/'reports/libero_evaluation_protocol_20261026/ledger_checks.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['passed']


if __name__=='__main__':main()
