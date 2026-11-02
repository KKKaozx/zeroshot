"""Run two isolated processes with matched frozen optimization protocols."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--phase",choices=["preflight","train"],required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--preflight-report",type=Path)
 a=p.parse_args(); here=Path(__file__).resolve().parent
 contract={name:sha(here/name) for name in ["bridge_xyz_normalization.py","run_bridge_xyz_normalization.py","run_bridge_xyz_pair.py","models.py","diffusion_decoder.py","training_xyz_stats.json","check_bridge_xyz_normalization.py","adapter.py"]}
 if a.phase=="train":
  if a.preflight_report is None:raise ValueError("Passed paired preflight required")
  check=json.loads(a.preflight_report.read_text())
  if not(check["phase"]=="preflight" and check["execution_checks_passed"] and check["contract_sha256"]==contract):raise ValueError("Preflight source contract changed or failed")
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 reports={}
 for group in ["baseline","axis_limits"]:
  cmd=[sys.executable,"-u",str(here/"run_bridge_xyz_normalization.py"),"--pack","/projects/Zeroshot/bridge_expansion_training_v1","--head-report","/projects/Zeroshot/runs/bridge-head-train-192651/report.json","--window-reference","/projects/Zeroshot/runs/bridge-full-chain-191004.json","--preflight","/projects/Zeroshot/runs/bridge-unet-real-193711.json","--baseline-report","/projects/Zeroshot/runs/bridge-unet-train-193761/report.json","--stats",str(here/"training_xyz_stats.json"),"--group",group,"--output",str(a.output/group)]
  if a.phase=="preflight":cmd.append("--temporary-preflight")
  subprocess.run(cmd,check=True)
  reports[group]=json.loads((a.output/group/"report.json").read_text())
 keys=["initial_trainable_sha256","initial_shared_sha256","clip_sha256","order_sha256","module_sha256","architecture","prediction_type","batch_size","epochs","learning_rate","weight_decay","lr_schedule","gradient_clip_norm"]
 for key in keys:
  if reports['baseline']['protocol'][key]!=reports['axis_limits']['protocol'][key]:raise ValueError("Paired protocol mismatch: "+key)
 for group,r in reports.items():
  if not r["clip_unchanged"]:raise ValueError("CLIP changed: "+group)
  if a.phase=="preflight" and not r["execution_checks_passed"]:raise ValueError("Preflight failed")
  if a.phase=="train" and not r["passed"]:raise ValueError("Training integrity failed")
 result={"phase":a.phase,"execution_checks_passed":True,"contract_sha256":contract,"matched_initialization_and_order":True,"reports":{g:str(a.output/g/"report.json") for g in reports},"trained":a.phase=="train","evaluated":False,"temporary_updates_discarded":a.phase=="preflight","performance_improvement_established":None,"limits":["Execution passing is not an accuracy verdict.","Position scaling, offset and corresponding diffusion noise geometry form this single normalization recipe.","Physical-unit xyz MSE, quaternion/gripper targets and gripper pose conditions remain unchanged."]}
 (a.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
 print("XYZ PAIR",a.phase,"PASSED",a.output/"report.json",flush=True)

if __name__=="__main__":main()
