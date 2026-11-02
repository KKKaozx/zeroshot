import json,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1] if len(sys.argv)>1 else '/projects/Zeroshot').resolve()
c=json.loads(Path(__file__).with_name('launch_contract.json').read_text())
used=int(subprocess.check_output(['du','-sB1',str(root)],text=True).split()[0])
required=2*c['raw_archive_bytes']+c['minimum_preflight_reserve_bytes']
free=shutil.disk_usage(root).free
result=dict(project_allocated_bytes=used,additional_peak_bytes_including_tar_and_extraction=required,project_budget_bytes=c['project_budget_bytes'],filesystem_free_bytes=free,passed=used+required<=c['project_budget_bytes'] and free>=required,scope='Conservative project du allocation, not authoritative account quota; no files deleted')
print(json.dumps(result,indent=2));sys.exit(0 if result['passed'] else 1)
