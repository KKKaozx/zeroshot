"""Read-only episode integrity/count verification, no model or simulator imports."""
import argparse
import json
from pathlib import Path

from evaluation_core import Ledger, digest, exclusive_run
from run_evaluation import package_check


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    protocol,package=package_check()
    here=Path(__file__).resolve().parent
    slots=json.loads((here/'episode_slots.json').read_text())
    roster=json.loads((here/'task_roster.json').read_text())
    with exclusive_run(args.output):
        saved=json.loads((args.output/'run_binding.json').read_text())
        binding=saved['binding']
        if (binding['protocol_identity']!=protocol['protocol_identity']
                or binding['package_identity']!=package or digest(binding)!=saved['binding_identity']):
            raise ValueError('Package/protocol/run binding differs')
        ledger=Ledger(args.output,binding)
        expected={ledger.path(slot).name for slot in slots if ledger.read(slot) is not None}
        if {p.name for p in (args.output/'episodes').glob('*.json')}!=expected:
            raise ValueError('Unexpected or unplanned episode result')
        summary=ledger.summary(slots,roster)
        latest=json.loads((args.output/'summary.json').read_text())
        if any(latest[k]!=v for k,v in summary.items()):
            raise ValueError('Summary differs from independently reconstructed episode ledger')
        print('LIBERO EPISODE LEDGER VERIFIED; complete episodes retained; incomplete is not a final rate')
        print(json.dumps({k:v for k,v in summary.items() if k not in ('tasks','suites')},indent=2))


if __name__=='__main__':main()
