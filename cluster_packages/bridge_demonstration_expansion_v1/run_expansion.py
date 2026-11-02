"""Execute the pinned207558 training runner with explicit size/input adaptations."""
import ast
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def adapted_source(source):
    """Only dimensions, resource gates, input adapter and replay hook may differ."""
    replacements = {
        'CODE=Path(__file__).resolve().parent': 'CODE=OVERRIDE_CODE',
        'from coverage_inputs import load_inputs,tokens': 'from expanded_inputs import load_inputs,tokens,old_replay_guard',
        "sha(CODE/'coverage_inputs.py')": "sha(CODE/'expanded_inputs.py')",
        '        trainables=[v for v in model.parameters() if v.requires_grad]':
        "        if args.stage=='preflight':old_replay_guard(args,CODE,p,model,visual,language,mask,features,items,r)\n"
        '        trainables=[v for v in model.parameters() if v.requires_grad]',
    }
    for old,new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old,new)
    # Change only numeric literal nodes, never strings, losses, seeds or schedules.
    tree = ast.parse(source)
    counts = {}
    class Sizes(ast.NodeTransformer):
        def visit_Constant(self,node):
            mapping={1836:3624,1876:3664,7.5:12.0,9000:12600}
            if type(node.value) in (int,float) and node.value in mapping:
                counts[str(node.value)]=counts.get(str(node.value),0)+1
                return ast.copy_location(ast.Constant(mapping[node.value]),node)
            return node
    tree = Sizes().visit(tree)
    assert set(counts) == {'1836','1876','7.5','9000'}, counts
    return ast.unparse(ast.fix_missing_locations(tree))+'\n',counts


def main():
    contract=json.loads((HERE/'adaptation_contract.json').read_text())
    original=HERE.parent/'bridge_coverage_replication_v1/run_training_coverage.py'
    assert sha(original)==contract['original_runner_sha256']
    source,counts=adapted_source(original.read_text())
    assert hashlib.sha256(source.encode()).hexdigest()==contract['effective_runner_sha256']
    # The executable source and wrapper are both preserved beside the report.
    assert '--output' in sys.argv
    output=Path(sys.argv[sys.argv.index('--output')+1])
    assert not output.exists(), 'Preserve previous run evidence'
    namespace=dict(__file__=str(original),__name__='frozen_coverage806_runner',OVERRIDE_CODE=HERE)
    exec(compile(source,str(original),'exec'),namespace)
    try:
        namespace['main']()
    finally:
        if output.exists():
            (output/'effective_runner_snapshot.py').write_text(source)
            (output/'wrapper_snapshot.py').write_bytes(Path(__file__).read_bytes())
            (output/'adaptation_contract_snapshot.json').write_bytes((HERE/'adaptation_contract.json').read_bytes())
            report_path=output/'report.json'
            if report_path.exists():
                r=json.loads(report_path.read_text())
                r['adaptation']=dict(wrapper_sha256=sha(Path(__file__)),contract_sha256=sha(HERE/'adaptation_contract.json'),
                    effective_runner_sha256=contract['effective_runner_sha256'],literal_replacements=counts,
                    original_runner_unchanged_sha256=sha(original))
                report_path.write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':
    main()
