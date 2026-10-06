"""Package existing small frozen-context inputs; never copy raw datasets or CLIP."""
import hashlib
import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "decoder_pair_v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-lr", action="store_true")
    args = parser.parse_args()
    name_prefix = "decoder_author_lr_v1" if args.author_lr else NAME
    output = ROOT.parent / "GPU cluster"
    output.mkdir(exist_ok=True)
    reference = ROOT / "training_cache/references/diffusion_policy"
    existing = ROOT / "results/bridge_diffusion_full_fit_prepare_v1"
    scratch = ROOT / "training_cache/exports"
    scratch.mkdir(parents=True, exist_ok=True)
    files = {
        "run_decoder_pair.py": ROOT / "diagnostics/run_decoder_pair.py",
        "run_full_fit.py": ROOT / "diagnostics/full_x0/run_full_fit.py",
        "cached_diffusion.py": ROOT / "diagnostics/full_x0/cached_diffusion.py",
        "diffusion_decoder.py": ROOT / "code/diffusion_decoder.py",
        "protocol.json": ROOT / "configs/decoder_pair_protocol.json",
        "full_context.pt": existing / "full_context.pt",
        "shared_initial_weights.pt": existing / "shared_initial_weights.pt",
        "AUTHOR_LICENSE": reference / "LICENSE",
    }
    for name in ["conditional_unet1d.py", "conv1d_components.py", "positional_embedding.py"]:
        relative = "diffusion_policy/model/diffusion/" + name
        files[relative] = reference / relative
    old = json.loads((existing / "integrity.json").read_text())
    for name in ["full_context.pt", "shared_initial_weights.pt"]:
        assert hashlib.sha256(files[name].read_bytes()).hexdigest() == old[name]
    total = sum(p.stat().st_size for p in files.values())
    assert total < 20*1024**2, "Unexpected large input; do not build bundle."
    with tempfile.TemporaryDirectory(prefix="decoder-pair-", dir=scratch) as temp:
        base = Path(temp)/name_prefix
        base.mkdir()
        for name, source in files.items():
            target = base/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        integrity = {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in files}
        if args.author_lr:
            source = output/'decoder_pair_187797_results.json'
            original = json.loads(source.read_text(encoding='utf-8-sig'))
            author = next(a for a in original['arms'] if a['arm']=='author')
            assert author['updates']==9875 and author['min_draws']==author['max_draws']==2000
            assert original['identical_training_inputs'] and original['identical_frozen_gripper']
            assert author['frozen_parameters_unchanged']
            assert hashlib.sha256((base/'protocol.json').read_bytes()).hexdigest()==original['protocol_sha256']
            reference = dict(baseline_job_id=187797, protocol_sha256=original['protocol_sha256'],
                source_report_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                identical_training_inputs=True, identical_frozen_gripper=True,
                initial_state_sha256=author['initial_state_sha256'], frozen_state_sha256=author['frozen_state_sha256'],
                paired_inputs_sha256=author['paired_inputs_sha256'], original_learning_rate=0.0003)
            (base/'baseline_reference.json').write_text(json.dumps(reference,indent=2),encoding='utf-8')
            integrity['baseline_reference.json']=hashlib.sha256((base/'baseline_reference.json').read_bytes()).hexdigest()
        (base/"integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
        archive = output/(name_prefix+".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in base.rglob("*"):
                if p.is_file(): z.write(p, str(p.relative_to(Path(temp))))
        with zipfile.ZipFile(archive) as z:
            for name, wanted in integrity.items():
                assert hashlib.sha256(z.read(name_prefix+"/"+name)).hexdigest() == wanted
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (output/(name_prefix+".sha256")).write_text(digest+"  "+archive.name+"\n", encoding="ascii")
    scripts = (["decoder_author_lr_setup.sh", "decoder_author_lr_preflight.sh", "decoder_author_lr_train.sh"] if args.author_lr
               else ["decoder_pair_setup.sh", "decoder_pair_preflight.sh", "decoder_pair_train.sh"])
    for name in scripts:
        (output/name).write_bytes((ROOT/"scripts/cluster"/name).read_bytes().replace(b"\r\n", b"\n"))
    report = dict(bundle=archive.name, bytes=archive.stat().st_size, sha256=digest,
        integrity=integrity, raw_datasets_included=False, clip_weights_included=False,
        scope="Prepared only; no GPU preflight or training executed locally.")
    report_name = "decoder_author_lr_bundle.json" if args.author_lr else "decoder_pair_bundle.json"
    (ROOT/"reports"/report_name).write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Prepared:", archive, "MiB:", round(archive.stat().st_size/1024**2, 2))


if __name__ == "__main__":
    main()
