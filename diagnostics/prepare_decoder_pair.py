"""Package existing small frozen-context inputs; never copy raw datasets or CLIP."""
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "decoder_pair_v1"


def main():
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
        base = Path(temp)/NAME
        base.mkdir()
        for name, source in files.items():
            target = base/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        integrity = {name: hashlib.sha256((base/name).read_bytes()).hexdigest() for name in files}
        (base/"integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
        archive = output/(NAME+".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in base.rglob("*"):
                if p.is_file(): z.write(p, str(p.relative_to(Path(temp))))
        with zipfile.ZipFile(archive) as z:
            for name, wanted in integrity.items():
                assert hashlib.sha256(z.read(NAME+"/"+name)).hexdigest() == wanted
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (output/(NAME+".sha256")).write_text(digest+"  "+archive.name+"\n", encoding="ascii")
    for name in ["decoder_pair_setup.sh", "decoder_pair_preflight.sh", "decoder_pair_train.sh"]:
        (output/name).write_bytes((ROOT/"scripts/cluster"/name).read_bytes().replace(b"\r\n", b"\n"))
    report = dict(bundle=archive.name, bytes=archive.stat().st_size, sha256=digest,
        integrity=integrity, raw_datasets_included=False, clip_weights_included=False,
        scope="Prepared only; no GPU preflight or training executed locally.")
    (ROOT/"reports/decoder_pair_bundle.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("Prepared:", archive, "MiB:", round(archive.stat().st_size/1024**2, 2))


if __name__ == "__main__":
    main()
