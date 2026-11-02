"""Verify and unpack only the pinned simulator, never LIBERO demonstrations."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import zipfile

from evaluation_core import file_sha
from run_evaluation import package_check, verify_files


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',type=Path,required=True)
    args=parser.parse_args()
    package_check()
    here=Path(__file__).resolve().parent
    reference=json.loads((here/'simulator_archive.json').read_text())
    archive=args.project_root/reference['file']
    if file_sha(archive)!=reference['sha256']:
        raise ValueError('Pinned simulator archive differs')
    output=args.project_root/'libero_runtime_v1'
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target=(output/member.filename).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError('Archive member escapes simulator directory')
            if target.exists():
                if target.is_file() and target.read_bytes()==z.read(member):
                    continue
                raise ValueError('Existing simulator file differs: '+str(target))
            z.extract(member,output)
    verify_files(output,json.loads((here/'simulator_files.json').read_text()))
    import torch
    if torch.__version__!='2.5.1+cu118':
        raise ValueError('Evaluation must inherit existing torch unchanged')
    print(json.dumps({'simulator_files_verified':True,'torch':torch.__version__,
        'simulator_root':str(output),'numpy':metadata.version('numpy'),
        'training':0,'physics':0,'environment_gpu_rendering_still_to_be_checked':True},indent=2))


if __name__=='__main__':main()
