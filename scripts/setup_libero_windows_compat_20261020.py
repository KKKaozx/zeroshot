"""Owned venv only: stock Windows DLL lookup and supported renderer macros."""
import hashlib
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
prefix = ROOT / 'training_cache/libero_native_env_v1'
assert Path(sys.prefix).resolve() == prefix.resolve()
packages = prefix / 'Lib/site-packages'
source = packages / 'mujoco/mujoco.dll'
target = packages / 'robosuite/utils/mujoco.dll'
assert source.is_file()
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if not target.exists():
    shutil.copyfile(source, target)
assert digest(source) == digest(target)
private = packages / 'robosuite/macros_private.py'
text = '''# Workspace LIBERO environment only, renderer/log portability, no dynamics changes.
import robosuite.macros as _workspace_macros
_workspace_macros.MUJOCO_GPU_RENDERING = False
_workspace_macros.FILE_LOGGING_LEVEL = None
'''
if private.exists():
    assert private.read_text() == text
else:
    private.write_text(text)
print('Windows renderer macros and byte-identical MuJoCo DLL lookup prepared')
