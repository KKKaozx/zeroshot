"""Import the actual training/evaluation dependencies on CPU before sbatch.

No model construction, downloads, feature encoding, sampling or training.
The full run_pair --check-only also validates the frozen dataset/runtime.
"""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'code'), str(HERE/'vendor/code'), str(HERE/'vendor')]


def runtime_dependencies():
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0')
    import numpy as np
    import torch
    import PIL
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer, Dinov2Model
    from safetensors import safe_open
    import train
    import models
    import bcap_core
    import bcap_runtime
    import bridge_timestep_repair
    import verify_results
    from oxe_nominal_dataset import NominalOXEDataset
    from oxe_nominal_target_store import NominalTargetStore
    from frozen_visual_backbone import FrozenDinoVision
    from pose_features import batch_features
    assert Path(train.__file__).resolve() == HERE/'vendor/code/train.py'
    assert Path(models.__file__).resolve() == HERE/'vendor/code/models.py'
    assert Path(bcap_runtime.__file__).resolve() == HERE/'vendor/bcap_runtime.py'
    if torch.cuda.is_initialized() or tf.config.get_visible_devices('GPU'):
        raise RuntimeError('CPU admission must not initialize a GPU training context')
    return {'numpy': np.__version__, 'torch': str(torch.__version__),
            'tensorflow': tf.__version__, 'PIL': PIL.__version__,
            'complete_training_and_saved_result_imports_passed': True,
            'model_constructed': False, 'cuda_initialized': False,
            'optimizer_updates': 0,
            'target_runtime_still_requires_full_run_pair_check_only': True}


if __name__ == '__main__':
    print(json.dumps(runtime_dependencies(), indent=2))
