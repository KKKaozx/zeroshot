"""GPU entry that deliberately does not import the CPU physics collector."""
import argparse
from pathlib import Path

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True,help='New, nonexistent output directory')
    args=ap.parse_args()
    import preflight_panda_minimal_generalization_20261015 as gpu
    gpu.DATA=root/'reports/panda_spatial_control_v2_preparation_20261015'
    gpu.OUT=args.output.resolve()
    gpu.main()
