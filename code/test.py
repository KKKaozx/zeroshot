"""Main offline evaluation entry; see docs/RUNNING.md for split precautions."""

from evaluate_offline import evaluate, parse_args


if __name__ == "__main__":
    evaluate(parse_args())
