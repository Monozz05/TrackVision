"""Run the pinned, unmodified TrackEval on a prepared MOTChallenge layout and write per-sequence results.

Usage: <trackeval python> -I scripts/trackeval_mot17.py CONFIG.json

Runs in the isolated TrackEval environment (Python 3.11, numpy 1.23.5, scipy 1.10.1), never in the project
.venv, and imports nothing from trackvision. TrackEval is imported from its source directory (CONFIG
"trackeval_source"); no bytecode is written into that tree. CONFIG holds "dataset_config", "eval_config"
and "metrics" for TrackEval's own constructors, and "results_file" for the output. Results are taken from
the dictionary Evaluator.evaluate returns; the COMBINED_SEQ entry is left out. Standard library and the
TrackEval environment only; this is repository tooling, not research code.
"""

import json
import platform
import sys


def native(value):
    """numpy scalars and arrays to JSON-ready Python values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main(config_path):
    with open(config_path, encoding="utf-8") as fp:
        cfg = json.load(fp)
    sys.dont_write_bytecode = True
    sys.path.insert(0, cfg["trackeval_source"])
    import numpy
    import scipy
    import trackeval

    # Count is added by Evaluator.evaluate itself (eval.py:58), so it is never listed here.
    metric_classes = {"HOTA": trackeval.metrics.HOTA, "CLEAR": trackeval.metrics.CLEAR,
                      "Identity": trackeval.metrics.Identity}
    metrics = [metric_classes[name](options) if options is not None else metric_classes[name]()
               for name, options in cfg["metrics"]]
    evaluator = trackeval.Evaluator(cfg["eval_config"])
    dataset = trackeval.datasets.MotChallenge2DBox(cfg["dataset_config"])
    output_res, output_msg = evaluator.evaluate([dataset], metrics)

    dataset_name = dataset.get_name()
    sequences = {}
    for tracker, res in output_res[dataset_name].items():
        if res is None:
            continue
        for seq, by_class in res.items():
            if seq == "COMBINED_SEQ":
                continue
            sequences[seq] = {metric: {field: native(v) for field, v in fields.items()}
                              for metric, fields in by_class["pedestrian"].items()}
    out = {
        "messages": output_msg[dataset_name],
        "dataset_name": dataset_name,
        "sequences": sequences,
        "hota_alphas": native(metrics[[n for n, _ in cfg["metrics"]].index("HOTA")].array_labels),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "executable": sys.executable, "numpy": numpy.__version__, "scipy": scipy.__version__,
                        "trackeval_file": trackeval.__file__},
    }
    with open(cfg["results_file"], "w", encoding="utf-8", newline="\n") as fp:
        json.dump(out, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return 0 if all(m == "Success" for m in output_msg[dataset_name].values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
