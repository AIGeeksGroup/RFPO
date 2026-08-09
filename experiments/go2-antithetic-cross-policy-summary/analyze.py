"""Build the retrospective H76 policy-level statistical summary."""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy import stats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path, hypothesis):
    result = json.loads(path.read_text())
    if result.get("hypothesis") != hypothesis or result.get("passed") is not True:
        raise ValueError(f"invalid {hypothesis} artifact: {path}")
    return result


def summarize(seed_effects):
    values = np.asarray(list(seed_effects.values()), dtype=np.float64)
    n = values.size
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = float(sd / np.sqrt(n))
    critical = float(stats.t.ppf(0.975, df=n - 1))
    t_statistic = float(mean / sem)
    two_sided_p = float(2.0 * stats.t.sf(abs(t_statistic), df=n - 1))
    positive_count = int((values > 0.0).sum())
    sign_p = float(
        stats.binomtest(positive_count, n=n, p=0.5, alternative="greater").pvalue
    )
    return {
        "seed_effects": seed_effects,
        "n_training_seeds": n,
        "mean": mean,
        "sample_sd": sd,
        "sem": sem,
        "student_t_95": [mean - critical * sem, mean + critical * sem],
        "t_statistic": t_statistic,
        "degrees_of_freedom": n - 1,
        "two_sided_t_p": two_sided_p,
        "positive_seed_count": positive_count,
        "exact_one_sided_sign_p": sign_p,
    }


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    root = args.project_root
    h67 = load_result(
        root / "experiments/go2-antithetic-attribution/results/analysis.json", "H67"
    )
    h69 = load_result(
        root / "experiments/go2-antithetic-independent-seeds/results/analysis.json",
        "H69",
    )
    h70 = load_result(
        root / "experiments/go2-antithetic-equal-total-nfe/results/analysis.json",
        "H70",
    )
    h72 = json.loads(
        (
            root
            / "experiments/go2-antithetic-reduced-zero-control/results/analysis.json"
        ).read_text()
    )
    if h72.get("hypothesis") != "H72" or h72.get("passed") is not False:
        raise ValueError("invalid H72 artifact")
    h73 = load_result(
        root
        / "experiments/go2-antithetic-independent-equal-total-nfe/results/analysis.json",
        "H73",
    )

    equal_total_nfe = {
        "42": h70["comparisons"]["antithetic32_minus_random64"]["gain"],
        **{
            seed: record["comparisons"]["antithetic32_minus_random64"]["gain"]
            for seed, record in h73["seed_results"].items()
        },
    }
    equal_pair_nfe = {
        "42": h67["comparisons"]["antithetic_minus_iid_pair"]["gain"],
        **{
            seed: record["comparisons"]["antithetic_minus_iid_pair"]["gain"]
            for seed, record in h69["seed_results"].items()
        },
    }
    deployment = {
        "42": h72["comparisons"]["antithetic32_minus_zero32"]["gain"],
        **{
            seed: record["comparisons"]["antithetic32_minus_zero32"]["gain"]
            for seed, record in h73["seed_results"].items()
        },
    }
    result = {
        "analysis": "H76",
        "status": "retrospective_no_gate",
        "independent_unit": "official_training_seed",
        "training_seeds": [42, 43, 44, 45],
        "estimands": {
            "antithetic32_minus_random64_equal_total_nfe": summarize(equal_total_nfe),
            "antithetic64_minus_iid_pair64_equal_pair_nfe": summarize(equal_pair_nfe),
            "antithetic32_minus_zero32_deployment_control": summarize(deployment),
        },
        "software": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
