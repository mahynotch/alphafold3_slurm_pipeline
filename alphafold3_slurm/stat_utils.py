import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from tqdm import tqdm


def plot_confidence_boxplot(df: pl.DataFrame, save_path: str | Path) -> None:
    df = df.drop_nans().drop_nulls()
    save_path = Path(save_path)

    if df.is_empty():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No valid confidence scores found", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    plddt_data = pl.DataFrame({"Score": df["pLDDT"], "Metric": ["pLDDT"] * len(df)})
    ptm_data = pl.DataFrame({"Score": df["pTM"], "Metric": ["pTM"] * len(df)})
    iptm_data = pl.DataFrame({"Score": df["ipTM"], "Metric": ["ipTM"] * len(df)})
    plot_data = pl.concat([plddt_data, ptm_data, iptm_data])

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(
        data=plot_data,
        x="Metric",
        y="Score",
        color="#2ecc71",
        width=0.7,
        linewidth=2,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="Metric",
        y="Score",
        dodge=True,
        size=4,
        alpha=0.3,
        color="#1f1f1f",
        jitter=0.2,
        ax=ax,
    )

    plt.title("Confidence Metrics", pad=20, fontsize=16, fontweight="bold")
    plt.xlabel("Confidence Metric", fontsize=12, fontweight="bold")
    plt.ylabel("Score", fontsize=12, fontweight="bold")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.7)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    stats_text = (
        "Median Scores:\n"
        f"pLDDT:\n{_format_number(df['pLDDT'].median())}\n"
        f"pTM:\n{_format_number(df['pTM'].median())}\n"
        f"ipTM:\n{_format_number(df['ipTM'].median())}\n\n"
        "Mean Scores:\n"
        f"pLDDT:\n{_format_mean_std(df['pLDDT'].mean(), df['pLDDT'].std())}\n"
        f"pTM:\n{_format_mean_std(df['pTM'].mean(), df['pTM'].std())}\n"
        f"ipTM:\n{_format_mean_std(df['ipTM'].mean(), df['ipTM'].std())}\n"
    )
    plt.text(
        1.15,
        0.95,
        stats_text,
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
        fontsize=10,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_number(value: float | None) -> str:
    return f"{0.0 if value is None else value:.2f}"


def _format_mean_std(mean: float | None, std: float | None) -> str:
    safe_mean = 0.0 if mean is None else mean
    safe_std = 0.0 if std is None else std
    return f"{safe_mean:.2f} ± {safe_std:.2f}"


def collect_statistics(name_set: list[list[str]] | tuple[list[str], ...], complex_dir: str | Path) -> pl.DataFrame:
    name_list = ["-".join(combination) for combination in product(*name_set)]
    return _collect_statistics(name_list, complex_dir)


def special_join(combination: tuple[str | None, ...]) -> str:
    return "-".join([value if value is not None else "" for value in combination])


def collect_statistics_exact(
    name_set: list[list[str]] | tuple[list[str], ...],
    complex_dir: str | Path,
) -> pl.DataFrame:
    name_list = [special_join(combination) for combination in zip(*name_set)]
    return _collect_statistics(name_list, complex_dir)


def _collect_statistics(name_list: list[str], complex_dir: str | Path) -> pl.DataFrame:
    complex_root = Path(complex_dir)
    results = []
    for name in tqdm(name_list):
        result = {
            "name": name,
            "pTM": np.nan,
            "chainPAE": np.nan,
            "pLDDT": np.nan,
            "ipTM": np.nan,
        }
        folder = complex_root / name
        if folder.exists():
            summary_path = folder / f"{folder.name}_summary_confidences.json"
            if summary_path.exists():
                summary_data = _load_json(summary_path)
                if summary_data is not None:
                    result["pTM"] = summary_data.get("ptm", np.nan)
                    result["ipTM"] = summary_data.get("iptm", np.nan)
                    chain_pair_pae_min = summary_data.get("chain_pair_pae_min")
                    if isinstance(chain_pair_pae_min, list) and len(chain_pair_pae_min) > 1:
                        result["chainPAE"] = chain_pair_pae_min[0][1]

            atom_metrics_path = folder / f"{folder.name}_confidences.json"
            if atom_metrics_path.exists():
                atom_metrics = _load_json(atom_metrics_path)
                if atom_metrics is not None:
                    atom_plddts = atom_metrics.get("atom_plddts")
                    if atom_plddts:
                        result["pLDDT"] = sum(atom_plddts) / (len(atom_plddts) * 100)
                    else:
                        print(f"Warning: atom_plddts missing or empty in {atom_metrics_path}")

        molecule_names = name.split("-")
        if len(molecule_names) > 1:
            for index, key in enumerate(molecule_names):
                result[f"name_molecule{index}"] = key
        results.append(result)

    return pl.DataFrame(results)


def _load_json(path: Path) -> dict | list | None:
    try:
        with path.open("r") as handle:
            return json.load(handle)
    except FileNotFoundError:
        print(f"Warning: expected file not found: {path}")
    except json.JSONDecodeError as error:
        print(f"Warning: could not parse JSON {path}: {error}")
    except OSError as error:
        print(f"Warning: could not read {path}: {error}")
    return None
