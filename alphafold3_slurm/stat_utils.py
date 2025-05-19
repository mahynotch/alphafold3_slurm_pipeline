import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from itertools import product
from tqdm import tqdm


def plot_confidence_boxplot(df: pl.DataFrame, save_path: str):
    # Prepare data for pLDDT plotting
    df = df.drop_nans()
    df = df.drop_nulls()
    plddt_data = pl.DataFrame(
        {"Score": pl.concat([df["pLDDT"]]), "Metric": ["pLDDT"] * (len(df))}
    )

    # Prepare data for pTM plotting
    ptm_data = pl.DataFrame(
        {"Score": pl.concat([df["pTM"]]), "Metric": ["pTM"] * (len(df))}
    )

    # Prepare data for ipTM plotting
    iptm_data = pl.DataFrame(
        {"Score": pl.concat([df["ipTM"]]), "Metric": ["ipTM"] * (len(df))}
    )
    plot_data = pl.concat([plddt_data, ptm_data, iptm_data])

    # Set style
    colors = ["#2ecc71"]  # Green for AF2, Red for AF3

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create box plot
    bp = sns.boxplot(
        data=plot_data, x="Metric", y="Score", palette=colors, width=0.7, linewidth=2
    )

    # Add individual points with jitter
    sns.stripplot(
        data=plot_data,
        x="Metric",
        y="Score",
        dodge=True,
        size=4,
        alpha=0.3,
        palette=colors,
        jitter=0.2,
    )

    # Customize the plot
    plt.title("Confidence Metrics", pad=20, fontsize=16, fontweight="bold")

    plt.xlabel("Confidence Metric", fontsize=12, fontweight="bold")
    plt.ylabel("Score", fontsize=12, fontweight="bold")

    # Set y-axis limits from 0 to 100
    plt.ylim(0, 1)

    # Add grid
    plt.grid(True, linestyle="--", alpha=0.7)

    # Customize spines
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Add statistics as text
    stats_text = (
        f"Median Scores:\n"
        f"pLDDT:\n"
        f"{df['pLDDT'].median() if df['pLDDT'].median() is not None else 0.00:.2f}\n"
        f"pTM:\n"
        f"{df['pTM'].median() if df['pTM'].median() is not None else 0.00:.2f}\n"
        f"ipTM:\n"
        f"{df['ipTM'].median() if df['ipTM'].median() is not None else 0.00:.2f}\n\n"
        f"Mean Scores:\n"
        f"pLDDT:\n"
        f"{df['pLDDT'].mean() if df['pLDDT'].mean() is not None else 0.00:.2f} ± {df['pLDDT'].std() if df['pLDDT'].std() is not None else 0.00:.2f}\n"
        f"pTM:\n"
        f"{df['pTM'].mean() if df['pTM'].mean() is not None else 0.00:.2f} ± {df['pTM'].std() if df['pTM'].std() is not None else 0.00:.2f}\n"
        f"ipTM:\n"
        f"{df['ipTM'].mean() if df['ipTM'].mean() is not None else 0.00:.2f} ± {df['ipTM'].std() if df['ipTM'].std() is not None else 0.00:.2f}\n\n"
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

    # Adjust layout
    plt.tight_layout()

    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def collect_statistics(name_set: set[list], complex_dir):
    """
    Collect statistics for a set of molecule names
    :param name_set: a set containing lists of molecule names, e.g. (bait_list, prey_list)
    :param complex_dir: path to the directory containing the complex folders
    """
    name_combinations = list(product(*name_set))
    name_list = list(map(lambda x: "-".join(x), name_combinations))
    df = _collect_statistics(name_list, complex_dir)
    return df

def special_join(combination):
    """
    Join a list of strings with a special separator
    :param name_set: a set containing lists of molecule names, e.g. (bait_list, prey_list)
    :return: a string with the joined names
    """
    return "-".join([x if x is not None else "" for x in combination])

def collect_statistics_exact(name_set: set[list], complex_dir):
    """
    Collect statistics for a set of molecule names, for exact input
    :param name_set: a set containing lists of molecule names, e.g. (bait_list, prey_list)
    :param complex_dir: path to the directory containing the complex folders
    """
    name_combinations = list(zip(*name_set))
    name_list = list(map(special_join, name_combinations))
    df = _collect_statistics(name_list, complex_dir)
    return df


def _collect_statistics(name_list, complex_dir):
    results = []
    for name in tqdm(name_list):
        result = {"name": name, "pTM": np.nan, "chainPAE": np.nan, "pLDDT": np.nan, "ipTM": np.nan}
        # Find matching folder
        matching_folder = os.path.join(complex_dir, name)

        if matching_folder and os.path.exists(matching_folder):
            folder = matching_folder
            basename = os.path.basename(folder)
            json_file = os.path.join(
                folder, f"{basename}_summary_confidences.json"
            )

            if os.path.exists(json_file):
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        result["pTM"] = data["ptm"]
                        result["ipTM"] = data["iptm"]
                        if len(data["chain_pair_pae_min"]) > 1:
                            result["chainPAE"] = data["chain_pair_pae_min"][0][1]
                except:
                    pass

            atom_met_file = os.path.join(
                folder, f"{basename}_confidences.json"
            )
            if os.path.exists(atom_met_file):
                try:
                    with open(atom_met_file, "r") as f:
                        data = json.load(f)
                        atom_plddts = data["atom_plddts"]
                        average_plddt = sum(atom_plddts) / (len(atom_plddts) * 100)
                        result["pLDDT"] = average_plddt
                except:
                    pass

        molecule_names = name.split("-")
        if len(molecule_names) > 1:
            for i, key in enumerate(molecule_names):
                result[f"name_molecule{i}"] = key
        results.append(result)
    df = pl.DataFrame(results)
    return df
