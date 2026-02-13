#!/usr/bin/env python3
"""
Subcluster analysis for concept mapping.

Runs hierarchical clustering within each main cluster to identify subclusters,
then computes subcluster-level importance/feasibility means and generates figures.

Requires that the main PyConceptMap analysis has been run first (output folder
with processed_data/ and statement summary).

Usage (from project root):
    python scripts/run_subcluster_analysis.py --output_folder ./output
    python scripts/run_subcluster_analysis.py --output_folder ./output --n_subclusters 4
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def load_outputs(output_folder: Path):
    """Load MDS coordinates, cluster assignments, and statement ratings from pipeline output."""
    output_folder = Path(output_folder)
    processed = output_folder / "processed_data"
    if not processed.exists():
        raise FileNotFoundError(f"Processed data not found: {processed}. Run the main analysis first.")

    mds = pd.read_csv(processed / "mds_coordinates.csv")
    if "StatementID" not in mds.columns:
        raise ValueError("mds_coordinates.csv must contain StatementID column.")

    # Cluster assignment: use statements_by_cluster if present, else infer from cluster_summary
    statements_by_cluster_path = output_folder / "statements_by_cluster.csv"
    if statements_by_cluster_path.exists():
        by_cluster = pd.read_csv(statements_by_cluster_path)
        stmt_cluster = by_cluster[["StatementID", "Cluster"]].drop_duplicates()
    else:
        # Fallback: cluster_summary has cluster indices; we need statement-level from somewhere
        # Try Table_All_Statements_by_Cluster
        table_path = output_folder / "Table_All_Statements_by_Cluster.csv"
        if not table_path.exists():
            raise FileNotFoundError(
                "Need statements_by_cluster.csv or Table_All_Statements_by_Cluster.csv. Run main analysis first."
            )
        table = pd.read_csv(table_path)
        id_col = "Statement ID" if "Statement ID" in table.columns else "StatementID"
        stmt_cluster = table[[id_col, "Cluster"]].rename(columns={id_col: "StatementID"})
        if stmt_cluster["Cluster"].dtype == object or str(stmt_cluster["Cluster"].iloc[0]).startswith("Cluster"):
            stmt_cluster["Cluster"] = stmt_cluster["Cluster"].astype(str).str.replace("Cluster ", "").astype(int)

    # Statement-level ratings (importance/feasibility means)
    for name in ["StatementSummary02.csv", "statement_summary_02.txt"]:
        summary_path = output_folder / name
        if summary_path.suffix == ".csv" and summary_path.exists():
            summary = pd.read_csv(summary_path)
            break
    else:
        # Try any StatementSummary*.csv
        summaries = list(output_folder.glob("StatementSummary*.csv"))
        if not summaries:
            raise FileNotFoundError("No StatementSummary CSV found in output folder.")
        summary = pd.read_csv(summaries[0])

    imp_col = "Importance_mean" if "Importance_mean" in summary.columns else "Importance Mean"
    feas_col = "Feasibility_mean" if "Feasibility_mean" in summary.columns else "Feasibility Mean"
    if imp_col not in summary.columns or feas_col not in summary.columns:
        raise ValueError("Statement summary must have importance and feasibility mean columns.")
    summary = summary.rename(columns={imp_col: "Importance_mean", feas_col: "Feasibility_mean"})

    return mds, stmt_cluster, summary[["StatementID", "Importance_mean", "Feasibility_mean"]]


def run_subclustering(
    mds: pd.DataFrame,
    stmt_cluster: pd.DataFrame,
    n_subclusters: int = 4,
) -> pd.DataFrame:
    """Assign subcluster labels within each main cluster using Ward clustering on MDS coords."""
    dim_cols = [c for c in mds.columns if c.startswith("Dimension") or c in ("Dimension1", "Dimension2", "x", "y")]
    if not dim_cols:
        dim_cols = [c for c in mds.columns if c != "StatementID" and np.issubdtype(mds[c].dtype, np.number)]
    if len(dim_cols) < 2:
        raise ValueError("MDS coordinates need at least 2 dimension columns.")

    merged = mds.merge(stmt_cluster, on="StatementID", how="inner")
    merged = merged.sort_values(["Cluster", "StatementID"]).reset_index(drop=True)

    subcluster_assign = []
    for cluster_id in sorted(merged["Cluster"].unique()):
        mask = merged["Cluster"] == cluster_id
        coords = merged.loc[mask, dim_cols].values
        n = coords.shape[0]
        k = min(n_subclusters, n)
        if k < 2:
            sublabels = np.zeros(n, dtype=int)
        else:
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            sublabels = model.fit_predict(coords)
        for i, (idx, row) in enumerate(merged.loc[mask].iterrows()):
            subcluster_assign.append((row["StatementID"], row["Cluster"], f"{int(row['Cluster'])}.{sublabels[i] + 1}"))

    return pd.DataFrame(subcluster_assign, columns=["StatementID", "Cluster", "Subcluster"])


def main():
    parser = argparse.ArgumentParser(description="Run subcluster analysis on concept mapping output.")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="./output",
        help="Path to output folder from main PyConceptMap run.",
    )
    parser.add_argument(
        "--n_subclusters",
        type=int,
        default=4,
        help="Number of subclusters per main cluster (default 4).",
    )
    args = parser.parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    figures_folder = output_folder / "figures"
    figures_folder.mkdir(exist_ok=True)
    processed_folder = output_folder / "processed_data"
    processed_folder.mkdir(exist_ok=True)

    print("Loading pipeline outputs...")
    mds, stmt_cluster, summary = load_outputs(output_folder)

    print("Running subclustering (Ward within each main cluster)...")
    subcluster_df = run_subclustering(mds, stmt_cluster, n_subclusters=args.n_subclusters)

    # Merge with ratings
    with_ratings = subcluster_df.merge(summary, on="StatementID", how="left")

    # Save statement-level subcluster assignment
    statements_with_subclusters_path = output_folder / "statements_with_subclusters.csv"
    with_ratings.to_csv(statements_with_subclusters_path, index=False)
    print(f"Saved {statements_with_subclusters_path}")

    # Subcluster-level summary
    subcluster_summary = (
        with_ratings.groupby("Subcluster", as_index=False)
        .agg(
            Importance_mean=("Importance_mean", "mean"),
            Feasibility_mean=("Feasibility_mean", "mean"),
            n_statements=("StatementID", "count"),
        )
        .round(4)
    )
    subcluster_summary["Cluster"] = subcluster_summary["Subcluster"].str.split(".").str[0].astype(int)
    subcluster_summary = subcluster_summary.sort_values(["Cluster", "Subcluster"]).reset_index(drop=True)
    subcluster_summary_path = output_folder / "subcluster_summary.csv"
    subcluster_summary.to_csv(subcluster_summary_path, index=False)
    print(f"Saved {subcluster_summary_path}")

    # Figures
    dim_cols = [c for c in mds.columns if c != "StatementID" and np.issubdtype(mds[c].dtype, np.number)][:2]
    merged = mds.merge(subcluster_df, on="StatementID")
    x = merged[dim_cols[0]].values
    y = merged[dim_cols[1]].values
    sublabels = merged["Subcluster"].values
    n_sub = len(np.unique(sublabels))
    try:
        cmap = matplotlib.colormaps.get_cmap("tab20")
    except AttributeError:
        cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(n_sub)]
    sub_to_idx = {s: i for i, s in enumerate(sorted(np.unique(sublabels)))}
    point_colors = [colors[sub_to_idx[s]] for s in sublabels]

    # 1. Subcluster point map
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    for sub in sorted(np.unique(sublabels)):
        mask = sublabels == sub
        ax1.scatter(x[mask], y[mask], label=sub, s=40, alpha=0.8)
    ax1.set_xlabel(dim_cols[0])
    ax1.set_ylabel(dim_cols[1])
    ax1.set_title("Subcluster map (MDS)")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", ncol=2, fontsize=8)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1.savefig(figures_folder / "subcluster_point_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {figures_folder / 'subcluster_point_map.png'}")

    # 2. Subcluster pattern match (importance vs feasibility by subcluster)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sc = subcluster_summary.sort_values("Subcluster")
    x_pos = np.arange(len(sc))
    w = 0.35
    ax2.bar(x_pos - w / 2, sc["Importance_mean"], width=w, label="Importance", color="steelblue", alpha=0.8)
    ax2.bar(x_pos + w / 2, sc["Feasibility_mean"], width=w, label="Feasibility", color="coral", alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(sc["Subcluster"])
    ax2.set_ylabel("Mean rating")
    ax2.set_xlabel("Subcluster")
    ax2.set_title("Subcluster pattern match: Importance vs Feasibility")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig2.savefig(figures_folder / "subcluster_pattern_match.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {figures_folder / 'subcluster_pattern_match.png'}")

    # 3. Subcluster go-zone (subcluster means in importance–feasibility space)
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    imp_mean = subcluster_summary["Importance_mean"].mean()
    feas_mean = subcluster_summary["Feasibility_mean"].mean()
    ax3.scatter(
        subcluster_summary["Feasibility_mean"],
        subcluster_summary["Importance_mean"],
        s=120,
        alpha=0.8,
        edgecolors="black",
        linewidths=1,
    )
    for _, row in subcluster_summary.iterrows():
        ax3.annotate(
            row["Subcluster"],
            (row["Feasibility_mean"], row["Importance_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            ha="left",
        )
    ax3.axhline(y=imp_mean, color="gray", linestyle="--", alpha=0.7)
    ax3.axvline(x=feas_mean, color="gray", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Feasibility (mean)")
    ax3.set_ylabel("Importance (mean)")
    ax3.set_title("Subcluster go-zone")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()
    fig3.savefig(figures_folder / "subcluster_go_zone.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)
    print(f"Saved {figures_folder / 'subcluster_go_zone.png'}")

    print("Subcluster analysis finished.")


if __name__ == "__main__":
    main()
