# -*- coding: utf-8 -*-
import logging
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scripts._helpers import configure_logging, mock_snakemake


logger = logging.getLogger(__name__)

scenario_dict = {
    "no_co2_network": "No CO\u2082 network",
    "frozen_H2_28": "H\u2082 + CO\u2082 (PCI 2029) frozen",
    "endo_H2": "H\u2082 + CO\u2082 endogenous",
    "pcipmi_H2_+": "H\u2082 + CO\u2082 (PCI 2035)",
    "onshore_sequestration": "Onshore Seq",
    "no_early_retirement": "No early BF-BOF retirement",
    "seq_50": "Low Seq Potential",
    "northern_lights": "No North Sea Seq",
    "onshore_sequestration_endo": "Onshore Seq endogenous",
    "no_north_sea_endo": "No North Sea endogenous",
}


def plot_co2_balance(
        scenarios: list,
        files_DE: list,
        files_EU: list,
        save_dir: str,
)->None:
    """
    Plotting Carbon Balances for Germany and Europe.
    """
    # load and organize data
    logger.info("Loading data for carbon balances...")
    CO2_DE = pd.DataFrame(columns=scenarios)
    for i, file in enumerate(files_DE):
        DE = pd.read_csv(file, index_col=0)

        all_idx = CO2_DE.index.union(DE.index)
        CO2_DE = CO2_DE.reindex(all_idx)
        CO2_DE.loc[DE.index, scenarios[i]] = DE["0"]

    CO2_EU = pd.DataFrame(columns=scenarios)
    for i, file in enumerate(files_EU):
        EU = pd.read_csv(file, index_col=0)

        all_idx = CO2_EU.index.union(EU.index)
        CO2_EU = CO2_EU.reindex(all_idx)
        CO2_EU.loc[EU.index, scenarios[i]] = EU["0"]

    # initialize plot
    logger.info("Plotting German carbon balance...")
    fig, ax = plt.subplots(1, 1, figsize=(3, 5))

    CO2_DE.T.plot.bar(
        ax=ax,
        color=[tech_colors[col] for col in CO2_DE.index],
        stacked=True,
        width=0.8,
        legend=False)

    totals = CO2_DE.clip(lower=0).sum(axis=0)

    # Annotate each bar with its total value
    for i, total in enumerate(totals):
        ax.text(
            i, total + 1,                # (x, y) position
            f"{total:.1f}",              # label text
            ha='center', va='bottom',
            fontsize=9, color='black'
        )
    ax.set_ylim([-max(totals)-2, max(totals)+5])
    ax.set_ylabel("Mt/a")
    ax.set_xticklabels([scenario_dict[scenario] for scenario in scenarios])
    ax.set_title(f"Germany Carbon Balance 2035")
    ax.set_facecolor('white')
    ax.grid(False)
    ax.axhline(0, color='black', linewidth=1.2)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(1.55, 1),
        ncol=1,
    )

    fig.savefig(save_dir + f"/co2_stored_balance_DE_{year}.pdf", bbox_inches="tight")
    plt.close()

    logger.info("Plotting European carbon balance...")
    fig, ax = plt.subplots(1, 1, figsize=(3, 5))

    CO2_EU.T.plot.bar(
        ax=ax,
        color=[tech_colors[col] for col in CO2_EU.index],
        stacked=True,
        width=0.8,
        legend=False)

    totals = CO2_EU.clip(lower=0).sum(axis=0)

    # Annotate each bar with its total value
    for i, total in enumerate(totals):
        ax.text(
            i, total + 1,                # (x, y) position
            f"{total:.1f}",              # label text
            ha='center', va='bottom',
            fontsize=9, color='black'
        )
    ax.set_ylim([-max(totals)-2, max(totals)+15])
    ax.set_ylabel("Mt/a")
    ax.set_xticklabels([scenario_dict[scenario] for scenario in scenarios])
    ax.set_title(f"Europe Carbon Balance 2035")
    ax.set_facecolor('white')
    ax.grid(False)
    ax.axhline(0, color='black', linewidth=1.2)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(1.55, 1),
        ncol=1,
    )

    fig.savefig(save_dir + f"/co2_stored_balance_EU_{year}.pdf", bbox_inches="tight")
    plt.close()


def plot_h2_balance(
        scenarios: list,
        files_DE: list,
        files_EU: list,
        save_dir: str,
)->None:
    """
    Plotting Hydrogen Balances for Germany and Europe.
    """
    logger.info("Loading data for hydrogen balances...")
    H2_DE = pd.DataFrame(columns=scenarios)
    H2_EU = pd.DataFrame(columns=scenarios)

    for i in range(len(scenarios)):
        DE = pd.read_csv(files_DE[i], index_col=0)

        EU = pd.read_csv(files_EU[i], index_col=0)

        all_idx = H2_DE.index.union(DE.index)
        H2_DE = H2_DE.reindex(all_idx)
        H2_DE.loc[DE.index, scenarios[i]] = DE["0"]
        all_idx = H2_EU.index.union(EU.index)
        H2_EU = H2_EU.reindex(all_idx)
        H2_EU.loc[EU.index, scenarios[i]] = EU["0"]
    
    logger.info("Plotting hydrogen balances...")
    fig, ax = plt.subplots(2, 1, figsize=(14, 9))

    H2_DE.T.plot.barh(ax=ax[0], color=[tech_colors[col] for col in H2_DE.index], stacked=True, width=0.8, legend=False)
    ax[0].set_xlabel("TWh")
    ax[0].set_yticklabels([scenario_dict[scenario] for scenario in scenarios])
    ax[0].set_title("Germany")

    H2_EU.T.plot.barh(ax=ax[1], color=[tech_colors[col] for col in H2_DE.index], stacked=True, width=0.8, legend=False)
    ax[1].set_xlabel("TWh")
    ax[1].set_yticklabels([scenario_dict[scenario] for scenario in scenarios])
    ax[1].set_title("Europe")

    handles, labels = ax[0].get_legend_handles_labels()

    supply_handles = handles[:6]
    supply_labels = labels[:6]
    demand_handles = handles[6:]
    demand_labels = labels[6:]
    subtitle_supply = Patch(color="none", label="Supply")
    subtitle_demand = Patch(color="none", label="Demand")
    # Combine all handles and labels
    combined_handles = (
        [subtitle_supply]
        + supply_handles
        + [subtitle_demand]
        + demand_handles
    )
    combined_labels = (
        ["Supply"] + supply_labels + ["Demand"] + demand_labels
    )

    ax[0].legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=(1.15, 1.0),
    )
    fig.suptitle(2035)
    
    fig.savefig(save_dir + f"/h2_balance_{year}.pdf", bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_balances",
            simpl="",
            clusters=89,
            planning_horizons=2035,
            opts="",
            ll="vopt",
            sector_opts="none",
            run="endo_H2"
        )

    configure_logging(snakemake)

    tech_colors = snakemake.params.plotting["tech_colors"]
    tech_colors["trade"] = '#f0d071'
    tech_colors["export"] = tech_colors["import H2"]
    tech_colors["import"] = tech_colors["import H2"]

    year = 2045
    path = snakemake.input.h2_DE[0]
    save_dir = path.split("results/")[0] + "results/" + path.split("results/")[1].split("/")[0]

    plot_co2_balance(
        scenarios=snakemake.params.scenarios,
        files_DE=snakemake.input.co2_stored_DE,
        files_EU=snakemake.input.co2_stored_EU,
        save_dir=save_dir,
        )
    
    plot_h2_balance(
        scenarios=snakemake.params.scenarios,
        files_DE=snakemake.input.h2_DE,
        files_EU=snakemake.input.h2_EU,
        save_dir=save_dir,
        )
