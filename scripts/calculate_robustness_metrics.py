# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Collects the data from all networks and calculates the robustness metrics for the CO2 pipeline topologies.

This script can be used standalone or within the snakemake workflow. It is called by the rule :mod:`calculate_robustness_metrics` in the snakemake workflow.

Description
-----------

Reads in all the topology informations from results("co2_topologies/*") and calculates the selection frequency, the utilization frequency, expected trhoughput, variability/fragility before drawing a total score for each pipeline. The results are written out to a csv file.

"""

import importlib
import logging
import os
import pathlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import yaml
from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)



if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "calculate_robustness_metrics",
            configfiles="config/config.de.yaml",
            run="endogenous",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    # get all edges

    # get number of subdirectories in results("co2_topologies/*")
    base_dir = pathlib.Path(snakemake.input[0])
    base_dir = pathlib.Path("/home/toni-seibold/dev/pypsa-de-co2/results/cluster/topologies_2190SEG/endogenous/co2_topologies")
    subdirectories = [d for d in base_dir.iterdir() if d.is_dir()]
    num_subdirectories = len(subdirectories)

    # read in metrics from every subdirectory existing in results("co2_topologies/*")
    topology_files = [d / "base_s_89__none_topology.csv" for d in subdirectories]

    all_topologies = []
    for topology_file in topology_files:
            topology = pd.read_csv(topology_file)
            all_topologies.append(topology)
    all_topologies = pd.concat(all_topologies, ignore_index=True)

    # group by bus0 and bus1 and calculate the selection frequency, utilization frequency, expected throughput, variability/fragility and total score for each pipeline
    all_topologies["selection_frequency"] = all_topologies.groupby(["bus0", "bus1"])["bus0"].transform("count") / num_subdirectories

    all_topologies["utilization_frequency"] = all_topologies.groupby(["bus0", "bus1"])["utilization"].transform(lambda x: (x > 0.1).sum() / len(x))

    all_topologies["expected_throughput"] = all_topologies.groupby(["bus0", "bus1"])["throughput"].transform("mean")

    # variability
    all_topologies["variability"] = all_topologies.groupby(["bus0", "bus1"])["throughput"].transform("std")

    all_topologies["cv"] = (
        all_topologies["variability"] / all_topologies["expected_throughput"]
    )
    all_topologies["variability_penalty"] = 1 / (1 + all_topologies["cv"])
    # total score
    all_topologies["total_score"] = (
        all_topologies["selection_frequency"]
        * all_topologies["utilization_frequency"]
        * all_topologies["variability_penalty"]
    )

    # plot the map of the pipelines with the total score as color
    regions_path = pathlib.Path("resources/03_17_run/endogenous/regions_onshore_base_s_89.geojson")
    regions = gpd.read_file(regions_path).set_index("name").to_crs("EPSG:3035")
    regions = regions[regions.index.str.contains("DE") | regions.index.str.contains("NL")]  # Filter to German and Austrian regions only for better visualization
    centroids = regions.geometry.centroid
    coords = pd.DataFrame({"x": centroids.x, "y": centroids.y}, index=regions.index.astype(str))
    fig, ax = plt.subplots(figsize=(10, 10))
    regions.plot(ax=ax, color="white", edgecolor="grey")
    data = all_topologies.groupby(["bus0", "bus1"]).total_score.mean().fillna(0)
    for bus0, bus1 in data.index:
        if data.loc[(bus0, bus1)] <= 0.0:  # Filter out pipelines with very low scores for better visualization
            continue
        # Get the coordinates of the buses (you need to have a mapping of bus names to coordinates)
        x0, y0 = coords.loc[bus0]
        x1, y1 = coords.loc[bus1]
        ax.plot([x0, x1], [y0, y1], color=plt.cm.viridis(data.loc[(bus0, bus1)]), linewidth=2)
    plt.colorbar(ax=ax, mappable=plt.cm.ScalarMappable(cmap="viridis"), label="Total Score")
    plt.title("Pipeline Robustness Score > 0.4")
    # remove x and y ticks
    plt.xticks([])
    plt.yticks([])
    plt.show()
    fig.savefig("robustness_score_selected.pdf", dpi=300, bbox_inches="tight")


    all_topologies.to_csv(snakemake.output[0], index=False)



