# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves the capacity and operations network.
Then sweeps across different post_discretization thresholds

"""

import importlib
import logging
import os
import pathlib
import re
import sys
from functools import partial
from typing import Any

import linopy
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import yaml
from pypsa.descriptors import get_activity_mask
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa import discretized
from scripts._benchmark import memory_logger
from scripts._helpers import (
    PYPSA_V1,
    configure_logging,
    get,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.solve_network import solve_network, prepare_network

logger = logging.getLogger(__name__)

# Allow for PyPSA versions <0.35
if PYPSA_V1:
    pypsa.network.power_flow.logger.setLevel(logging.WARNING)
else:
    pypsa.pf.logger.setLevel(logging.WARNING)


class ObjectiveValueError(Exception):
    pass


def discretized_capacity(
    nom_opt: float,
    nom_max: float,
    unit_size: float,
    threshold: float,
    fractional_last_unit_size: bool,
) -> float:
    
    units = nom_opt // unit_size + (nom_opt % unit_size >= threshold * unit_size)

    block_capacity = units * unit_size
    if nom_max % unit_size == 0:
        return block_capacity

    if (nom_max - nom_opt) < unit_size:
        if (
            fractional_last_unit_size
            and ((nom_opt % unit_size) / (nom_max % unit_size)) >= threshold
        ):
            return nom_max
        if nom_max < unit_size:
            return nom_max
        return (nom_opt // unit_size) * unit_size
    return block_capacity


def discretize_infrastructure(n):
    fractional_last_unit_size = snakemake.params.solving["options"]["post_discretization"]["fractional_last_unit_size"]
    # take care of h2 pipelines
    unit_size = snakemake.params.solving["options"]["post_discretization"]["link_unit_size"]["H2 pipeline"]
    threshold = snakemake.params.solving["options"]["post_discretization"]["link_threshold"]["H2 pipeline"]
    h2_pipes = n.links[
        (n.links.carrier.str.contains("H2 pipeline")) & 
        (n.links.p_nom_extendable)].index
    logger.info(f"Discretizing {len(h2_pipes)} H2 pipelines")
    n.links.loc[h2_pipes, "p_nom"] = n.links.loc[h2_pipes].apply(
         lambda row: discretized_capacity(
                            nom_opt=row["p_nom_opt"],
                            nom_max=row["p_nom_max"],
                            unit_size=unit_size,
                            threshold=threshold,
                            fractional_last_unit_size=fractional_last_unit_size,
                        ),
                        axis=1,
                    )
    n.links.loc[h2_pipes, "p_nom_extenable"] = False
    logger.info(f"New p_nom of {n.links.loc[h2_pipes, "p_nom"].sort_values().unique()} MW")

    # take care of electricity links
    unit_size = snakemake.params.solving["options"]["post_discretization"]["link_unit_size"]["DC"]
    threshold = snakemake.params.solving["options"]["post_discretization"]["link_threshold"]["DC"]
    electricity = n.links[
        (n.links.carrier.str.contains("DC")) & 
        (n.links.p_nom_extendable)].index
    logger.info(f"Discretizing {len(electricity)} DC links")
    n.links.loc[electricity, "p_nom"] = n.links.loc[electricity].apply(
         lambda row: discretized_capacity(
                            nom_opt=row["p_nom_opt"],
                            nom_max=row["p_nom_max"],
                            unit_size=unit_size,
                            threshold=threshold,
                            fractional_last_unit_size=fractional_last_unit_size,
                        ),
                        axis=1,
                    )
    n.links.loc[electricity, "p_nom_extenable"] = False
    logger.info(f"New p_nom of {n.links.loc[electricity, "p_nom"].sort_values().unique()} MW")
    
    # take care of co2 pipelines
    diameter = snakemake.wildcards.diameter
    threshold = snakemake.wildcards.threshold
    # unit_size = dictionary
    unit_size=434 if diameter=="40" else 1328
    logger.info(f"Discretizing CO2 pipelines with {diameter} cm ({unit_size} t/h) and a threshold of {threshold}")
    co2_pipes = n.links[
        (n.links.carrier=="CO2 pipelines") & 
        (n.links.p_nom_extendable)].index
    logger.info(f"Discretizing {len(co2_pipes)} CO2 links")
    n.links.loc[co2_pipes, "p_nom"] = n.links.loc[co2_pipes].apply(
         lambda row: discretized_capacity(
                            nom_opt=row["p_nom_opt"],
                            nom_max=row["p_nom_max"],
                            unit_size=unit_size,
                            threshold=threshold,
                            fractional_last_unit_size=fractional_last_unit_size,
                        ),
                        axis=1,
                    )
    n.links.loc[co2_pipes, "p_nom_extenable"] = False
    logger.info(f"New p_nom of {n.links.loc[co2_pipes, "p_nom"].sort_values().unique()} t/h")



if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_sweep",
            opts="",
            clusters="89",
            configfiles="config/config.de.yaml",
            sector_opts="none",
            planning_horizons="2035",
            diameter=40,
            threshold=0.05,
            run="no_co2_network",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)
    planning_horizons = "2035"

    # change post_discretization config
    logger.info("Deactivate post discretization for sweep")
    snakemake.params.solving["options"]["skip_iterations"] = True
    snakemake.params.solving["options"]["post_discretization"]["enable"] = False
    # discrezization of pipeline infrastructure
    logger.info("Discretization of electricity, hydrogen and carbon infrastructure.")
    discretize_infrastructure(n)

    prepare_network(
        n,
        solve_opts=snakemake.params.solving["options"],
        foresight=snakemake.params.foresight,
        planning_horizons=planning_horizons,
        co2_sequestration_potential=snakemake.params["co2_sequestration_potential"],
        limit_max_growth=snakemake.params.get("sector", {}).get("limit_max_growth"),
    )

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        solve_network(
            n,
            config=snakemake.config,
            params=snakemake.params,
            solving=snakemake.params.solving,
            planning_horizons=planning_horizons,
            rule_name=snakemake.rule,
            log_fn=snakemake.log.solver,
            snakemake=snakemake,
        )

    logger.info(f"Maximum memory usage: {mem.mem_usage}")

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output.network)
