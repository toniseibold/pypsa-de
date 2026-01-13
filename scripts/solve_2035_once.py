# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""

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

diameter_dict = {
    "40": 4930.9,
    "70": 2284.8,
}
cost_dict = {
    4930.9: 40,
    2284.8: 70,
}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_2035_once",
            opts="",
            clusters="89",
            configfiles="config/config.de.yaml",
            sector_opts="none",
            planning_horizons="2035",
            diameter="70",
            run="no_co2_network",
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    solve_opts = snakemake.params.solving["options"]

    np.random.seed(solve_opts.get("seed", 123))

    n = pypsa.Network(snakemake.input.network)

    # change post_discretization config
    logger.info("Deactivate post discretization for sweep")
    snakemake.params.solving["options"]["skip_iterations"] = True
    snakemake.params.solving["options"]["post_discretization"]["enable"] = False
    # change diameter cost assumptions
    logger.info(f"Changing co2 pipeline costs to match {snakemake.wildcards.diameter} cm assumptions.")
    # get current assumption
    costs = pd.read_csv(snakemake.input.costs, index_col=[0, 1]).sort_index()
    co2_costs = costs.loc[("CO2 pipeline", "investment"), "value"]

    # find right factor for capital_costs
    factor = co2_costs / diameter_dict[snakemake.wildcards.diameter]
    logger.info(f"Diameter was {cost_dict[co2_costs]} cm and is changed to {snakemake.wildcards.diameter}cm")
    logger.info(f"Multiplying capital cost for CO2 pipelines by factor {1/factor}")
    co2_links = n.links[n.links.carrier=="CO2 pipelines"].index
    n.links.loc[co2_links, "capital_cost"] *= 1/factor

    planning_horizons = "2035"

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
