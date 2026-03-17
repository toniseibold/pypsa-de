# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Solves linear optimal dispatch using the capacities of previous capacity expansion in rule :mod:`solve_network`.
Custom constraints and extra_functionality can be set in the config.
"""

import logging
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
import pypsa
from pathlib import Path
from scripts._benchmark import memory_logger
from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)
from scripts.solve_network import solve_network

logger = logging.getLogger(__name__)


def set_minimum_investment(
    n: pypsa.Network,
    planning_horizons: str,
    comps: list=["Generator", "Link", "Store", "Line"],
) -> None:
    """
    Sets a minimum investment for a given carrier in the network, allows for extendable components of the planning horizon.
    """
    logger.info(f"Fixing optimal capacities for components before the investment run.")
    logger.info("Setting minimum capacities of components based on results from investment run.")
    logger.info(f"Components: {comps}")

    planning_horizons = int(planning_horizons)

    for c in comps:
        ext_i = n.get_extendable_i(c)
        attrs = n.component_attrs[c]
        nominal_attr = attrs.loc[attrs.index.str.endswith("_nom")].index.values[0]

        c_in_build_year = n.static(c).loc[ext_i, "build_year"] == planning_horizons

        mask = ext_i[c_in_build_year]

        if mask.any():
            # For case where optimal capacity is slightly higher than maximum capacity due to solver tolerances
            n.static(c).loc[mask, nominal_attr+"_opt"] = np.minimum(
                n.static(c).loc[mask, nominal_attr+"_opt"],
                n.static(c).loc[mask, nominal_attr+"_max"],
            )

            b_reached_max = n.static(c).loc[mask, nominal_attr+"_opt"] == n.static(c).loc[mask, nominal_attr+"_max"]


            # If maximum potential is reached:
            n.static(c).loc[b_reached_max[b_reached_max].index, nominal_attr] = n.static(c).loc[b_reached_max[b_reached_max].index, nominal_attr+"_opt"]

            n.static(c).loc[b_reached_max[b_reached_max].index, nominal_attr+"_extendable"] = False

            # If maximum potential is not reached:
            n.static(c).loc[b_reached_max[~b_reached_max].index, nominal_attr+"_min"] = n.static(c).loc[b_reached_max[~b_reached_max].index, nominal_attr+"_opt"]

            n.static(c).loc[b_reached_max[~b_reached_max].index, nominal_attr+"_extendable"] = True


def add_load_shedding(
    n: pypsa.Network,
    marginal_cost: float=10000,
) -> None:
    """
    Adds load shedding to the network.
    """
    n.add("Carrier", "load", color="#dd2e23", nice_name="Load Shedding")
    buses_i = pd.Index(n.loads.bus.unique())

    logger.info(f"Adding load shedding to buses with carriers {n.buses.carrier[buses_i].unique()}.")
    logger.info(f"Load shedding marginal cost: {marginal_cost} EUR/MWh.")
    n.add(
        "Generator",
        buses_i,
        " load",
        bus=buses_i,
        carrier="load",
        marginal_cost=marginal_cost,
        p_nom_extendable=True,
    )    

    n.add(
        "Generator",
        buses_i,
        " load negative",
        bus=buses_i,
        carrier="load",
        marginal_cost=-marginal_cost,
        p_nom_extendable=True,
        p_min_pu=-1,
        p_max_pu=0,
    )    


def remove_pipelines(
    n: pypsa.Network,
    carrier: str,
) -> None:
    """
    Removes carrier pipelines from the network.
    """
    logger.info(f"Removing {carrier}s from the network.")
    if carrier in n.links.carrier.values:
        sum_active = n.links.loc[n.links.carrier.str.contains(carrier), "active"].sum()
        if sum_active == 0:
            logger.info(f"No active {carrier}s in the network.")
            return
        n.remove("Link", n.links.loc[n.links.carrier.str.contains(carrier)].index)
        logger.info(f"Removed {sum_active} active {carrier}s from the network.")
    else:
        logger.warning(f"No {carrier}s found in the network.")
        return


def add_pipeline_topology(n, c):
    """
    Adds the H2 and CO2 pipeline topology from the run c
    """
    logger.info(f"Adding H2 pipeline topology of run {c}")
    fn = snakemake.input.h2_links.replace(snakemake.wildcards.run, c)
    topology = pd.read_csv(fn, index_col=0)
    # make non_extendable
    topology.p_nom_extendable = False
    # save p_nom_opt to p_nom
    topology.p_nom = topology.p_nom_opt
    for name, row in topology.iterrows():
        n.add("Link", name, **row.dropna().to_dict())
    
    logger.info(f"Adding CO2 buses of run {c}")
    fn = snakemake.input.co2_buses.replace(snakemake.wildcards.run, c)
    topology = pd.read_csv(fn, index_col=0)
    n.buses.drop(n.buses[n.buses.carrier.isin(["co2 stored", "co2 sequestered"])].index, inplace=True)
    for name, row in topology.iterrows():
        n.add("Bus", name, **row.dropna().to_dict())

    logger.info(f"Adding CO2 pipeline topology of run {c}")
    fn = snakemake.input.co2_links.replace(snakemake.wildcards.run, c)
    topology = pd.read_csv(fn, index_col=0)
    # make non_extendable
    topology.p_nom_extendable = False
    # save p_nom_opt to p_nom
    topology.p_nom = topology.p_nom_opt
    for name, row in topology.iterrows():
        n.add("Link", name, **row.dropna().to_dict(), overwrite=True)
    
    logger.info(f"Adding CO2 stores of run {c}")
    fn = snakemake.input.co2_stores.replace(snakemake.wildcards.run, c)
    topology = pd.read_csv(fn, index_col=0)
    ext = topology[topology.e_nom_extendable==True].index
    # make non_extendable
    topology.e_nom_extendable = False
    # save e_nom_opt to e_nom
    topology.loc[ext, "e_nom"] = topology.loc[ext, "e_nom_opt"]
    n.stores.drop(n.stores[n.stores.carrier.isin(["co2 stored", "co2 sequestered"])].index, inplace=True)
    for name, row in topology.iterrows():
        n.add("Store", name, **row.dropna().to_dict())


def add_onshore_seq(n):
    # get the potential for onshore sequestration run
    fn = snakemake.input.co2_sequestration_potential.replace(snakemake.wildcards.run, "onshore_sequestration")

    upper_limit = 25*1e3  # Mt
    annualiser = 25

    # Regional potential
    if Path(fn).exists:
        sequestration_potential = gpd.read_file(fn).set_index("cluster")
    else:
        logger.error(f"File for onshore potential {fn} does not exist. Please rerun basic workflow to create this file")
        sys.exit()

    sequestration_potential["e_nom_max"] = (
        sequestration_potential["total_estimate_Mt"]
        .fillna(0.0)
        .mul(1e6)
        .div(annualiser)
        .clip(upper=upper_limit*1e6)
    )  # tpa
    seq_cost = n.links[n.links.carrier=="co2 sequestered"].marginal_cost.mean().round()
    for seq_site in sequestration_potential.index:
        if seq_site + " co2 sequestered" in n.buses.index:
            # just adjust the potential
            store_ind = n.stores[n.stores.bus==seq_site + " co2 sequestered"].index
            n.stores.loc[store_ind, "e_nom"] = sequestration_potential.loc[seq_site, "e_nom_max"]
        else:
            # co2 storage bus
            n.add(
                "Bus",
                seq_site + " co2 sequestered",
                x=sequestration_potential.loc[seq_site, "x"],
                y=sequestration_potential.loc[seq_site, "y"],
                carrier="co2 sequestered",
                unit="t_co2",
            )
            # storage
            n.add(
                "Store",
                seq_site + " co2 sequestered",
                e_nom_extendable=False,
                e_nom=sequestration_potential.loc[seq_site, "e_nom_max"],
                marginal_cost=-0.1,
                bus=seq_site + " co2 sequestered",
                lifetime=50,
                carrier="co2 sequestered",
                build_year=planning_horizons,
            )
            # link between stored and sequestered
            n.add(
                "Link",
                seq_site,
                bus0=sequestration_potential.loc[seq_site, "bus_onshore"] + " co2 stored",
                bus1=seq_site + " co2 sequestered",
                marginal_cost=seq_cost,
                capital_cost=0.1,
                carrier="co2 sequestered",
                efficiency=1.0,
                p_nom_extendable=True,
                reversed=False,
            )


def fix_network(n):
    carriers=[
        "H2 pipeline",
        "H2 pipeline (Kernnetz)",
        "H2 pipeline retrofitted",
        "CO2 pipeline",
    ]
    logger.info(f"Fixing the built capacity of {carriers}.")
    net_links = n.links[n.links.carrier.isin(carriers)].index
    n.links.loc[net_links, "p_nom"] = n.links.loc[net_links, "p_nom_opt"]
    n.links.loc[net_links, "p_nom_extendable"] = False


def add_co2_network(n):
    # get the filename
    fn = "/home/toni-seibold/dev/pypsa-de-co2/results/cluster/2025_10_31_25H/pcipmi_H2_+/topology/CO2_pipelines_base_s_49__none_2035.csv"
    topology = pd.read_csv(fn, index_col=0)
    # make non_extendable
    topology.p_nom_extendable = False
    # save p_nom_opt to p_nom
    topology.p_nom = topology.p_nom_opt
    for name, row in topology.iterrows():
        n.add("Link", name, **row.dropna().to_dict())


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "solve_operations_sector_network",
            opts="",
            clusters="89",
            sector_opts="none", 
            planning_horizons="2035",
            column="low_electrolysis",
            run="no_co2_network",
            configfiles="config/config.de.yaml",
        )
 
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    rule_name = snakemake.rule
    params = snakemake.params
    config = snakemake.config
    solving = snakemake.params.solving

    planning_horizons = "2035"
    c = snakemake.wildcards.get("column", None)
    snakemake.wildcards.planning_horizons = "2035"
    np.random.seed(solving.get("seed", 123))

    # Update solving options
    solving["options"]["noisy_costs"] = False # Only apply noisy_costs once to enable the same solution space
    solving["options"]["skip_iterations"] = True
    solving["options"]["post_discretization"]["enable"] = False

    n = pypsa.Network(snakemake.input.network)
    # skip if the same
    if c == snakemake.wildcards.run:
        logger.info(f"Rerunning optimization without adjustments.")

    if {snakemake.wildcards.run, c} == {"pcipmi_H2_+", "no_co2_network"}:
        remove_pipelines(n, carrier="CO2 pipeline")
        remove_pipelines(n, carrier="H2 pipeline")
        add_pipeline_topology(n, c)

    else: 
        fix_network(n)

    set_minimum_investment(n, planning_horizons)

    if c == "onshore_sequestration":
        # add onshore sequestration potentials
        add_onshore_seq(n)

    if c == "low_gas_price":
        logger.info("low gas price of 11.4 €/MWh")
        n.generators.loc["EU gas primary", "marginal_cost"] = 22.4/2
        n.generators.loc["DE gas primary", "marginal_cost"] = 22.4/2
    if c == "high_gas_price":
        logger.info("high gas price of 44.8 €/MWh")
        n.generators.loc["EU gas primary", "marginal_cost"] = 22.4*2
        n.generators.loc["DE gas primary", "marginal_cost"] = 22.4*2
    if c == "low_seq_cost":
        logger.info("low sequestration costs of 17.5 €/t")
        seq_links = n.links[n.links.carrier=="co2 sequestered"].index
        n.links.loc[seq_links, "marginal_cost"] = 35/2
        seq_stores = n.stores[n.stores.carrier=="co2 sequestered"].index
        n.stores.loc[seq_stores, "capital_cost"] = 35/2
    if c == "high_seq_cost":
        logger.info("low sequestration costs of 70 €/t")
        seq_links = n.links[n.links.carrier=="co2 sequestered"].index
        n.links.loc[seq_links, "marginal_cost"] = 35*2
        seq_stores = n.stores[n.stores.carrier=="co2 sequestered"].index
        n.stores.loc[seq_stores, "capital_cost"] = 35*2
    
    if c == "low_seq_potential":
        logger.info("sequestration potential of 50 Mt/a")
        n.global_constraints.loc["co2_sequestration_limit", "constant"] = -50*1e6
    if c == "high_seq_potential":
        logger.info("sequestration potential of 200 Mt/a")
        n.global_constraints.loc["co2_sequestration_limit", "constant"] = -200*1e6

    if c == "low_electrolysis":
        logger.info(f"Constraint for {c} run is added in additional functionality.")

    if c == "high_industry_demand":
        # adjust all industry related loads
        industry_carrier=[
            "solid biomass for industry",
            "gas for industry",
            "H2 for industry",
            "industry methanol",
            "low-temperature heat for industry",
            "naphtha for industry",
            "industry electricity",
            "process emission",
            "NH3",
            "coal for industry",
            "steel",
            "cement",
            
        ]
        ind_constant = n.loads[(n.loads.carrier.isin(industry_carrier)) & (n.loads.index.str.startswith("DE"))].index
        n.loads.loc[ind_constant, "p_set"] *= 1.2

    # # Debugging: Load shedding
    # # if snakemake.params.solve_operations["load_shedding"]:
    # config["solving"]["options"]["load_shedding"] = True
    # marginal_cost = 10000
    # add_load_shedding(n, marginal_cost)

    # #################################

    # Store updated params and config in network file
    n.params = params
    n.config = config

    # Run the operational stage of the model
    logger.info("---")
    logger.info(f"Running operational optimisation for column ['{c}']")

    logging_frequency = snakemake.config.get("solving", {}).get(
        "mem_logging_frequency", 30
    )
    with memory_logger(
        filename=getattr(snakemake.log, "memory", None), interval=logging_frequency
    ) as mem:
        solve_network(
            n,
            config=config,
            params=params,
            solving=solving,
            planning_horizons=planning_horizons,
            rule_name=rule_name,
            log_fn=snakemake.log.solver,
            snakemake=snakemake,
        )

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
