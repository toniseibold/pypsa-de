# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

from pathlib import Path


def _generated_co2_topologies(run, clusters):
    topologies_dir = Path(
        checkpoints.build_co2_topologies.get(run=run, clusters=clusters).output.topologies
    ) / "topologies"
    return sorted(path.stem for path in topologies_dir.glob("topology_*.csv"))


def solved_co2_topology_networks(_w):
    run_names = config["run"].get("name")
    if run_names is None:
        run_names = [""]
    elif isinstance(run_names, str):
        run_names = [run_names]

    solved_networks: set[str] = set()
    for run in run_names:
        for clusters in config["scenario"]["clusters"]:
            topologies = _generated_co2_topologies(run, clusters)
            if not topologies:
                continue
            solved_networks.update(
                expand(
                    RESULTS
                    + "co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}.nc",
                    run=[run],
                    topology=topologies,
                    clusters=[clusters],
                    opts=config["scenario"]["opts"],
                    sector_opts=config["scenario"]["sector_opts"],
                )
            )

    return sorted(solved_networks)


rule solve_co2_topology:
    message:
        "Solving sector-coupled network for CO2 topology {wildcards.topology}"
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        energy_year=config_provider("energy", "energy_totals_year"),
    input:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_2035_final.nc"
        ),
        topologies_dir=resources("co2_topologies/base_s_{clusters}_2035"),
        co2_totals_name=resources("co2_totals.csv"),
        energy_totals=resources("energy_totals.csv"),
        costs=resources("costs_2035_processed.csv"),
    output:
        network=RESULTS
        + "co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}.nc",
        config=RESULTS
        + "co2_topologies/{topology}/config.base_s_{clusters}_{opts}_{sector_opts}.yaml",
        revenue=RESULTS
        + "co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}_revenue.txt",
        system_costs=RESULTS
        + "co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}_system_costs.txt",
        carbon_balance=RESULTS
        + "co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}_carbon_balance.csv",
        topology=RESULTS
        + "co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}_topology.csv",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}_solver.log",
        memory=RESULTS
        + "logs/co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}_memory.log",
        python=RESULTS
        + "logs/co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        RESULTS
        + "benchmarks/solve_co2_topology/{topology}/base_s_{clusters}_{opts}_{sector_opts}",
    script:
        scripts("solve_co2_topology.py")


rule solve_co2_topologies:
    message:
        "Collecting all CO2 topology solves"
    input:
        topology_dirs=expand(
            resources("co2_topologies/base_s_{clusters}_2035"),
            clusters=config["scenario"]["clusters"],
            run=config["run"]["name"],
        ),
        solved_networks=solved_co2_topology_networks


# ---------------------------------------------------------------------------
# 2050 re-optimisation through CO2 topology brownfield
# ---------------------------------------------------------------------------


def solved_co2_topology_2050_networks(_w):
    run_names = config["run"].get("name")
    if run_names is None:
        run_names = [""]
    elif isinstance(run_names, str):
        run_names = [run_names]

    solved_networks: set[str] = set()
    for run in run_names:
        for clusters in config["scenario"]["clusters"]:
            topologies = _generated_co2_topologies(run, clusters)
            if not topologies:
                continue
            solved_networks.update(
                expand(
                    RESULTS
                    + "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}.nc",
                    run=[run],
                    topology=topologies,
                    clusters=[clusters],
                    opts=config["scenario"]["opts"],
                    sector_opts=config["scenario"]["sector_opts"],
                )
            )
    return sorted(solved_networks)


rule add_brownfield_co2_topology_2050:
    message:
        "Adding 2050 brownfield from solved CO2 topology {wildcards.topology} for {wildcards.clusters} clusters, {wildcards.opts} opts, {wildcards.sector_opts} sector_opts"
    params:
        H2_retrofit=config_provider("sector", "H2_retrofit"),
        H2_retrofit_capacity_per_CH4=config_provider(
            "sector", "H2_retrofit_capacity_per_CH4"
        ),
        threshold_capacity=config_provider("existing_capacities", "threshold_capacity"),
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        carriers=config_provider("electricity", "renewable_carriers"),
        heat_pump_sources=config_provider("sector", "heat_pump_sources"),
        tes=config_provider("sector", "tes"),
        dynamic_ptes_capacity=config_provider(
            "sector", "district_heating", "ptes", "dynamic_capacity"
        ),
    input:
        **{
            f"profile_{tech}": resources("profile_{clusters}_" + tech + ".nc")
            for tech in config.get("electricity", {}).get("renewable_carriers", [])
            if tech != "hydro"
        },
        simplify_busmap=resources("busmap_base_s.csv"),
        cluster_busmap=resources("busmap_base_s_{clusters}.csv"),
        # Greenfield 2050 prepared network (same as standard add_brownfield 2050 input)
        network=resources("networks/base_s_{clusters}_{opts}_{sector_opts}_2050.nc"),
        # Topology-specific solved 2035 network replaces the standard network_p
        network_p=RESULTS
        + "co2_topologies/{topology}/base_s_{clusters}_{opts}_{sector_opts}.nc",
    output:
        resources(
            "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_2050_brownfield.nc"
        ),
    threads: 4
    resources:
        mem_mb=10000,
    log:
        logs(
            "add_brownfield_co2_topology_2050_{topology}_s_{clusters}_{opts}_{sector_opts}.log"
        ),
    benchmark:
        benchmarks(
            "add_brownfield_co2_topology_2050/{topology}/s_{clusters}_{opts}_{sector_opts}"
        )
    script:
        scripts("add_brownfield.py")


rule modify_prenetwork_co2_topology_2050:
    message:
        "Modifying 2050 pre-network for CO2 topology {wildcards.topology}"
    params:
        efuel_export_ban=config_provider("solving", "constraints", "efuel_export_ban"),
        enable_kernnetz=config_provider("wasserstoff_kernnetz", "enable"),
        technology_occurrence=config_provider("first_technology_occurrence"),
        fossil_boiler_ban=config_provider("new_decentral_fossil_boiler_ban"),
        coal_ban=config_provider("coal_generation_ban"),
        nuclear_ban=config_provider("nuclear_generation_ban"),
        planning_horizons=config_provider("scenario", "planning_horizons"),
        H2_transmission_efficiency=config_provider(
            "sector", "transmission_efficiency", "H2 pipeline"
        ),
        H2_retrofit=config_provider("sector", "H2_retrofit"),
        H2_retrofit_capacity_per_CH4=config_provider(
            "sector", "H2_retrofit_capacity_per_CH4"
        ),
        transmission_costs=config_provider("costs", "transmission"),
        must_run=config_provider("must_run"),
        clustering=config_provider("clustering", "temporal", "resolution_sector"),
        H2_plants=config_provider("electricity", "H2_plants"),
        onshore_nep_force=config_provider("onshore_nep_force"),
        offshore_nep_force=config_provider("offshore_nep_force"),
        shipping_methanol_efficiency=config_provider(
            "sector", "shipping_methanol_efficiency"
        ),
        shipping_oil_efficiency=config_provider("sector", "shipping_oil_efficiency"),
        shipping_methanol_share=config_provider("sector", "shipping_methanol_share"),
        scale_capacity=config_provider("scale_capacity"),
        bev_charge_rate=config_provider("sector", "bev_charge_rate"),
        bev_energy=config_provider("sector", "bev_energy"),
        bev_dsm_availability=config_provider("sector", "bev_dsm_availability"),
        uba_for_industry=config_provider("pypsa-de", "uba_for_industry", "enable"),
        scale_industry_non_energy=config_provider(
            "pypsa-de", "uba_for_industry", "scale_industry_non_energy"
        ),
        limit_cross_border_flows_ac=config_provider(
            "pypsa-de", "limit_cross_border_flows_ac"
        ),
        industry_relocation=config_provider("sector", "industry_relocation"),
        ammonia=config_provider("sector", "ammonia"),
    input:
        network=resources(
            "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_2050_brownfield.nc"
        ),
        wkn=lambda w: (
            resources("wasserstoff_kernnetz_base_s_{clusters}.csv")
            if config_provider("wasserstoff_kernnetz", "enable")(w)
            else []
        ),
        costs=resources("costs_2050_processed.csv"),
        modified_mobility_data=resources("modified_mobility_data_{clusters}_2050.csv"),
        biomass_potentials=resources("biomass_potentials_s_{clusters}_2050.csv"),
        industrial_demand=resources(
            "industrial_energy_demand_base_s_{clusters}_2050.csv"
        ),
        industrial_production_per_country_tomorrow=resources(
            "industrial_production_per_country_tomorrow_2050-modified.csv"
        ),
        industry_sector_ratios=resources("industry_sector_ratios_2050.csv"),
        pop_weighted_energy_totals=resources(
            "pop_weighted_energy_totals_s_{clusters}.csv"
        ),
        shipping_demand=resources("shipping_demand_s_{clusters}.csv"),
        regions_onshore=resources("regions_onshore_base_s_{clusters}.geojson"),
        regions_offshore=resources("regions_offshore_base_s_{clusters}.geojson"),
        offshore_connection_points="data/pypsa-de/offshore_connection_points.csv",
        new_industrial_energy_demand="data/pypsa-de/UBA_Projektionsbericht2025_Abbildung31_MWMS.csv",
    output:
        network=resources(
            "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_2050_final.nc"
        ),
    resources:
        mem_mb=4000,
    log:
        logs(
            "modify_prenetwork_co2_topology_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}.log"
        ),
    script:
        scripts("pypsa-de/modify_prenetwork.py")


rule solve_co2_topology_2050:
    message:
        "Solving 2050 sector-coupled network for CO2 topology {wildcards.topology}"
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        energy_year=config_provider("energy", "energy_totals_year"),
    input:
        network=resources(
            "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_2050_final.nc"
        ),
        co2_totals_name=resources("co2_totals.csv"),
        energy_totals=resources("energy_totals.csv"),
        delaunay_candidates=resources("delaunay_candidates_{clusters}.csv"),
    output:
        network=RESULTS
        + "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}.nc",
        config=RESULTS
        + "co2_topologies_2050/{topology}/config.base_s_{clusters}_{opts}_{sector_opts}.yaml",
        revenue=RESULTS
        + "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_revenue.txt",
        system_costs=RESULTS
        + "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_system_costs.txt",
        carbon_balance=RESULTS
        + "co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_carbon_balance.csv",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_solver.log",
        memory=RESULTS
        + "logs/co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_memory.log",
        python=RESULTS
        + "logs/co2_topologies_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        RESULTS
        + "benchmarks/solve_co2_topology_2050/{topology}/base_s_{clusters}_{opts}_{sector_opts}",
    script:
        scripts("solve_network.py")


rule solve_co2_topologies_2050:
    message:
        "Collecting all 2050 CO2 topology re-optimisations"
    input:
        solved_networks=solved_co2_topology_2050_networks


rule calculate_robustness_metrics:
    message:
        "Calculating robustness metrics for CO2 topologies"
    input:
        topolgy_dirs=expand(
            RESULTS + "co2_topologies",
            clusters=config["scenario"]["clusters"],
            run=config["run"]["name"],
        ),
    output:
        RESULTS + "robustness_metrics.csv",
    script:
        scripts("calculate_robustness_metrics.py")