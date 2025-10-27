# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

rule solve_operations_sector_network:
    params:
        solving=config_provider("solving"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        solve_operations=config_provider("solve_operations"),
        energy_year=config_provider("energy", "energy_totals_year"),
        foresight=config_provider("foresight"),
    input:
        network=RESULTS + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
        co2_totals_name=resources("co2_totals.csv"),
        energy_totals=resources("energy_totals.csv"),
        h2_links=RESULTS
        + "topology/H2_pipelines_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        co2_links=RESULTS
        + "topology/CO2_pipelines_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        co2_buses=RESULTS
        + "topology/CO2_buses_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        co2_stores=RESULTS
        + "topology/CO2_stores_base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        co2_sequestration_potential=resources("co2_sequestration_potential_base_s_{clusters}.geojson"),
    output:
        network=RESULTS + "networks/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    log:
        solver=RESULTS
        + "logs/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_solver.log",
        memory=RESULTS
        + "logs/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_memory.log",
        python=RESULTS
        + "logs/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_operations_sector_network/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}"
        )
    threads: 4
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/solve_operations_sector_network.py"


rule solve_operations_sector_networks:
    input:
        expand(
            RESULTS 
            + "networks/operations/{column}/base_s_ops_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        ),