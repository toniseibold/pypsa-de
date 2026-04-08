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
    input:
        network=RESULTS + "networks/5/base_s_{clusters}_{opts}_{sector_opts}_2035.nc",
        co2_totals_name=resources("co2_totals.csv"),
        energy_totals=resources("energy_totals.csv"),
    output:
        network=RESULTS + "networks/base_s_ops_{clusters}_{opts}_{sector_opts}_{column}_2035.nc",
    log:
        solver=RESULTS
        + "logs/operations/base_s_ops_{clusters}_{opts}_{sector_opts}_{column}_2035_solver.log",
        memory=RESULTS
        + "logs/operations/base_s_ops_{clusters}_{opts}_{sector_opts}_{column}_2035_memory.log",
        python=RESULTS
        + "logs/operations/base_s_ops_{clusters}_{opts}_{sector_opts}_{column}_2035_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_operations_sector_network/base_s_ops_{clusters}_{opts}_{sector_opts}_{column}_2035"
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
            + "networks/base_s_ops_{clusters}_{opts}_{sector_opts}_{column}_2035.nc",
            **config["scenario"],
            run=config["run"]["name"],
            column=config["solve_operations"]["columns"]
        )

rule collect_sweep:
    input:
        expand(
            RESULTS 
            + "networks/{length_max}/base_s_{clusters}_{opts}_{sector_opts}_2035.nc",
            **config["scenario"],
            run=config["run"]["name"],
            length_max=config["sweep"]["length_max"]
        )

rule solve_length_max:
    message:
        "Solving sector-coupled network again with for 2035 with different CO2 pipeline expansion length in Germany"
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
        presolve=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_2035.nc",
        co2_totals_name=resources("co2_totals.csv"),
        energy_totals=resources("energy_totals.csv"),
    output:
        network=RESULTS
        + "networks/{length_max}/base_s_{clusters}_{opts}_{sector_opts}_2035.nc",
        config=RESULTS
        + "configs/{length_max}/config.base_s_{clusters}_{opts}_{sector_opts}_2035.yaml",
    shadow:
        shadow_config
    log:
        solver=RESULTS
        + "logs/operations/{length_max}/base_s_ops_{clusters}_{opts}_{sector_opts}_2035_solver.log",
        memory=RESULTS
        + "logs/operations/{length_max}/base_s_ops_{clusters}_{opts}_{sector_opts}_2035_memory.log",
        python=RESULTS
        + "logs/operations/{length_max}/base_s_ops_{clusters}_{opts}_{sector_opts}_2035_python.log",
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_length_max/{length_max}/base_s_{clusters}_{opts}_{sector_opts}_2035"
        )
    script:
        scripts("solve_length_max.py")