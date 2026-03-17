# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

rule solve_2035_once:
    params:
        solving=config_provider("solving"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        energy_year=config_provider("energy", "energy_totals_year"),
        foresight=config_provider("foresight"),
        costs=config_provider("costs"),
    input:
        network=resources(
            "networks/base_s_{clusters}_{opts}_{sector_opts}_2035_final.nc"
        ),
        co2_totals_name=resources("co2_totals.csv"),
        energy_totals=resources("energy_totals.csv"),
        costs=resources("costs_2035.csv"),
    output:
        network=RESULTS + "sweep/{diameter}/base_s_{clusters}_{opts}_{sector_opts}_2035.nc",
    log:
        solver=RESULTS
        + "logs/sweep/base_s_{clusters}_{opts}_{sector_opts}_{diameter}_2035_solver.log",
        memory=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{diameter}_2035_memory.log",
        python=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{diameter}_2035_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_2035_once/base_s_{clusters}_{opts}_{sector_opts}_{diameter}_2035"
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
        "../scripts/solve_2035_once.py"


rule solve_sweep:
    params:
        solving=config_provider("solving"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        energy_year=config_provider("energy", "energy_totals_year"),
        foresight=config_provider("foresight"),
    input:
        network=RESULTS + "sweep/{diameter}/base_s_{clusters}_{opts}_{sector_opts}_2035.nc",
        co2_totals_name=resources("co2_totals.csv"),
        energy_totals=resources("energy_totals.csv"),
        costs=resources("costs_2035.csv"),
    output:
        network=RESULTS + "sweep/{diameter}/base_s_{clusters}_{opts}_{sector_opts}_{threshold}_2035.nc",
    log:
        solver=RESULTS
        + "logs/sweep/base_s_{clusters}_{opts}_{sector_opts}_{diameter}_{threshold}_2035_solver.log",
        memory=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{diameter}_{threshold}_2035_memory.log",
        python=RESULTS
        + "logs/base_s_{clusters}_{opts}_{sector_opts}_{diameter}_{threshold}_2035_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_sweep/base_s_ops_{clusters}_{opts}_{sector_opts}_{diameter}_{threshold}_2035"
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
        "../scripts/solve_sweep.py"


rule solve_sweeps:
    input:
        expand(
            RESULTS + 
            "sweep/{diameter}/base_s_{clusters}_{opts}_{sector_opts}_{threshold}_2035.nc",
            **config["scenario"],
            run=config["run"]["name"],
            diameter=config["solving"]["sweep"]["diameter"],
            threshold=config["solving"]["sweep"]["threshold"],
        ),