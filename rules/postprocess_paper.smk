# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT

# DONE
rule extract_data:
    params:
        carrier_networks=config_provider("carrier_networks"),
    input:
        network=RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_{planning_horizons}.nc",
    output:
        co2_stored_DE=RESULTS + "DE_co2_stored_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        co2_stored_EU=RESULTS + "EU_co2_stored_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        h2_DE=RESULTS + "DE_h2_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        h2_EU=RESULTS + "EU_h2_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        infrastructure=RESULTS + "infrastructure_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        fh_co2_stored=RESULTS + "fraunhofer_co2_stored_balance_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
        fh_co2_flow=RESULTS + "fraunhofer_co2_flow_{clusters}_{opts}_{sector_opts}_{planning_horizons}.csv",
    log:
        RESULTS + "logs/extract_data_{clusters}_{opts}_{sector_opts}_{planning_horizons}.log",
    benchmark:
        RESULTS + "benchmark/extract_data_{clusters}_{opts}_{sector_opts}_{planning_horizons}",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/extract_data.py"


rule plot_balances:
    params:
        plotting=config_provider("plotting"),
        scenarios=config_provider("run", "name"),
    input:
        co2_stored_DE=expand(
            RESULTS + "DE_co2_stored_{clusters}_{opts}_{sector_opts}_2035.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        co2_stored_EU=expand(
            RESULTS + "EU_co2_stored_{clusters}_{opts}_{sector_opts}_2035.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        h2_DE=expand(
            RESULTS + "DE_h2_{clusters}_{opts}_{sector_opts}_2035.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
        h2_EU=expand(
            RESULTS + "EU_h2_{clusters}_{opts}_{sector_opts}_2035.csv",
            **config["scenario"],
            run=config["run"]["name"],
        ),
    log:
        RESULTS + "logs/plot_balance_2035.log",
    benchmark:
        RESULTS + "benchmark/plot_balance_2035",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_balances.py"


rule plot_maps:
    params:
        plotting=config_provider("plotting"),
        scenarios=config_provider("run", "name"),
    input:
        networks=expand(
        RESULTS
        + "networks/base_s_{clusters}_{opts}_{sector_opts}_2045.nc",
        **config["scenario"],
        run=config["run"]["name"],
        ),
        regions="resources/regions_onshore_base_s_{clusters}.geojson",
    log:
        RESULTS + "logs/plot_maps_{clusters}_{opts}_{sector_opts}_2045.log",
    benchmark:
        RESULTS + "benchmark/plot_maps_{clusters}_{opts}_{sector_opts}_2045",
    conda:
        "../envs/environment.yaml"
    script:
        "../scripts/plot_maps.py"


rule plot_report:
    input:
        expand(
            RESULTS + "logs/plot_balance_2035.log",
            run=config_provider("run", "name"),
        ),
        # expand(
        #     RESULTS + "logs/plot_maps_{clusters}_{opts}_{sector_opts}_2035.log",
        #     run=config_provider("run", "name"),
        # ),
