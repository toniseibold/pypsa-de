# -*- coding: utf-8 -*-
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pandas as pd
import pypsa
from scripts._helpers import configure_logging, mock_snakemake


logger = logging.getLogger(__name__)

def isi_data(
        n: pypsa.Network
)->None:

    """"
    Extracting the co2 stored balance to report to Fraunhofer ISI.

    Parameters
    ----------
    n : pypsa.Network
    """
    logger.info("Collect data for Fraunhofer ISI")
    co2_stored = n.statistics.energy_balance(bus_carrier="co2 stored", groupby=["bus", "carrier"]).droplevel(0).div(1e6)
    df_co2_stored = co2_stored.unstack().T
    
    # little hack to find contribution of process emission, naphtha and methanol-to-o/a
    logger.info("Hacking process emissions...")
    process_emission = n.statistics.supply(bus_carrier="process emissions", groupby=["bus", "carrier"])
    process_emission_df = process_emission.droplevel(0).unstack().T
    process_emission_df.columns = process_emission_df.columns.str.replace("process emissions", "co2 stored")
    partial = process_emission_df.div(process_emission_df.sum())
    partial.fillna(0, inplace=True)
    # partial multiplied by volume
    df_co2_stored.loc["naphtha for industry"] = df_co2_stored.loc["process emissions CC"].mul(partial.loc["naphtha for industry"])
    df_co2_stored.loc["methanol-to-olefins/aromatics"] = df_co2_stored.loc["process emissions CC"].mul(partial.loc["methanol-to-olefins/aromatics"])
    df_co2_stored.loc["process_emissions CC"] = df_co2_stored.loc["process emissions CC"].mul(partial.loc["process emissions"])
    
    # Taking care of sequestration not transport:
    if snakemake.params.carrier_networks["CO2"]["enable"]:
        logger.info("Hacking co2 trade...")
        index = ['PCI-13.8-01', 'PCI-13.8-03', 'PCI-13.4-05', 'PCI-13.5-02',
        'PCI-13.1+1-02', 'PCI-13.13']
        seq_vol = n.links_t.p0[index].sum().div(1e6)
        df_co2_stored.loc["co2 sequestered", n.links.loc[index, "bus0"]] = seq_vol.values
        # subtract from trade
        seq_vol.index = n.links.loc[index, "bus1"]
        seq_vol = seq_vol.groupby(seq_vol.index).sum()
        df_co2_stored.loc["CO2 pipeline pcipmi", seq_vol.index] += seq_vol.values

    logger.info("Saving co2 stored balance for Fraunhofer ISI")
    df_co2_stored.to_csv(snakemake.output.fh_co2_stored)
    
    logger.info("Getting co2 stored flows")
    index = n.links[(n.links.carrier.str.contains("CO2 pipeline")) & ~(n.links.index.str.contains("offshore"))].index
    vol = n.links_t.p0[index].mul(n.snapshot_weightings.generators, axis=0).sum()
    data = pd.DataFrame([n.links.loc[index, "bus0"].values, n.links.loc[index, "bus1"].values, vol.values], columns=index, index=["bus0", "bus1", "volume t/yr"]).T
    # don't show the sequestration pipelines
    data = data[~data.bus1.str.startswith("PCI")]
    
    logger.info("Saving co2 stored flows for Fraunhofer ISI")
    data.to_csv(snakemake.output.fh_co2_flow)


def get_co2_stored(
        n: pypsa.Network
)->None:

    """"
    Extracting the co2 stored balance for balance plots later on.

    Parameters
    ----------
    n : pypsa.Network
    """

    logger.info("Getting co2 stored balance for DE and EU")
    CO2_DE = (
        n.statistics.energy_balance(bus_carrier="co2 stored", groupby=["bus", "carrier"])
        .filter(like="DE")
        .groupby("carrier")
        .sum()
        .div(1e6)
    )
    # add import/export + sequestration to Germany
    seq = n.links[
        (n.links.bus0.str[:2] == "DE") & 
        (n.links.bus1.str.startswith("PCI")) &
        (n.links.carrier.str.contains("CO2 pipeline"))
    ].index
    incoming = n.links[
        (n.links.bus0.str[:2] != "DE") & 
        (n.links.bus1.str[:2] == "DE") &
        ~(n.links.bus0.str.startswith("PCI")) &
        (n.links.carrier.str.contains("CO2 pipeline"))
    ].index
    outgoing = n.links[
        (n.links.bus0.str[:2] == "DE") & 
        (n.links.bus1.str[:2] != "DE") &
        ~(n.links.bus1.str.startswith("PCI")) &
        (n.links.carrier.str.contains("CO2 pipeline"))
    ].index
    seq_vol = n.links_t.p0[seq].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6
    if "co2 sequestered" in CO2_DE.index:
        CO2_DE["co2 sequestered"] -= seq_vol 
    else:
        CO2_DE["co2 sequestered"] = -seq_vol
    CO2_DE["trade"] = n.links_t.p0[incoming].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6 - n.links_t.p0[outgoing].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6
    
    # drop "CO2 pipeline pcipmi"
    if snakemake.params.carrier_networks["CO2"]["enable"]:
        CO2_DE.drop("CO2 pipeline pcipmi", inplace=True)

    CO2_EU = (
        n.statistics.energy_balance(bus_carrier="co2 stored")
        .groupby("carrier")
        .sum()
        .div(1e6)
    )

    # drop store & pipelines
    CO2_DE = CO2_DE[~(CO2_DE.index.str.contains("pipeline"))]
    CO2_EU = CO2_EU[~(CO2_EU.index.str.contains("pipeline"))]

    # reorder
    order = list(n.statistics.energy_balance(bus_carrier="co2 stored").sort_values(ascending=False).index.get_level_values(1))
    order.insert(-1, "trade")
    CO2_DE = CO2_DE.reindex(order)
    CO2_EU = CO2_EU.reindex(order)
    CO2_DE = CO2_DE[(abs(CO2_DE) >= 1)]
    CO2_EU = CO2_EU[(abs(CO2_EU) >= 1)]
    CO2_DE = CO2_DE[::-1]
    CO2_EU = CO2_EU[::-1]

    logger.info("Saving co2 stored balances")
    CO2_DE.to_csv(snakemake.output.co2_stored_DE)
    CO2_EU.to_csv(snakemake.output.co2_stored_EU)


def get_h2(
        n: pypsa.Network
)->None:

    """"
    Extracting the h2 balance for balance plots later on.

    Parameters
    ----------
    n : pypsa.Network
    """

    logger.info("Getting h2 balance for DE and EU")

    H2_DE = (
        n.statistics.energy_balance(bus_carrier="H2", groupby=["bus", "carrier"])
        .filter(like="DE")
        .groupby("carrier")
        .sum()
        .div(1e6)
    )
    # add import/export to Germany
    incoming = n.links[
        (n.links.bus0.str[:2] != "DE") & 
        (n.links.bus1.str[:2] == "DE") &
        (n.links.carrier.str.contains("H2 pipeline"))
    ].index
    outgoing = n.links[
        (n.links.bus0.str[:2] == "DE") & 
        (n.links.bus1.str[:2] != "DE") &
        (n.links.carrier.str.contains("H2 pipeline"))
    ].index
    H2_DE["import"] = n.links_t.p0[incoming].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6
    H2_DE["export"] = -n.links_t.p0[outgoing].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6

    H2_EU = (
        n.statistics.energy_balance(bus_carrier="H2")
        .groupby("carrier")
        .sum()
        .div(1e6)
    )

    # drop store & pipelines
    H2_DE = H2_DE[~(H2_DE.index.str.contains("pipeline")) & ~(H2_DE.index.str.contains("Store"))]
    H2_EU = H2_EU[~(H2_EU.index.str.contains("pipeline")) & ~(H2_EU.index.str.contains("Store"))]

    # reorder
    order = list(n.statistics.energy_balance(bus_carrier="H2").sort_values(ascending=False).index.get_level_values(1))
    order.insert(5, "import")
    order.insert(-1, "export")
    H2_DE = H2_DE.reindex(order)
    H2_EU = H2_EU.reindex(order)

    logger.info("Saving co2 stored balances")
    H2_DE.to_csv(snakemake.output.h2_DE)
    H2_EU.to_csv(snakemake.output.h2_EU)



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "extract_data",
            simpl="",
            clusters=89,
            planning_horizons=2035,
            opts="",
            ll="vopt",
            sector_opts="none",
            run="endo_H2",
        )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    isi_data(n)

    get_co2_stored(n)

    get_h2(n)

    del n