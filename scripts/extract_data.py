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
    df_co2_stored.loc["process emissions CC"] = df_co2_stored.loc["process emissions CC"].mul(partial.loc["process emissions"])
    
    # Taking care of sequestration not transport:
    if snakemake.params.carrier_networks["CO2"]["include"]["pcipmi"]:
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
    if not data.empty:
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
    if snakemake.params.carrier_networks["CO2"]["include"]["pcipmi"]:
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


def get_infrastructure(
        n: pypsa.Network
)->None:

    """"
    Extracting the h2 infrastructure for balance plots later on.

    Parameters
    ----------
    n : pypsa.Network
    """

    logger.info("Getting h2 infrastructure for DE and EU")

    data = pd.DataFrame(columns=["DE", "EU"], index=["H2 pipeline GWkm", "CO2 pipeline Mtkm"])
    for carrier in ["H2 pipeline", "CO2 pipeline"]:
        # get all h2 pipelines [GWkm]
        pipelines = n.links[(n.links.carrier.str.contains("H2 pipeline")) & (n.links.active)].index
        length = n.links.loc[pipelines, "length"]
        capacity = n.links.loc[pipelines, "p_nom"].fillna(n.links.loc[pipelines, "p_nom_opt"])
        GWkm = (length * capacity).sum() / 1e6

        pipelines_de = n.links[
            (n.links.carrier.str.contains("H2 pipeline")) & 
            (n.links.active) &
            ((n.links.bus0.str[:2] == "DE") | (n.links.bus1.str[:2] == "DE"))
        ].index
        length_de = n.links.loc[pipelines_de, "length"]
        capacity_de = n.links.loc[pipelines_de, "p_nom"]
        GWkm_de = (length_de * capacity_de).sum() / 1e6

        if carrier == "CO2 pipeline":
            data.loc["CO2 pipeline Mtkm", "DE"] = GWkm_de
            data.loc["CO2 pipeline Mtkm", "EU"] = GWkm
        else:
            data.loc["H2 pipeline GWkm", "DE"] = GWkm_de
            data.loc["H2 pipeline GWkm", "EU"] = GWkm

    logger.info("Saving infrastructure data")
    data.to_csv(snakemake.output.infrastructure)


def get_misc(n):
    # get totex
    totex = n.statistics.capex().sum() + n.statistics.opex().sum()
    # get TWkm H2
    h2_endo = n.links[(n.links.carrier == "H2 pipeline") & (n.links.active) & ((n.links.bus0.str[:2] == "DE") | (n.links.bus1.str[:2] == "DE"))].index
    if not h2_endo.empty:
        h2_cap_endo = n.links.loc[h2_endo, "p_nom_opt"].mul(n.links.loc[h2_endo, "length"]).sum()
    # get Mt/h km CO2
    co2_endo = n.links[(n.links.carrier == "CO2 pipeline") & (n.links.active) & ~(n.links.index.str.contains("offshore")) & ((n.links.bus0.str[:2] == "DE") | (n.links.bus1.str[:2] == "DE"))].index
    if not co2_endo.empty:
        co2_cap_endo = n.links.loc[co2_endo, "p_nom_opt"].mul(n.links.loc[co2_endo, "length"]).sum()
    # get pci/pmi TWkm H2
    h2_pci = n.links[(n.links.carrier == "H2 pipeline pcipmi") & (n.links.active) & ((n.links.bus0.str[:2] == "DE") | (n.links.bus1.str[:2] == "DE"))].index
    if not h2_pci.empty:
        h2_cap_pci = n.links.loc[h2_pci, "p_nom_opt"].mul(n.links.loc[h2_pci, "length"]).sum()
    # get pci/pmi Mt/h km CO2
    co2_pci = n.links[(n.links.carrier == "CO2 pipeline pcipmi") & (n.links.active) & ((n.links.bus0.str[:2] == "DE") | (n.links.bus1.str[:2] == "DE"))].index
    if not co2_pci.empty:
        co2_cap_pci = n.links.loc[co2_pci, "p_nom_opt"].mul(n.links.loc[co2_pci, "length"]).sum()

    sn = n.snapshot_weightings.generators
    # german totex
    capex = 0
    for component in n.components:
        if component.name in ["Bus", "Carrier", "TransformerType", "GlobalConstraint", "SubNetwork", "LineType", "Load"]:
            continue
        if component.name == "Store":
            attr = "e"
        elif component.name == "Line":
            attr = "s"
        else:
            attr = "p"
        # german ones
        valid = component.static[component.static.index.str[:2] == "DE"].index
        capex += component.static.loc[valid, f"{attr}_nom_opt"].mul(component.static.loc[valid, "capital_cost"]).sum()

    opex = n.statistics.opex(groupby=["country"]).xs("DE", level="country").sum()
    gas = -n.links_t.p1["DE gas compressing"].mul(sn, axis=0).mul(22.4).sum()
    oil = -n.links_t.p1["DE oil refining"].mul(sn, axis=0).mul(38.5629).sum()
    opex = opex + gas + oil

    # value of import
    imp = 0
    exp = 0
    # all imports
    import_l = n.links[~(n.links.bus0.str.startswith("DE")) & (n.links.bus1.str.startswith("DE"))].index
    for index in import_l:
        flow = n.links_t.p0[index]
        bus = n.links.loc[index, "bus1"]
        price = n.buses_t.marginal_price[bus]
        imp += flow.mul(sn, axis=0).mul(price, axis=0).sum()
    # all exports
    export_l = n.links[(n.links.bus0.str.startswith("DE")) & ~(n.links.bus1.str.startswith("DE"))].index
    for index in export_l:
        flow = n.links_t.p0[index]
        bus = n.links.loc[index, "bus0"]
        price = n.buses_t.marginal_price[bus]
        exp += flow.mul(sn, axis=0).mul(price, axis=0).sum()
    
    trade = imp + exp

    revenue = n.statistics.revenue(groupby=["carrier", "bus"]).loc["Load", :, :]
    revenue_de = revenue[revenue.index.get_level_values("bus").str.startswith("DE")]
    revenue_de = revenue_de.sum()

    summary = pd.Series(
        {
            "totex": totex,
            "ger_totex": (opex + capex),
            "trade": trade,
            "revenue": revenue_de,
            "h2_endo_TWkm": h2_cap_endo if not h2_endo.empty else 0,
            "co2_endo_Mt_h_km": co2_cap_endo if not co2_endo.empty else 0,
            "h2_pci_TWkm": h2_cap_pci if not h2_pci.empty else 0,
            "co2_pci_Mt_h_km": co2_cap_pci if not co2_pci.empty else 0,
        }
    )
    summary.to_csv(snakemake.output.paper_metrics)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "extract_data",
            simpl="",
            clusters=89,
            planning_horizons=2025,
            opts="",
            ll="vopt",
            sector_opts="none",
            run="no_co2_network",
        )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)
    if snakemake.wildcards.planning_horizons != "2025":
        isi_data(n)

        get_co2_stored(n)

        get_h2(n)

        get_infrastructure(n)
    else:
        data = pd.DataFrame()
        for output in snakemake.output[:-1]:
            data.to_csv(output)

    get_misc(n)

    del n