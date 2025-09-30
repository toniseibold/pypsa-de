# -*- coding: utf-8 -*-
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cartopy
import cartopy.crs as ccrs
import country_converter as coco
cc = coco.CountryConverter()
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pypsa
from matplotlib.patches import Patch
from pypsa.plot.maps.static import add_legend_circles, add_legend_lines, add_legend_patches

import seaborn as sns
from pypsa.statistics import get_transmission_carriers
from scripts._helpers import configure_logging, mock_snakemake
from scripts.make_summary import assign_locations
from scripts.plot_power_network import load_projection
import textwrap

logger = logging.getLogger(__name__)

color_palette = {
        'hbi EU in': '#4682B4',
        'hbi EU out': '#4682B4',
        'hbi non EU': '#738399',
        'hydrogen EU in': '#d9b8d3',
        'hydrogen EU out': '#d9b8d3',
        'hydrogen non EU': '#db8ccd',
        'Fischer-Tropsch EU in': '#95decb',
        'Fischer-Tropsch EU out': '#95decb',
        'Fischer-Tropsch non EU': '#598579',
        'ammonia EU in': '#87d0e6',
        'ammonia EU out': '#87d0e6',
        'ammonia non EU': '#598896',
        'methanol EU in': '#b7dbdb',
        'methanol EU out': '#b7dbdb',
        'methanol non EU': '#6a8080',
        'electricity non EU': '#91856a',
        'electricity EU in': '#ccbb95',
        'electricity EU out': '#ccbb95',
        'fossil gas': '#edaf1c',
        'fossil oil': '#de571d',
        'coal': '#402920',
        'lignite': 'brown',
    }
order = [
        'hbi EU in',
        'hbi EU out',
        'hbi non EU',
        'hydrogen EU in',
        'hydrogen EU out',
        'hydrogen non EU',
        'Fischer-Tropsch EU in',
        'Fischer-Tropsch EU out',
        'Fischer-Tropsch non EU',
        'ammonia EU in',
        'ammonia EU out',
        'ammonia non EU',
        'methanol EU in',
        'methanol EU out',
        'methanol non EU',
        'electricity non EU',
        'electricity EU in',
        'electricity EU out',
        'fossil gas',
        'fossil oil',
        'coal',
        'lignite',
    ]

non_eu_carriers = [
    'import hvdc-to-elec',
    'import infrastructure pipeline-h2',
    'import infrastructure shipping-lh2',
    'import shipping-meoh',
    'import shipping-lnh3',
    'import shipping-ftfuel',
    'import shipping-hbi'
]

scenario_dict = {
  "Base": "Base",
  "EH": "EH",
  "WH": "WH",
  "EHP": "EHP",
  "WHP": "WHP",
  "400Mt_seq_Base": "Base 400Mt Seq",
  "400Mt_seq_EH": "EH 400Mt Seq",
  "400Mt_seq_WH": "WH 400Mt Seq",
  "400Mt_seq_EHP": "EHP 400Mt Seq",
  "400Mt_seq_WHP": "WHP 400Mt Seq",
  "50per_electrolysis_Base": "Base -50%",
  "50per_electrolysis_EH": "EH -50%",
  "50per_electrolysis_WH": "WH -50%",
  "50per_electrolysis_EHP": "EHP -50%",
  "50per_electrolysis_WHP": "WHP -50%",
  "wacc_10_Base": "Base wacc 10%",
  "wacc_10_EH": "EH wacc 10%",
  "wacc_10_WH": "WH wacc 10%",
  "wacc_10_EHP": "EHP wacc 10%",
  "wacc_10_WHP": "WHP wacc 10%",
}

def prepare_colors():
    colors = snakemake.params.plotting["tech_colors"]

    colors["urban decentral heat"] = colors["residential urban decentral heat"]
    colors["Electrolysis"] = colors["H2 Electrolysis"]
    colors["Storage discharge"] = colors["hydrogen storage"]
    colors["Storage charge"] = colors["hydrogen storage"]
    colors["Import"] = colors["H2 pipeline"]
    colors["Export"] = colors["H2 pipeline"]
    colors["DRI"] = "#8f9c9a"
    colors["H2 OCGT"] =  colors["H2 turbine"]
    colors["H2 CHP"] = colors["CHP"]
    colors["Non EU Import"] = "grey"
    colors["Trade"] = "grey"

    return colors


def get_time_resolution(resolution_sector):
    seg_method = re.match(r'(\d+)([a-zA-Z]+)', resolution_sector).group(2)
    if seg_method == "H":
        # evenly distributed time resolution
        time_resolution = int(re.match(r'(\d+)([a-zA-Z]+)', resolution_sector).group(1))
    elif seg_method == "SEG":
        # segmentation
        time_resolution = 8760 / int(re.match(r'(\d+)([a-zA-Z]+)', resolution_sector).group(1))
    else:
        logger.error("Config Resolution Sector is not 'H' or 'SEG'. Please check!")
    return time_resolution


def plot_hydrogen_balance(n, colors, year, scenario, savepath):

    buses = n.buses.index[(n.buses.index.str[:2] == "DE")].drop("DE")
    nodal_balance = (
        n.statistics.energy_balance(
            aggregate_time=False,
            nice_names=False,
            groupby=pypsa.statistics.groupers["bus", "carrier", "bus_carrier"],
        )
        .loc[:, buses, :, :]
        .droplevel("bus").mul(n.snapshot_weightings.generators)) # multiply with snapshot weightings

    carriers = ["H2"]
    loads = ["land transport fuel cell", "H2 for industry"]

    period = n.generators_t.p.index
    mask = nodal_balance.index.get_level_values("bus_carrier").isin(carriers)
    nb = nodal_balance[mask].groupby("carrier").sum().div(1e3).T.loc[period] # GWh
    df_loads = abs(nb[loads].sum(axis=1))
    # round and remove all carriers which accumulate to zero except for Store
    summe = nb.round(decimals=1).sum()
    ind_to_drop = summe[summe==0].index
    if "H2 Store" in ind_to_drop:
        ind_to_drop.drop("H2 Store")

    nb.drop(columns=ind_to_drop, inplace=True)

    # consolidate pipeline links
    if "H2 pipeline (Kernnetz)" in nb.columns:
        nb["Trade"] = nb["H2 pipeline"] + nb["H2 pipeline (Kernnetz)"] + nb["H2 pipeline retrofitted"]
        nb.drop(columns=["H2 pipeline", "H2 pipeline (Kernnetz)", "H2 pipeline retrofitted"], inplace=True)
    elif "H2 pipeline" in nb.columns:
        nb["Trade"] = nb["H2 pipeline"] + nb["H2 pipeline retrofitted"]
        nb.drop(columns=["H2 pipeline", "H2 pipeline retrofitted"], inplace=True)
    else:
        nb["Trade"] = 0
    trade_bal = nb["Trade"].cumsum()

    # separate demand and supply of hydrogen
    nb_neg, nb_pos = nb.clip(upper=0), nb.clip(lower=0)
    nb_neg = nb_neg.loc[:, nb_neg.sum(axis=0) < 0]
    nb_pos = nb_pos.loc[:, nb_pos.sum(axis=0) > 0]

    # rename
    if "urban central H2 retrofit CHP" in nb_neg.columns:
        nb_neg["H2 CHP"] = nb_neg.get("urban central H2 CHP", 0) + nb_neg.get("urban central H2 retrofit CHP", 0)
        nb_neg.drop(columns=["urban central H2 CHP", "urban central H2 retrofit CHP"], errors="ignore", inplace=True)
    else:
        nb_neg.rename(columns={"urban central H2 CHP": "H2 CHP"}, inplace=True)
    if "H2 OCGT" in nb_neg.columns:
        nb_neg["H2 OCGT"] = nb_neg.get("H2 OCGT", 0) + nb_neg.get("H2 retrofit OCGT", 0)
        nb_neg.drop(columns=["H2 retrofit OCGT"], errors="ignore", inplace=True)
    rename_neg = {
        "H2 Store": "Storage charge",
        "Trade": "Export",
    }
    # rename Trade to Import and H2 Store to Store discharge
    rename_pos = {
        "H2 Store": "Storage discharge",
        "Trade": "Import",
        "H2 Electrolysis": "Electrolysis",
        "import infrastructure shipping-lh2": "Non EU Import"
    }
    nb_pos.rename(columns=rename_pos, inplace=True)
    nb_neg.rename(columns=rename_neg, inplace=True)

    # sort values
    pref_order_pos = ["Electrolysis", "SMR", "SMR CC", "Storage discharge", "Import", "Non EU Import"]
    pref_order_neg = ["H2 for industry", "DRI", "Haber-Bosch", "methanolisation", "Fischer-Tropsch", "H2 OCGT", "H2 CHP", "Storage charge", "Export"]

    valid_columns_pos = [col for col in pref_order_pos if col in nb_pos.columns]
    valid_columns_neg = [col for col in pref_order_neg if col in nb_neg.columns]

    nb_neg = nb_neg[valid_columns_neg]
    nb_pos = nb_pos[valid_columns_pos]

    barplot = pd.concat([nb_neg.sum(), nb_pos.sum()]).div(1e3)
    if year == "2050":
        barplot.to_csv(snakemake.output.hydrogen)
    # avoid error when temporal resolution is too low
    if time_resolution < 24:
        nb_neg = nb_neg.resample("D").mean()
        nb_pos = nb_pos.resample("D").mean()

    fig, ax = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [4, 1]})
    
    ax1 = nb_pos.plot.area(ax=ax[0], stacked=True, color=[colors[col] for col in nb_pos.columns], linewidth=0.0)
    ax1 = nb_neg.plot.area(ax=ax[0], stacked=True, color=[colors[col] for col in nb_neg.columns], linewidth=0.0)

    trade_bal.div(1e3).plot(ax=ax[0], color="black", linestyle="-", linewidth=2, label="Trade balance", secondary_y=True)

    # take care of legend
    handles, labels = ax1.get_legend_handles_labels()

    supply_handles = handles[:5]
    supply_labels = labels[:5]
    demand_handles = handles[5:]
    demand_labels = labels[5:]
    subtitle_supply = Patch(color="none", label="Supply")
    subtitle_demand = Patch(color="none", label="Demand")

    # Combine all handles and labels
    combined_handles = (
        [subtitle_supply]
        + supply_handles
        + [subtitle_demand]
        + demand_handles
    )
    combined_labels = (
        ["Supply"] + supply_labels + ["Demand"] + demand_labels
    )

    legend = ax1.legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=(1.15, 0.3),
    )

    # adjust the plot
    ax1.set_ylim([1.05 * nb_neg.sum(axis=1).min(), 1.05 * nb_pos.sum(axis=1).max()])
    ax1.set_ylabel("GWh")
    ax1.set_title(f"Hydrogen Balance Germany {scenario} {year}")
    ax1.set_xlabel("")

    ax1.right_ax.legend(["Trade Balance [TWh]"], loc="upper right")

    barplot.to_frame().T.plot.barh(ax=ax[1], color=[colors[col] for col in barplot.to_frame().T.columns], stacked=True, width=0.8, legend=False)
    ax[1].set_xlabel("TWh")
    ax[1].set_title("Aggregated")
    ax[1].set_ylabel("")
    ax[1].grid(axis='x')

    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_carbon_balance(n, colors, year, scenario, savepath):

    co2_DE = n.statistics.energy_balance(bus_carrier="co2 stored", groupby=["bus", "carrier"]).filter(like="DE").groupby("carrier").sum().div(1e6)
    
    co2_DE.drop("CO2 pipeline", inplace=True)

    # take into account CO2 transport
    out_ind = n.links[(n.links.carrier=="CO2 pipeline") & 
            (n.links.bus0.str.contains("DE")) & 
            (~n.links.bus1.str.contains("DE"))].index # positive: DE -> EU
    in_ind = n.links[(n.links.carrier=="CO2 pipeline") & 
            (~n.links.bus0.str.contains("DE")) & 
            (n.links.bus1.str.contains("DE"))].index # positive: EU -> DE

    out = n.links_t.p0[out_ind].mul(n.snapshot_weightings.generators, axis=0).sum().sum()/1e6
    inf = n.links_t.p0[in_ind].mul(n.snapshot_weightings.generators, axis=0).sum().sum()/1e6

    # assemble data
    co2_bal = co2_DE.to_frame().T
    co2_bal["Trade"] = - (out-inf)

    if year == "2050":
        co2_bal.to_csv(snakemake.output.carbon)

    fig, ax = plt.subplots(figsize=(12, 4))

    co2_bal.plot.barh(ax=ax, stacked=True, color=[colors[col] for col in co2_bal.columns], width=0.8, legend=False)
    
    # adjust plot
    ax.legend(
            loc="upper center",
            bbox_to_anchor=(1.2, 1.1),
        )
    ax.set_title(f"Carbon Balance Germany {year} {scenario}")
    ax.set_xlabel("Mt CO2")

    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def get_import_volumes(n, year, non_eu_import, relocation):
    
    # snapshot weightings
    weights = n.snapshot_weightings.generators
    
    # indices: non-eu import volumes
    if non_eu_import["enable"]:
        noneu_h2 = n.links.loc[(n.links.carrier == "import shipping-lh2") & (n.links.index.str.contains("DE"))].index
        noneu_hbi = n.links.loc[(n.links.carrier == "import shipping-hbi") & (n.links.index.str.contains("DE"))].index
        noneu_nh3 = n.links.loc[(n.links.carrier == "import shipping-lnh3") & (n.links.index.str.contains("DE"))].index
        noneu_meoh = n.links.loc[(n.links.carrier == "import shipping-meoh") & (n.links.index.str.contains("DE"))].index
        noneu_ftfuel = n.links.loc[(n.links.carrier == "import shipping-ftfuel") & (n.links.index.str.contains("DE"))].index

    # indices: eu import
    h2_in = n.links.index[(n.links.carrier.str.contains("H2 pipeline")) & 
                          (n.links.bus0.str[:2] != "DE") &
                          (n.links.bus1.str[:2] == "DE")]
    h2_out = n.links.index[(n.links.carrier.str.contains("H2 pipeline")) & 
                           (n.links.bus0.str[:2] == "DE") &
                           (n.links.bus1.str[:2] != "DE")]
    elec_links_in = n.links.index[((n.links.carrier == "DC") | (n.links.carrier == "AC")) 
                                  & (n.links.bus0.str[:2] != "DE") 
                                  & (n.links.bus1.str[:2] == "DE")]
    elec_links_out = n.links.index[((n.links.carrier == "DC") | (n.links.carrier == "AC")) 
                                   & (n.links.bus0.str[:2] == "DE") 
                                   & (n.links.bus1.str[:2] != "DE")]
    eu_lines_in = n.lines.index[(n.lines.carrier == "AC") 
                                  & (n.lines.bus0.str[:2] != "DE") 
                                  & (n.lines.bus1.str[:2] == "DE")]
    eu_lines_out = n.lines.index[(n.lines.carrier == "AC") 
                                   & (n.lines.bus0.str[:2] == "DE") 
                                   & (n.lines.bus1.str[:2] != "DE")]

    # empty dataframe
    import_volumes = pd.DataFrame()
    ### non-eu import volumes 
    if non_eu_import["enable"]:
        import_volumes.loc["Fischer-Tropsch non EU", year] = n.links_t.p0[noneu_ftfuel].multiply(weights, axis=0).sum().sum()
        import_volumes.loc["hydrogen non EU", year] = n.links_t.p0[noneu_h2].multiply(weights, axis=0).sum().sum()
        import_volumes.loc["hbi non EU", year] = -n.links_t.p1[noneu_hbi].multiply(weights, axis=0).sum().sum() * 2.1
        import_volumes.loc["ammonia non EU", year] = n.links_t.p0[noneu_nh3].multiply(weights, axis=0).sum().sum()
        import_volumes.loc["methanol non EU", year] = n.links_t.p0[noneu_meoh].multiply(weights, axis=0).sum().sum()
        
        
    ###
    if relocation and "ammonia" in relocation:
        import_volumes.loc["ammonia EU in", year] = n.links_t.p0["EU NH3 -> DE NH3"].multiply(weights, axis=0).sum().sum()
        import_volumes.loc["ammonia EU out", year] = 0 # -n.links_t.p0["DE NH3 -> EU NH3"].multiply(weights, axis=0).sum().sum()
    else:
        import_volumes.loc["ammonia EU in", year] = 0
        import_volumes.loc["ammonia EU out", year] = 0
    if relocation and "methanol" in relocation:
        import_volumes.loc["methanol EU in", year] = n.links_t.p0["EU methanol -> DE methanol"].multiply(weights, axis=0).sum().sum()
        import_volumes.loc["methanol EU out", year] = 0 # -n.links_t.p0["DE methanol -> EU methanol"].multiply(weights, axis=0).sum().sum()
    else:
        ship_meoh = n.links[n.links.carrier=="shipping methanol for bunkers"].index
        import_volumes.loc["methanol EU in", year] = n.links_t.p0[ship_meoh].multiply(weights, axis=0).sum().sum()
        import_volumes.loc["methanol EU out", year] = 0
    if relocation and "hbi" in relocation:
        import_volumes.loc["hbi EU in", year] = n.links_t.p0["EU hbi -> DE hbi"].multiply(weights, axis=0).sum().sum() * 2.1
        import_volumes.loc["hbi EU out", year] = 0 # -n.links_t.p0["DE hbi -> EU hbi"].multiply(weights, axis=0).sum().sum() * 2.1
    else:
        import_volumes.loc["hbi EU in", year] = 0
        import_volumes.loc["hbi EU out", year] = 0

    import_volumes.loc["Fischer-Tropsch EU in", year] = n.links_t.p0["EU renewable oil -> DE oil"].multiply(weights, axis=0).sum().sum()
    import_volumes.loc["Fischer-Tropsch EU out", year] = -n.links_t.p0["DE renewable oil -> EU oil"].multiply(weights, axis=0).sum().sum()
    import_volumes.loc["hydrogen EU in", year] = (n.links_t.p0[h2_in].multiply(weights, axis=0).sum(axis=1) - n.links_t.p0[h2_out].multiply(weights, axis=0).sum(axis=1)).clip(lower=0).sum()
    import_volumes.loc["hydrogen EU out", year] = (n.links_t.p0[h2_in].multiply(weights, axis=0).sum(axis=1) - n.links_t.p0[h2_out].multiply(weights, axis=0).sum(axis=1)).clip(upper=0).sum()
    import_volumes.loc["electricity EU in", year] = n.links_t.p0[elec_links_in].multiply(weights, axis=0).sum().sum() + (n.lines_t.p0[eu_lines_in].multiply(weights, axis=0).sum(axis=1) - n.lines_t.p0[eu_lines_out].multiply(weights, axis=0).sum(axis=1)).clip(lower=0).sum()
    import_volumes.loc["electricity EU out", year] = -n.links_t.p0[elec_links_out].multiply(weights, axis=0).sum().sum() + (n.lines_t.p0[eu_lines_in].multiply(weights, axis=0).sum(axis=1) - n.lines_t.p0[eu_lines_out].multiply(weights, axis=0).sum(axis=1)).clip(upper=0).sum()
    import_volumes.loc["fossil gas"] = n.generators_t.p[n.generators[n.generators.index.str.contains("DE gas")].index].multiply(weights, axis=0).sum().sum()
    import_volumes.loc["fossil oil"] = n.generators_t.p["DE oil primary"].multiply(weights, axis=0).sum().sum()

    return import_volumes


def adjust_share(df, non_eu_carriers):
    # hydrogen
    if "pipeline-h2" in non_eu_carriers:
        h2_stat = n.statistics.supply(bus_carrier="H2").droplevel(0)
        non_eu = h2_stat[["import infrastructure pipeline-h2"]].sum()
        eu = h2_stat["H2 Electrolysis"]
        non_eu_share = non_eu / (non_eu + eu)
        plus = non_eu_share * df.loc["hydrogen EU in"]
        logger.info(f"Moving {plus.loc["2050"]/1e6} TWh of H2 from European to non-European origin")
        # subtract non-European H2 from European H2
        df.loc["hydrogen EU in"] -= plus.loc["2050"]
        # add non-European H2 to non-Eureopan H2
        df.loc["hydrogen non EU"] += plus.loc["2050"]
    return df


def plot_import_volumes(networks, colors, years, scenario, savepath):
    
    import_volumes = pd.DataFrame()
    non_eu_import = snakemake.params.non_eu_import
    relocation = snakemake.params.relocation
    # get import volumes
    for i, year in enumerate(years):
        data = get_import_volumes(networks[i], year, non_eu_import, relocation)
        import_volumes = import_volumes.reindex(import_volumes.index.union(data.index))
        data = adjust_share(data, non_eu_import["carriers"])
        import_volumes[year] = data

    import_volumes.fillna(0, inplace=True)
    import_volumes.to_csv(snakemake.output.import_vols)

    # plt.rcParams.update({'font.size': 14})  # Adjust the size as needed
    fig, ax = plt.subplots(figsize=(15, len(years)*2), sharey=True)

    bar = import_volumes.reindex(order).T.div(1e6).plot.barh(stacked=True, ax=ax, width=0.8, color=color_palette, title=scenario, legend=False)

    ax.grid(axis='x')

    ax.set_xlim(-200, 800)
    ax.set_xlabel(r"export $\leftarrow$ TWh $\rightarrow$ import                                                                                                                                                        ")

    colors = list(color_palette.values())
    labels = list(color_palette.keys())

    legend_handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]

    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.95, 0.5), ncol=1)

    plt.subplots_adjust(bottom=0.15)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def get_industrial_demand(i):
    # import ratios [MWh/t_Material]
    fn = snakemake.input.industry_sector_ratios[i]
    sector_ratios = pd.read_csv(fn, header=[0, 1], index_col=0)

    # material demand per node and industry [kt/a]
    fn = snakemake.input.industrial_production[i]
    nodal_production = pd.read_csv(fn, index_col=0) / 1e3  # kt/a -> Mt/a

    nodal_sector_ratios = pd.concat(
        {node: sector_ratios[node[:2]] for node in nodal_production.index}, axis=1
    )

    nodal_production_stacked = nodal_production.stack()
    nodal_production_stacked.index.names = [None, None]

    # final energy consumption per node and industry (TWh/a)
    nodal_df = (nodal_sector_ratios.multiply(nodal_production_stacked)).T

    return nodal_df


def consumer_costs(networks, colors, years, scenario, savepath):

    carrier = networks[0].loads.carrier.unique()
    rev_de = pd.DataFrame(index=years, columns=carrier)

    legend = ['industry methanol', 'NH3', 'steel', "ethylene for industry"]

    for i, n in enumerate(networks):
        revenue = n.statistics.revenue(groupby=["carrier", "bus"]).loc["Load", :, :]
        revenue_de = revenue[revenue.index.get_level_values("bus").str.startswith("DE")]
        revenue_de = revenue_de.groupby("carrier").sum()
        revenue_de = revenue_de.reindex(carrier, fill_value=0)
        rev_de.iloc[i, :] = revenue_de

    fig, ax = plt.subplots(2, 1, figsize=(len(years)*4, 10))

    rev_de.abs().div(1e9).plot.bar(ax=ax[0], stacked=True, width=0.8, legend=True, color=[colors[col] for col in rev_de.columns])
    ax[0].legend(loc="upper center", bbox_to_anchor=(1.5, 1))
    ax[0].set_title(f"Consumer Costs DE {scenario}")
    ax[0].set_ylabel("Bill €")

    rev_de[legend].abs().div(1e9).plot.bar(ax=ax[1], stacked=True, width=0.8, legend=False, color=[colors[col] for col in legend])
    ax[1].set_title("Consumer Costs Industry")
    ax[1].set_ylabel("Bill €")

    rev_de.abs().to_csv(snakemake.output.revenue)

    plt.subplots_adjust(bottom=0.15)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_steel_map(networks, regions, proj, years, scenario, savepath):
    
    fig, ax = plt.subplots(2, len(years), figsize=(len(years)*5, 8))

    for i, n in enumerate(networks):
        if snakemake.config["foresight"] == "overnight":
            axis = ax
        else:
            axis = ax[:, i]
        assign_locations(n)
        # steel production is only for EAF endogenous
        steel_list = n.statistics.energy_balance(bus_carrier="AC", groupby=pypsa.statistics.groupers["bus", "carrier"])
        if "EAF" not in steel_list.index.get_level_values("carrier").unique():
            steel_list = 0
        else:
            steel_list = steel_list.xs("EAF", level="carrier").xs("Link", level="component")
            steel_list = steel_list.reindex(regions.index, level="bus").abs().div(0.6395) # divide by MWh_el/t_steel to get tons of steel
            steel_list = steel_list.fillna(0)
            steel_list = steel_list.values
        regions["Steel"] = steel_list
        regions["Steel"] = regions["Steel"].div(1e6)  # Mt
        regions["Steel"] = regions.groupby("country")["Steel"].transform("sum")
        regions = regions.to_crs(proj.proj4_init)
        regions.plot(
            ax=axis[0],
            column="Steel",
            cmap="viridis",
            linewidths=0,
            # vmax=40,
            # vmin=0,
            legend=True,
            legend_kwds={
                "label": "Steel Production [Mt]",
                "shrink": 0.7,
                "extend": "max",
            },
        )
        axis[0].set_xticks([])
        axis[0].set_yticks([])
        axis[0].set_facecolor("white")
        axis[0].spines[:].set_visible(True)
        axis[0].spines["top"].set_color("black")
        axis[0].spines["right"].set_color("black")
        axis[0].spines["bottom"].set_color("black")
        axis[0].spines["left"].set_color("black")

        axis[0].set_title(years[i])

        # hbi production is only for DRI + EAF endogenous
        dri_list = n.statistics.energy_balance(bus_carrier="AC", groupby=pypsa.statistics.groupers["bus", "carrier"])
        if "DRI" not in dri_list.index.get_level_values("carrier").unique():
            dri_list = 0
        else:
            dri_list = dri_list.xs("DRI", level="carrier").xs("Link", level="component")
            dri_list = dri_list.reindex(regions.index, level="bus").abs().div(0.97087) # divide by MWh_el/t_steel to get tons of steel
            dri_list = dri_list.fillna(0)
            dri_list = dri_list.values
        regions["DRI"] = dri_list
        regions["DRI"] = regions["DRI"].div(1e6)  # Mt
        regions["DRI"] = regions.groupby("country")["DRI"].transform("sum")
        regions = regions.to_crs(proj.proj4_init)
        regions.plot(
            ax=axis[1],
            column="DRI",
            cmap="viridis",
            linewidths=0,
            # vmax=40,
            # vmin=0,
            legend=True,
            legend_kwds={
                "label": "DRI Production [Mt]",
                "shrink": 0.7,
                "extend": "max",
            },
        )
        axis[1].set_xticks([])
        axis[1].set_yticks([])
        axis[1].set_facecolor("white")
        axis[1].spines[:].set_visible(True)
        axis[1].spines["top"].set_color("black")
        axis[1].spines["right"].set_color("black")
        axis[1].spines["bottom"].set_color("black")
        axis[1].spines["left"].set_color("black")

        axis[1].set_title(years[i])
    # save figure
    fig.text(0.5, 0.92, f"Production Volume {scenario}", ha='center', fontsize=16)
    fig.savefig(savepath + "/steel_map.png", bbox_inches="tight")
    plt.close()


def plot_ammonia_map(networks, regions, proj, years, scenario, savepath):

    fig, ax = plt.subplots(2, len(years), figsize=(len(years)*5, 9))

    for i, n in enumerate(networks):
        if snakemake.config["foresight"] == "overnight":
            axis = ax
        else:
            axis = ax[:, i]
        assign_locations(n)
        ammonia_list = n.statistics.energy_balance(bus_carrier="AC", groupby=["bus", "carrier"])
        ammonia_list = ammonia_list.xs("Haber-Bosch", level="carrier").xs("Link", level="component")
        ammonia_list = ammonia_list.reindex(regions.index, level="bus").abs().div(0.2473) # divide by MWh_el/MWh_NH3 to get MWh ammonia
        ammonia_list = ammonia_list.fillna(0)
        regions["NH3"] = ammonia_list.values
        regions["NH3"] = regions["NH3"].div(1e6)  # TWh
        regions["NH3"] = regions.groupby("country")["NH3"].transform("sum")

        regions = regions.to_crs(proj.proj4_init)

        regions.plot(
            ax=axis[0],
            column="NH3",
            cmap="cividis",
            linewidths=0,
            # vmax=15,
            # vmin=0,
            legend=True,
            legend_kwds={
                "label": "Ammonia Production [TWh]",
                "shrink": 0.7,
                "extend": "max",
            },
        )
        axis[0].set_xticks([])
        axis[0].set_yticks([])
        axis[0].set_facecolor("white")
        axis[0].spines[:].set_visible(True)
        axis[0].spines["top"].set_color("black")
        axis[0].spines["right"].set_color("black")
        axis[0].spines["bottom"].set_color("black")
        axis[0].spines["left"].set_color("black")

        axis[0].set_title(years[i])

        hb_links = n.links[n.links.carrier=="Haber-Bosch"].index
        hb_cap = n.links.loc[hb_links].groupby(n.links.bus0).sum().p_nom_opt
        # reindex to match regions index
        hb_cap = hb_cap.reindex(regions.index, level="bus")
        # do I have to divide/multiply with an efficiency?
        regions["Haber-Bosch"] = hb_cap
        regions["Haber-Bosch"] = regions.groupby("country")["Haber-Bosch"].transform("sum")
        regions = regions.to_crs(proj.proj4_init)

        regions.plot(
            ax=axis[1],
            column="Haber-Bosch",
            cmap="cividis",
            linewidths=0,
            # vmax=15,
            # vmin=0,
            legend=True,
            legend_kwds={
                "label": "Haber Bosch Capacity [MW]",
                "shrink": 0.7,
                "extend": "max",
            },
        )
        axis[1].set_xticks([])
        axis[1].set_yticks([])
        axis[1].set_facecolor("white")
        axis[1].spines[:].set_visible(True)
        axis[1].spines["top"].set_color("black")
        axis[1].spines["right"].set_color("black")
        axis[1].spines["bottom"].set_color("black")
        axis[1].spines["left"].set_color("black")

        axis[1].set_title(years[i])

    fig.text(0.5, 0.92, f"Production Volume {scenario}", ha='center', fontsize=16)
    fig.text(0.5, 0.50, "Haber-Bosch Installed Capacity", ha='center', fontsize=16)
    fig.savefig(savepath + "/ammonia_map.png", bbox_inches="tight")
    plt.close()


def plot_methanol_map(networks, regions, proj, years, scenario, savepath):

    fig, ax = plt.subplots(1, len(years), figsize=(len(years)*5, 5))

    for i, n in enumerate(networks):
        if snakemake.config["foresight"] == "overnight":
            axis = ax
        else:
            axis = ax[i]
        assign_locations(n)
        if not "DE methanol" in n.buses.index:
            nodal_balance = (
                n.statistics.energy_balance(
                    aggregate_time=False,
                    nice_names=False,
                    groupby=pypsa.statistics.groupers["bus", "carrier", "bus_carrier"],
                )
                .loc[:, :, :, :]
            )
            mask = nodal_balance.index.get_level_values("bus_carrier").isin(["methanol"])
            nb = nodal_balance[mask].loc["Link", :, :, :].droplevel("bus_carrier").sum(axis=1)
            meoh = nb[nb>0].groupby("bus").sum()
            meoh.index = meoh.index.str.replace(" methanol", "", regex=False)
            regions["MeOH"] = meoh.div(1e6).mul(3)
        else:
            # biomass routes
            eff_bio_meoh = n.links[n.links.carrier=="biomass-to-methanol"].efficiency.mean()
            bio_meoh = n.statistics.energy_balance(bus_carrier="solid biomass", groupby=pypsa.statistics.groupers["bus", "carrier"])
            bio_meoh_cc = bio_meoh.xs("biomass-to-methanol CC", level="carrier").xs("Link", level="component")
            bio_meoh = bio_meoh.xs("biomass-to-methanol", level="carrier").xs("Link", level="component")
            bio_meoh.index = bio_meoh.index.str.replace(" solid biomass", "", regex=False)
            bio_meoh = bio_meoh.reindex(regions.index, level="bus").abs().div(eff_bio_meoh)
            bio_meoh = bio_meoh.fillna(0)
            regions["BioMeOH"] = bio_meoh.values

            bio_meoh_cc.index = bio_meoh_cc.index.str.replace(" solid biomass CC", "", regex=False)
            bio_meoh_cc = bio_meoh_cc.reindex(regions.index, level="bus").abs().div(eff_bio_meoh)
            bio_meoh_cc = bio_meoh_cc.fillna(0)
            regions["BioMeOHCC"] = bio_meoh_cc.values

            # methanolisation
            eff_h2 = n.links[n.links.carrier=="methanolisation"].efficiency.mean()
            H2_route = n.statistics.energy_balance(bus_carrier="H2", groupby=pypsa.statistics.groupers["bus", "carrier"])
            methanolisation = H2_route.xs("methanolisation", level="carrier").xs("Link", level="component")
            methanolisation.index = methanolisation.index.str.replace(" H2", "", regex=False)
            methanolisation = methanolisation.reindex(regions.index, level="bus").abs().mul(eff_h2)
            methanolisation = methanolisation.fillna(0)
            regions["Methanolisation"] = methanolisation.values

            regions["MeOH"] = regions["BioMeOH"] + regions["BioMeOHCC"] + regions["Methanolisation"]
            regions["MeOH"] = regions["MeOH"].div(1e6)

        regions["MeOH"] = regions.groupby("country")["MeOH"].transform("sum")
        regions = regions.to_crs(proj.proj4_init)
        regions.plot(
            ax=axis,
            column="MeOH",
            cmap="cividis",
            linewidths=0,
            # vmax=175,
            # vmin=0,
            legend=True,
            legend_kwds={
                "label": "Methanol Production [TWh]",
                "shrink": 0.7,
                "extend": "max",
            },
        )

        axis.set_title(years[i])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_facecolor("white")
        axis.spines[:].set_visible(True)
        axis.spines["top"].set_color("black")
        axis.spines["right"].set_color("black")
        axis.spines["bottom"].set_color("black")
        axis.spines["left"].set_color("black")

        axis.set_title(years[i])
    
    fig.text(0.5, 0.92, f"MeOH Production Volume {scenario}", ha='center', fontsize=16)
    fig.savefig(savepath + "/methanol_map.png", bbox_inches="tight")
    plt.close()


def colormaps(networks, regions, years, scenario, savepath):

    snakemake.params.plotting["projection"] = {"name": "EqualEarth"}
    proj = load_projection(snakemake.params.plotting)

    plot_steel_map(networks, regions, proj, years, scenario, savepath)
    
    plot_ammonia_map(networks, regions, proj, years, scenario, savepath)

    plot_methanol_map(networks, regions, proj, years, scenario, savepath)


def plot_prices(networks, regions, years, scenario, relocation, savepath):

    snakemake.params.plotting["projection"] = {"name": "EqualEarth"}
    proj = load_projection(snakemake.params.plotting)
    reg_DE = regions[regions.index.str[:2] == "DE"].index
    reg_EU = regions[regions.index.str[:2] != "DE"].index
    for carrier in ["steel", "NH3", "methanol", "H2", "co2 stored"]:
        # initialize figure
        fig, ax = plt.subplots(1, len(years), figsize=(len(years)*5, 5))
        for i, n in enumerate(networks):
            if snakemake.config["foresight"] == "overnight":
                axis = ax
            else:
                axis = ax[i]
            ind = n.buses[n.buses.carrier == carrier].index
            marginal_prices = n.buses_t.marginal_price[ind].mean().clip(lower=0)
            if len(marginal_prices) == 2:
                regions[carrier] = marginal_prices.loc[f"EU {carrier}"]
                regions.loc[reg_EU, carrier] = marginal_prices.loc[f"DE {carrier}"]
            else:
                regions[carrier] = marginal_prices.values
            
            regions = regions.to_crs(proj.proj4_init)
            unit = "€/t" if carrier=="steel" else "€/MWh"
            regions.plot(
                ax=axis,
                column=carrier,
                cmap="cividis",
                linewidths=0,
                # vmin=64,
                # vmax=65,
                legend=True,
                legend_kwds={
                    "label": f"{carrier} price [{unit}]",
                    "shrink": 0.7,
                    "extend": "max",
                },
            )

            axis.set_title(years[i])
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_facecolor("white")
            axis.spines[:].set_visible(True)
            axis.spines["top"].set_color("black")
            axis.spines["right"].set_color("black")
            axis.spines["bottom"].set_color("black")
            axis.spines["left"].set_color("black")
        fig.text(0.5, 0.92, f"{carrier} price {scenario}", ha='center', fontsize=16)
        fig.savefig(savepath + f"/{carrier}_price.png", bbox_inches="tight")
        plt.close()

    emission_price = -n.global_constraints.loc["CO2Limit", "mu"] - (n.global_constraints.loc["co2_limit-DE", "mu"] / 1e6)
    data = pd.DataFrame(index=["HBI", "NH3", "methanol"], columns=["mean", "min", "max"])
    # make it demand weighted
    # demand weighted price steel
    buses = n.buses[(n.buses.carrier=="hbi") & (n.buses.index.str[:2]=="DE")].index
    buses_steel = n.buses[(n.buses.carrier=="steel") & (n.buses.index.str[:2]=="DE")].index
    load = n.loads_t.p[buses_steel]
    if len(buses)==1:
        hbi = n.buses_t.marginal_price[buses].mean()["DE hbi"]
    else:
        hbi = n.buses_t.marginal_price[buses].mul(load.values).sum() / load.sum().values
    data.loc["HBI", "mean"] = hbi.mean()
    data.loc["HBI", "min"] = hbi.min()
    data.loc["HBI", "max"] = hbi.max()
    # demand weighted price ammonia
    buses = n.buses[(n.buses.carrier=="NH3") & (n.buses.index.str[:2]=="DE")].index
    load = n.loads_t.p[buses]
    ammonia = n.buses_t.marginal_price[buses].mul(load).sum() / load.sum()
    data.loc["NH3", "mean"] = ammonia.mean()
    data.loc["NH3", "min"] = ammonia.min()
    data.loc["NH3", "max"] = ammonia.max()
    # demand weighted price meoh
    buses = n.buses[(n.buses.carrier=="methanol") & (n.buses.index.str[:2]=="DE")].index
    load_ind = n.loads[(n.loads.carrier=="industry methanol") & (n.loads.index.str[:2]=="DE")].index
    load = n.loads_t.p[load_ind]
    load.columns = load.columns.str.replace("industry ", "", regex=False)
    if len(buses) == 1:
        load = load.sum(axis=1)
        load = pd.DataFrame(load, columns=["DE methanol"])
    # + co2 price
    methanol = n.buses_t.marginal_price[buses].mul(load).sum() / load.sum() + 0.248*emission_price
    data.loc["methanol", "mean"] = methanol.mean()
    data.loc["methanol", "min"] = methanol.min()
    data.loc["methanol", "max"] = methanol.max()
    # only renewable oil price
    # ethylene = n.buses_t.marginal_price["DE renewable oil"].mean() + 0.2571*emission_price
    # data.loc["ethylene for industry", "mean"] = ethylene
    # data.loc["ethylene for industry", "min"] = ethylene
    # data.loc["ethylene for industry", "max"] = ethylene
    data.to_csv(savepath + "/prices.csv")


def plot_non_eu_import(n, year, scenario, savepath):
    sw = n.snapshot_weightings.generators
    import_vol = pd.DataFrame(index=n.snapshots)
    for carrier in non_eu_carriers:
        ind = n.links[n.links.carrier==carrier].index
        import_vol[carrier] = n.links_t.p1[ind].mul(sw, axis=0).sum(axis=1)

    import_vol = import_vol.abs().div(1e3)
    import_vol = import_vol.loc[:, (import_vol != 0).any(axis=0)]
    import_vol = import_vol.resample("D").mean()

    fig, ax = plt.subplots(figsize=(16, 8))
    import_vol.plot.area(ax=ax, stacked=True, linewidth=0.0)
    ax.set_ylabel("GWh")
    ax.set_title(f"Daily average non EU import {year} {scenario}")
    ax.legend(loc="upper center", bbox_to_anchor=(1.15, 1))
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_non_eu_import_balance(networks, years, scenario, savepath):
    
    import_vol = pd.DataFrame(index=non_eu_carriers, columns=years)
    for i, n in enumerate(networks):
        sw = n.snapshot_weightings.generators
        for carrier in non_eu_carriers:
            ind = n.links[n.links.carrier==carrier].index
            import_vol.loc[carrier, years[i]] = n.links_t.p1[ind].mul(sw, axis=0).sum(axis=1).sum()

        import_vol.loc["fossil oil", years[i]] = n.links_t.p1[['EU oil refining', 'DE oil refining']].mul(sw, axis=0).sum(axis=1).sum()

        import_vol.loc["fossil gas", years[i]] = n.links_t.p1[['EU gas compressing', 'DE gas compressing']].mul(sw, axis=0).sum(axis=1).sum()

    import_vol = import_vol.abs().div(1e6) # .round(2) # .sort_values()
    import_vol = import_vol.loc[:, (import_vol != 0).any(axis=0)]
    import_vol.loc["import shipping-hbi"] *= 2.1

    fig, ax = plt.subplots(figsize=(15, len(years)*2), sharey=True)

    import_vol.T.plot.barh(stacked=True, ax=ax, width=0.8, title=scenario, legend=True)

    ax.grid(axis='y')
    ax.legend(loc="upper center", bbox_to_anchor=(1.15, 1))
    ax.set_xlabel("TWh")
    ax.set_title(f"Non EU Import to EU {scenario}")

    plt.subplots_adjust(bottom=0.15)
    fig.savefig(savepath + "/non_eu_import_balance.png", bbox_inches="tight")
    plt.close()
    import_vol.to_csv(savepath + "/non_eu_import.csv")


def plot_hydrogen_carbon_map(n, regions, year, scenario, savepath):
    # set up figure
    fig, ax = plt.subplots(2, 1, figsize=(10, 10), subplot_kw={"projection": ccrs.EqualEarth()}, layout="constrained")
    legend_kwargs = {
        "frameon": False,
        "framealpha": 1,
        "edgecolor": "None",
    }
    # plot hydrogen network + demand/supply + storage
    carriers = ["H2"]
    df = n.statistics.energy_balance(bus_carrier=carriers, groupby=["bus", "carrier"], aggregate_time="mean")
    # drop transmission links from energy balance
    transmission_carriers = ["H2 pipeline", "H2 pipeline retrofitted", "H2 pipeline (Kernnetz)"]
    sub = df.loc[["Link"]].drop(transmission_carriers, level=2, errors="ignore")
    # group for each bus reagion
    df = pd.concat([df.drop("Link"), sub])
    df = df.rename(lambda x: x.replace(" CC", ""), level=2)
    df = df.groupby(level=[1, 2]).sum().rename(n.buses.location, level=0)
    df = df[df.abs() > 1]
    # consolidate H2 CHP and H2 OCGT

    # plotting specs
    bus_scale = 5e-5
    branch_scale = 1e-3
    flow_scale = 4e-1
    unit = "GW"
    conversion = 1e3
    # TONI TODO: Aggregate H2 pipelines!
    # TONI TODO: Aggregate DE nodes
    # Make Map plot for Germany only
    bus_sizes = df.sort_index()
    flow = n.statistics.transmission(
        groupby=False, bus_carrier=carriers, aggregate_time="mean"
    )
    branch_colors = {carrier: colors[carrier] for carrier in transmission_carriers}
    fallback = pd.Series()
    line_widths = (
        (
            flow.get("Line", fallback).abs().reindex(n.lines.index, fill_value=0)
            * branch_scale
        )
        .astype(float)
        .round(2)
    )
    link_widths = (
        (
            flow.get("Link", fallback).abs().reindex(n.links.index, fill_value=0)
            * branch_scale
        )
        .astype(float)
        .round(2)
    )
    n.plot.map(
        bus_sizes=bus_sizes * bus_scale,
        bus_colors=colors,
        bus_split_circles=True,
        line_widths=line_widths,
        link_widths=link_widths,
        line_colors=branch_colors.get("Line", "lightgrey"),
        link_colors=branch_colors.get("Link", "lightgrey"),
        flow=flow * flow_scale,
        ax=ax[0],
        margin=0.2,
        color_geomap={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        geomap_resolution="10m",
    )

    # Add lat/lon gridlines
    gl = ax[0].gridlines(
        draw_labels=True, linewidth=0.1, color="gray", alpha=0.3, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER

    buses = n.buses.query("carrier in @carriers").index
    price = (
        n.buses_t.marginal_price.mean()
        .reindex(buses)
        .rename(n.buses.location)
        .groupby(level=0)
        .mean()
    )

    regions["price"] = price.reindex(regions.index).fillna(0)
    region_unit = "€/MWh"
    cmap = "Blues"
    vmin, vmax = 65, 90

    regions.plot(
        ax=ax[0],
        column="price",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor=None,
        linewidth=0,
        transform=ccrs.PlateCarree(),
        aspect="equal",
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    cbr = fig.colorbar(
        sm,
        ax=ax[0],
        label=f"Average Marginal Price of Hydrogen [{region_unit}]",
        shrink=0.6,
        pad=0.05,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")

    prod_carriers = (
        bus_sizes[bus_sizes > 0].index.unique("carrier").sort_values()
    )
    cons_carriers = (
        bus_sizes[bus_sizes < 0]
        .index.unique("carrier")
        .difference(prod_carriers)
        .sort_values()
    )

    # fix bug related to falsely clipped normalization
    if "H$_2$ For Industry" in prod_carriers:
        prod_carriers = prod_carriers.difference(["H2 for industry"])
        cons_carriers = cons_carriers.union(["H2 for industry"])
    wrapper = textwrap.TextWrapper(width=18)
    add_legend_patches(
        ax[0],
        [colors[c] for c in prod_carriers],
        prod_carriers.map(wrapper.fill),
        legend_kw={
            "loc": "upper left",
            "bbox_to_anchor": (1, 1),
            "title": "Production",
            **legend_kwargs,
        },
    )
    colors["H2 CHP"] = colors["urban central H2 CHP"]
    new = cons_carriers.drop(["urban central H2 CHP", "urban central H2 retrofit CHP"], errors="ignore")
    new = new.append(pd.Index(["H2 CHP"]))
    add_legend_patches(
        ax[0],
        [colors[c] for c in new],
        new.map(
            wrapper.fill
        ),
        legend_kw={
            "loc": "upper left",
            "bbox_to_anchor": (1, 0.8),
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    # only use one legend for both branches and flows
    legend_bus_sizes = [50, 10]
    add_legend_circles(
        ax[0],
        [s * bus_scale * conversion for s in legend_bus_sizes],
        [f"{s} {unit}" for s in legend_bus_sizes],
        legend_kw={
            "loc": "lower left",
            "bbox_to_anchor": (1, 0.05),
            **legend_kwargs,
        },
    )
    # only use one legend for both branches and flows
    legend_branch_sizes = [20, 10]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax[0],
            [s * branch_scale * conversion for s in legend_branch_sizes],
            [f"{s} {unit}" for s in legend_branch_sizes],
            legend_kw={
                "loc": "lower left",
                "bbox_to_anchor": (1, -0.05),
                **legend_kwargs,
            },
        )

    title = f"Hydrogen Balance {year} ({scenario})"
    ax[0].set_title(title)

    # Carbon
    colors["biomass-to-methanol"] = colors["biomass to liquid"]
    colors["waste CHP"] = colors["urban central solid biomass CHP"]
    carriers = ["co2 stored", "co2 sequestered"]
    df = n.statistics.energy_balance(bus_carrier=carriers, groupby=["bus", "carrier"], aggregate_time="mean")
    transmission_carriers = ["CO2 pipeline"]
    sub = df.loc[["Link"]].drop(transmission_carriers, level=2, errors="ignore")
    df = pd.concat([df.drop("Link"), sub])
    df = df.rename(lambda x: x.replace(" CC", ""), level=2)
    df = df.groupby(level=[1, 2]).sum().rename(n.buses.location, level=0)
    df = df[df.abs() > 1]

    # plot adjustments
    bus_scale = 2e-4
    branch_scale = 6e-4
    flow_scale = 8e-1
    unit = "kt/h"
    conversion = 1e3

    bus_sizes = df.sort_index()
    flow = n.statistics.transmission(groupby=False, bus_carrier=carriers, aggregate_time="mean")
    branch_colors = {carrier: colors[carrier] for carrier in transmission_carriers}
    fallback = pd.Series()

    sequestration_sizes = -bus_sizes.loc[:, ["co2 sequestered"]] / 2
    bus_sizes = bus_sizes.drop("co2 sequestered", level=1)
    n.plot.map(
        bus_sizes=sequestration_sizes * bus_scale,
        bus_colors=colors,
        line_widths=0,
        link_widths=0,
        ax=ax[1],
        color_geomap=False,
        geomap=True,
    )
    regions.plot(
        ax=ax[1],
        facecolor="None",
        edgecolor="darkgrey",
        linewidth=0,
        transform=ccrs.PlateCarree(),
        aspect="equal",
    )

    line_widths = (
        (
            flow.get("Line", fallback).abs().reindex(n.lines.index, fill_value=0)
            * branch_scale
        )
        .astype(float)
        .round(2)
    )
    link_widths = (
        (
            flow.get("Link", fallback).abs().reindex(n.links.index, fill_value=0)
            * branch_scale
        )
        .astype(float)
        .round(2)
    )

    n.plot.map(
        bus_sizes=bus_sizes * bus_scale,
        bus_colors=colors,
        bus_split_circles=True,
        line_widths=line_widths,
        link_widths=link_widths,
        line_colors=branch_colors.get("Line", "lightgrey"),
        link_colors=branch_colors.get("Link", "lightgrey"),
        flow=flow * flow_scale,
        ax=ax[1],
        margin=0.2,
        color_geomap={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        geomap_resolution="10m",
    )

    # Add lat/lon gridlines
    gl = ax[1].gridlines(
        draw_labels=True, linewidth=0.1, color="gray", alpha=0.3, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.xformatter = cartopy.mpl.gridliner.LONGITUDE_FORMATTER
    gl.yformatter = cartopy.mpl.gridliner.LATITUDE_FORMATTER

    buses = n.buses.query("carrier in @carriers").index
    price = (
        n.buses_t.marginal_price.mean()
        .reindex(buses)
        .rename(n.buses.location)
        .groupby(level=0)
        .mean()
    )

    price = price - n.global_constraints.mu["CO2Limit"]

    regions["price"] = price.reindex(regions.index).fillna(0)
    region_unit = "€/t"
    cmap = "Purples"
    vmin, vmax = 100, 135

    regions.plot(
        ax=ax[1],
        column="price",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor=None,
        linewidth=0,
        transform=ccrs.PlateCarree(),
        aspect="equal",
    )

    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    title = "Capturing Carbon"
    cbr = fig.colorbar(
        sm,
        ax=ax[1],
        label=f"Average Marginal Price of {title} [{region_unit}]",
        shrink=0.6,
        pad=0.05,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")

    prod_carriers = (
        bus_sizes[bus_sizes > 0].index.unique("carrier").sort_values()
    )
    cons_carriers = (
        bus_sizes[bus_sizes < 0]
        .index.unique("carrier")
        .difference(prod_carriers)
        .sort_values()
    )

    # fix bug related to falsely clipped normalization
    cons_carriers = cons_carriers.union(["co2 sequestered"])
    colors["waste/biomass CHP"] = colors["urban central solid biomass CHP"]
    new = prod_carriers.drop(["biomass-to-methanol", "waste CHP", "urban central solid biomass CHP"], errors="ignore")
    new = new.append(pd.Index(["waste/biomass CHP"]))
    add_legend_patches(
        ax[1],
        [colors[c] for c in new],
        new.map(wrapper.fill),
        legend_kw={
            "loc": "upper left",
            "bbox_to_anchor": (1, 1.1),
            "title": "Production",
            **legend_kwargs,
        },
    )

    add_legend_patches(
        ax[1],
        [colors[c] for c in cons_carriers],
        cons_carriers.str.replace("Sequestration", "Seq.", regex=True).map(
            wrapper.fill
        ),
        legend_kw={
            "loc": "upper left",
            "bbox_to_anchor": (1, 0.5),
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    # only use one legend for both branches and flows
    legend_bus_sizes = [10, 5]
    add_legend_circles(
        ax[1],
        [s * bus_scale * conversion for s in legend_bus_sizes],
        [f"{s} {unit}" for s in legend_bus_sizes],
        legend_kw={
            "loc": "lower left",
            "bbox_to_anchor": (1, 0.07),
            **legend_kwargs,
        },
    )
    # only use one legend for both branches and flows
    legend_branch_sizes = [10, 5]
    if legend_branch_sizes is not None:
        add_legend_lines(
            ax[1],
            [s * branch_scale * conversion for s in legend_branch_sizes],
            [f"{s} {unit}" for s in legend_branch_sizes],
            legend_kw={
                "loc": "lower left",
                "bbox_to_anchor": (1, -0.1),
                **legend_kwargs,
            },
        )

    title = f"Carbon Balance {year} ({scenario})"
    ax[1].set_title(title)

    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def group_pipes(df, drop_direction=False):
    """
    Group pipes which connect same buses and return overall capacity.
    """
    df = df.copy()
    if drop_direction:
        positive_order = df.bus0 < df.bus1
        df_p = df[positive_order]
        swap_buses = {"bus0": "bus1", "bus1": "bus0"}
        df_n = df[~positive_order].rename(columns=swap_buses)
        df = pd.concat([df_p, df_n])

    # there are pipes for each investment period rename to AC buses name for plotting
    df["index_orig"] = df.index
    df.index = df.apply(
        lambda x: f"H2 pipeline {x.bus0.replace(' H2', '')} -> {x.bus1.replace(' H2', '')}",
        axis=1,
    )
    return df.groupby(level=0).agg(
        {"p_nom_opt": "sum", "bus0": "first", "bus1": "first", "index_orig": "first"}
    )


def plot_h2_map(n, regions, savepath, only_de=False):
    n = n.copy()
    logger.info("Plotting H2 map")
    logger.info("Assigning location")
    snakemake.params.plotting["projection"] = {"name": "EqualEarth"}
    assign_locations(n)
    map_opts = snakemake.params.plotting["map"]

    h2_storage = n.stores[n.stores.carrier.isin(["H2", "H2 Store"])]
    regions["H2"] = (
        h2_storage.rename(index=h2_storage.bus.map(n.buses.location))
        .e_nom_opt.groupby(level=0)
        .sum()
        .div(1e6)
    )  # TWh
    regions["H2"] = regions["H2"].where(regions["H2"] > 0.1)

    bus_size_factor = 1e5
    linewidth_factor = 4e3
    # MW below which not drawn
    line_lower_threshold = 0

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

    carriers = ["H2 Electrolysis", "H2 Fuel Cell"]

    elec = n.links[n.links.carrier.isin(carriers)].index

    bus_sizes = (
        n.links.loc[elec, "p_nom_opt"].groupby([n.links["bus0"], n.links.carrier]).sum()
        / bus_size_factor
    )

    # make a fake MultiIndex so that area is correct for legend
    bus_sizes.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
    # drop all links which are not H2 pipelines
    n.links.drop(
        n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
    )

    h2_new = n.links[n.links.carrier == "H2 pipeline"]
    h2_retro = n.links[n.links.carrier == "H2 pipeline retrofitted"]
    h2_kern = n.links[n.links.carrier == "H2 pipeline (Kernnetz)"]

    # sum capacitiy for pipelines from different investment periods
    logger.info("Grouping pipes")
    h2_new = group_pipes(h2_new)

    if not h2_retro.empty:
        h2_retro = (
            group_pipes(h2_retro, drop_direction=True).reindex(h2_new.index).fillna(0)
        )

    if not h2_kern.empty:
        h2_kern = (
            group_pipes(h2_kern, drop_direction=True).reindex(h2_new.index).fillna(0)
        )

    h2_total = n.links.p_nom_opt.groupby(level=0).sum()
    link_widths_total = h2_total / linewidth_factor

    # drop all reversed pipe
    n.links.drop(n.links.index[n.links.index.str.contains("reversed")], inplace=True)
    n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    n.links = n.links.groupby(level=0).agg(
        {
            **{
                col: "first" for col in n.links.columns if col != "p_nom_opt"
            },  # Take first value for all columns except 'p_nom_opt'
            "p_nom_opt": "sum",  # Sum values for 'p_nom_opt'
        }
    )
    link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    carriers_pipe = ["H2 pipeline", "H2 pipeline retrofitted", "H2 pipeline (Kernnetz)"]
    total = n.links.p_nom_opt.where(n.links.carrier.isin(carriers_pipe), other=0.0)

    retro = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline retrofitted", other=0.0
    )

    kern = n.links.p_nom_opt.where(
        n.links.carrier == "H2 pipeline (Kernnetz)", other=0.0
    )

    link_widths_total = total / linewidth_factor
    link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_retro = retro / linewidth_factor
    link_widths_retro[n.links.p_nom_opt < line_lower_threshold] = 0.0

    link_widths_kern = kern / linewidth_factor
    link_widths_kern[n.links.p_nom_opt < line_lower_threshold] = 0.0

    n.links.bus0 = n.links.bus0.str.replace(" H2", "")
    n.links.bus1 = n.links.bus1.str.replace(" H2", "")
    proj = load_projection(snakemake.params.plotting)
    regions = regions.to_crs(proj.proj4_init)
    logger.info("Plotting map")
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})

    color_h2_pipe = "#b3f3f4"
    color_retrofit = "#499a9c"
    color_kern = "#6b3161"

    bus_colors = {"H2 Electrolysis": "#ff29d9", "H2 Fuel Cell": "#805394"}

    n.plot.map(
        geomap=True,
        bus_sizes=bus_sizes,
        bus_colors=bus_colors,
        link_colors=color_h2_pipe,
        link_widths=link_widths_total,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot.map(
        geomap=True,
        bus_sizes=0,
        link_colors=color_retrofit,
        link_widths=link_widths_retro,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    n.plot.map(
        geomap=True,
        bus_sizes=0,
        link_colors=color_kern,
        link_widths=link_widths_kern,
        branch_components=["Link"],
        ax=ax,
        **map_opts,
    )

    regions.plot(
        ax=ax,
        column="H2",
        cmap="Blues",
        linewidths=0,
        legend=True,
        vmax=6,
        vmin=0,
        legend_kwds={
            "label": "H2 Storage [TWh]",
            "shrink": 0.7,
            "extend": "max",
        },
    )

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]
    logger.info("Adding legend")
    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_circles(
        ax,
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0.23, 1),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    pypsa.plot.maps.static.add_legend_lines(
        ax,
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = [bus_colors[c] for c in carriers] + [
        color_h2_pipe,
        color_retrofit,
        color_kern,
    ]

    labels = carriers + [
        "H2 pipeline (new)",
        "H2 pipeline (repurposed)",
        "H2 pipeline (Kernnetz)",
    ]

    # labels = [carriers_in_german.get(c, c) for c in labels]

    legend_kw = dict(
        loc="upper left",
        bbox_to_anchor=(0, 1.13),
        ncol=2,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_patches(ax, colors, labels, legend_kw=legend_kw)

    ax.set_facecolor("white")

    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_h2_pipeline_loading(n, tech_colors, scenario, savepath):
    n = n.copy()
    # get balance map plotting parameters
    bus_size_factor = 1e-2
    branch_width_factor = 1e-1

    eu_location = {"x": -5.5,"y": 46}
    fig_size = [5, 6.5]
    alpha = 1
    boundaries = [-11, 30, 34, 71]
    carrier = "H2"
    n.buses.loc["EU", ["x", "y"]] = eu_location["x"], eu_location["y"]
    conversion = 1e6 # MWh to TWh
    # for plotting change bus to location
    n.buses["location"] = n.buses["location"].replace("", "EU").fillna("EU")

    # set location of buses to EU if location is empty and set x and y coordinates to bus location
    # n.buses.loc[((n.buses["location"]=="")|n.buses["location"].isna()),"location"] = "EU"
    n.buses["x"] = n.buses.location.map(n.buses.x)
    n.buses["y"] = n.buses.location.map(n.buses.y)

    s = n.statistics
    s.set_parameters(round=3, drop_zero=True)
    grouper = ["name", "bus", "carrier"]

    # bus_sizes according to energy balance of bus carrier
    nice_names = True
    energy_balance_df = s.energy_balance(
        nice_names=nice_names, bus_carrier=carrier, groupby=grouper
    )
    # remove energy balance of transmission carriers, which are relate to losses
    transmission_carriers = get_transmission_carriers(n, bus_carrier=carrier).rename(
        {"name": "carrier"}
    )

    # TODO change in pypsa
    energy_balance_df.loc[transmission_carriers.unique(0)] = energy_balance_df.loc[
        transmission_carriers.unique(0)
    ].drop(index=transmission_carriers.unique(1), level="carrier")
    energy_balance_df = energy_balance_df.dropna()

    name_level_idx = energy_balance_df.index.names.index("name")
    energy_balance_df.index = energy_balance_df.index.set_levels(
        energy_balance_df.index.levels[name_level_idx].map(lambda x: x[:5]),
        level="name",
        verify_integrity=False,
    )
    # for production we can use the location of the sites, while for consumption we use the location of the buses which can be regionalised or not
    bus_sizes = energy_balance_df.groupby(level=["bus", "carrier"]).sum().div(conversion)
    colors = (
        bus_sizes.index.get_level_values("carrier")
        .unique()
        .to_series()
        .map(tech_colors)
    )

    bus_sizes = bus_sizes.loc[bus_sizes.abs() > 1]
    carrier = "H2"
    bus_sizes = bus_sizes.sort_values(ascending=False)

    # line and links widths according to optimal capacity
    flow = s.transmission(groupby=False, bus_carrier=carrier).div(conversion)

    if not flow.index.get_level_values(1).empty:
        flow_reversed_mask = flow.index.get_level_values(1).str.contains("reversed")
        flow_reversed = flow[flow_reversed_mask].rename(
            lambda x: x.replace("-reversed", "")
        )
        flow = flow[~flow_reversed_mask].subtract(flow_reversed, fill_value=0)

    # if there are not lines or links for the bus carrier, use fallback for plotting
    link_widths = flow.get("Link")
    link_widths = link_widths.abs().sort_values()

    # get average line loading of pipeline links
    link_loading = s.capacity_factor(comps="Link", groupby=False, bus_carrier=carrier)
    link_loading = link_loading.loc[link_loading.index.get_level_values(0).str.contains("pipeline")]
    link_loading = link_loading.groupby(lambda x: x.replace("-reversed", ""), axis=0).mean()

    fig, ax = plt.subplots(
        figsize=fig_size,
        subplot_kw={"projection": ccrs.EqualEarth()},
        layout="constrained",
    )

    # plot map
    n.plot.map(
        bus_sizes=bus_sizes * bus_size_factor,
        bus_colors=colors,
        bus_alpha=alpha,
        bus_split_circles=True,
        line_widths=0,
        link_widths=link_widths * branch_width_factor,
        link_colors=link_loading,
        link_cmap="viridis",
        ax=ax,
        margin=0.2,
        color_geomap={"border": "darkgrey", "coastline": "darkgrey"},
        geomap=True,
        geomap_resolution="10m",
        boundaries=boundaries,
    )

    # Add legend
    legend_kwargs = {
        "loc": "upper left",
        "frameon": True,
        "framealpha": 0.5,
        "edgecolor": "None",
    }

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    carrier = n.carriers.loc[carrier, "nice_name"]
    cbr = fig.colorbar(
        sm,
        ax=ax,
        label=f" Average Pipeline loading [-]",
        shrink=0.95,
        pad=0.03,
        aspect=50,
        orientation="horizontal",
    )
    cbr.outline.set_edgecolor("None")

    pad = 0.18
    prod_carriers = bus_sizes[bus_sizes > 0].index.unique("carrier").sort_values()
    cons_carriers = (
        bus_sizes[bus_sizes < 0]
        .index.unique("carrier")
        .difference(prod_carriers)
        .sort_values()
    )

    # Add production carriers
    add_legend_patches(
        ax,
        colors[prod_carriers],
        prod_carriers,
        patch_kw={"alpha": alpha},
        legend_kw={
            "bbox_to_anchor": (1, 1),# -pad),
            "ncol": 1,
            "title": "Production",
            **legend_kwargs,
        },
    )

    # Add consumption carriers
    add_legend_patches(
        ax,
        colors[cons_carriers],
        cons_carriers,
        patch_kw={"alpha": alpha},
        legend_kw={
            "bbox_to_anchor": (1, 0.8),# -pad),
            "ncol": 1,
            "title": "Consumption",
            **legend_kwargs,
        },
    )

    # TODO with config
    # Add bus legend
    legend_bus_sizes = [10, 50]
    carrier_unit = "TWh"
    if legend_bus_sizes is not None:
        add_legend_circles(
            ax,
            [s * bus_size_factor for s in legend_bus_sizes],
            [f"{s} {carrier_unit}" for s in legend_bus_sizes],
            legend_kw={
                "bbox_to_anchor": (0, 1),
                "title": "Supply/Demand",
                **legend_kwargs,
            },
        )

    legend_branch_sizes = [1, 10]
    if legend_branch_sizes:
        # Add branch legend
        if legend_branch_sizes is not None:
            add_legend_lines(
                ax,
                [s * branch_width_factor for s in legend_branch_sizes],
                [f"{s} {carrier_unit}" for s in legend_branch_sizes],
                legend_kw={"bbox_to_anchor": (0, 0.85), **legend_kwargs},
            )

    # Set geographic extent for Germany
    ax.set_extent([5.5, 15.5, 47, 56], crs=ccrs.PlateCarree())
    ax.set_title(f"Hydrogen Grid Loading 2050 {scenario}")
    fig.savefig(
        savepath,
        bbox_inches="tight",
    )
    plt.close()


def print_h2_FT_info(n):
    # get percentage for loading of hydrogen network
    de_pipe = n.links[(n.links.carrier.str.contains("H2 pipeline")) 
                & (n.links.bus0.str.contains("DE"))
                & (n.links.bus1.str.contains("DE"))
                ].index
    p_nom = n.links.p_nom_opt[de_pipe].sum()
    p = n.links_t.p0[de_pipe].mul(n.snapshot_weightings.generators, axis=0).sum().sum()
    loading = p/8760/p_nom * 100
    logger.info(f"Loading of German H2 Pipelines: {loading.round(2)} % ")

    # get share of domestically produced FT
    loads = n.loads[(n.loads.carrier.isin(["kerosene for aviation", "agriculture machinery oil"])) & (n.loads.bus.str.startswith("DE"))].index
    p = n.loads_t.p[loads].multiply(n.snapshot_weightings.generators, axis=0).sum().sum()
    domestic = n.links[(n.links.bus1=="DE renewable oil") & (n.links.bus0.str.startswith("DE"))].index
    domestic_FT = n.links_t.p1[domestic].multiply(n.snapshot_weightings.generators, axis=0).sum().sum()
    # share of domestic production
    logger.info(f"Domestic share of FT demand: {(-domestic_FT/p*100).round(2)} %")
    logger.info(f"Domestic share of FT compared to oil demand 2020: {(-domestic_FT/(931*1e6)).round(2)} %")


def print_RE_info(n):
    for carrier in ["offwind", "onwind", "solar"]:
        index = n.generators[(n.generators.carrier.str.contains(carrier)) &
                     (n.generators.index.str.startswith("DE"))].index
        p = n.generators.loc[index].p_nom_opt.sum() / 1e3
        logger.info(f"{carrier} capacity: {p.round(2)} GW")


def get_demand(n, savepath):
    data = pd.Series()

    for carrier in ["NH3", "steel", "oil"]:
        stat = n.statistics.withdrawal(bus_carrier=carrier, groupby=["bus", "carrier"]).filter(like="DE")
        stat.drop(index="Store", level=0, inplace=True)
        data.loc[carrier] = stat.sum()
    
    buses = n.buses.index[(n.buses.index.str[:2] == "DE")].drop("DE")
    nodal_balance = (
        n.statistics.energy_balance(
            aggregate_time=False,
            nice_names=False,
            groupby=pypsa.statistics.groupers["bus", "carrier", "bus_carrier"],
        )
        .loc[:, buses, :, :]
        .droplevel("bus").mul(n.snapshot_weightings.generators)) # multiply with snapshot weightings
    # methanol
    mask = nodal_balance.index.get_level_values("bus_carrier").isin(["methanol", "shipping methanol", "industry methanol"])
    nb = nodal_balance[mask].groupby("carrier").sum().T
    summe = nb.round(decimals=1).sum()
    data.loc["methanol"] = -summe.clip(upper=0).sum()
    # gas
    mask = nodal_balance.index.get_level_values("bus_carrier").isin(["gas", "renewable gas"])
    nb = nodal_balance[mask].groupby("carrier").sum().T
    summe = nb.round(decimals=1).sum()
    data.loc["gas"] = -summe.clip(upper=0).sum()
    # hydrogen
    mask = nodal_balance.index.get_level_values("bus_carrier").isin(["H2"])
    nb = nodal_balance[mask].groupby("carrier").sum().T
    summe = nb.round(decimals=1).sum()
    summe.drop(index=summe.index[summe.index.str.contains("pipeline")], inplace=True)
    data.loc["H2"] = -summe.clip(upper=0).sum()

    # electricity
    stat = n.statistics.withdrawal(bus_carrier=["AC", "low voltage"], groupby=["bus", "carrier"]).filter(like="DE")
    stat.drop(index="StorageUnit", level=0, inplace=True)
    stat.drop(index="Line", level=0, inplace=True)
    stat.drop(index="DC", level=2, inplace=True)
    stat.drop(index="electricity distribution grid", level=2, inplace=True)
    data.loc["Electricity"] = stat.sum()

    data.to_csv(savepath + "/demand.csv")



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_report",
            simpl="",
            clusters=68,
            planning_horizons=2050,
            opts="",
            ll="vopt",
            sector_opts="None",
            run="WHP",
        )

    os.makedirs(snakemake.output.report, exist_ok=True)
    configure_logging(snakemake)
    # collect general info
    scenario = scenario_dict[snakemake.wildcards.run]
    modelyears = [fn[-7:-3] for fn in snakemake.input.networks]
    time_resolution = get_time_resolution(snakemake.params.hours)

    # read in networks
    networks = [pypsa.Network(fn) for fn in snakemake.input.networks]

    # collect necessary plotting info
    colors = prepare_colors()

    # plot for every planning_horizon:
    for i,n in enumerate(networks):
        plot_hydrogen_balance(
            n,
            colors,
            year=modelyears[i],
            scenario=scenario,
            savepath=snakemake.output.report + f"/hydrogen_balance_{modelyears[i]}.png",
        )

        plot_carbon_balance(
            n,
            colors,
            year=modelyears[i],
            scenario=scenario,
            savepath=snakemake.output.report + f"/carbon_balance_{modelyears[i]}.png",
        )
        # print out some general information:
        # co2 price
        logger.info(f"CO2 emission shadow price {n.global_constraints.loc["CO2Limit", "mu"]} €/t)")
        # co2 price germany
        logger.info(f"CO2 emission Germany shadow price {n.global_constraints.loc["co2_limit-DE", "mu"] / 1e6} €/t)")
        # FT price EU
        logger.info(f"Renewable oil EU price {n.buses_t.marginal_price["EU renewable oil"].mean()} €/MWh)")
        # FT price Germany
        logger.info(f"Renewable oil Germany price {n.buses_t.marginal_price["DE renewable oil"].mean()} €/MWh)")

        weights = n.snapshot_weightings.generators
        logger.info("CH4 Consumption Germany")
        for carrier in n.links[n.links.bus0=="DE gas"].carrier.unique():
            ind = n.links[(n.links.carrier==carrier) & (n.links.index.str.startswith("DE"))].index
            cons = n.links_t.p0[ind].mul(weights, axis=0).sum().sum()/1e6
            logger.info(f"{carrier}: {cons} TWh")

    print_h2_FT_info(networks[-1])
    print_RE_info(networks[-1])
    dac = n.statistics.energy_balance(bus_carrier="co2").xs("DAC", level="carrier").values[0]/1e6
    logger.info(f"Direct air capture: {dac} Mt")
    oil_demand = -n.statistics.energy_balance(bus_carrier="oil").clip(upper=0).sum()
    fossil = n.statistics.energy_balance(bus_carrier="oil").xs("oil refining", level="carrier").values[0]
    logger.info(f"European fossil oil share {(fossil/oil_demand).round(decimals=4)*100} %")

    plot_import_volumes(
        networks,
        colors,
        years=modelyears,
        scenario=scenario,
        savepath=snakemake.output.import_vol,
    )

    get_demand(
        networks[-1],
        savepath=snakemake.output.report,
    )

    consumer_costs(
        networks,
        colors,
        years=modelyears,
        scenario=scenario,
        savepath=snakemake.output.cons_cost,
        )

    regions = gpd.read_file("/home/toni-seibold/dev/pypsa-de-import/resources/regions_onshore_base_s_68.geojson").set_index("name")
    regions["country"] = regions.index.str[:2]
    colormaps(
        networks,
        regions,
        years=modelyears,
        scenario=scenario,
        savepath=snakemake.output.report,
    )
    # price maps
    plot_prices(
        networks,
        regions,
        years=modelyears,
        scenario=scenario,
        relocation=snakemake.params.relocation,
        savepath=snakemake.output.report,
    )

    # non european import
    if snakemake.params.non_eu_import["enable"]:
        plot_non_eu_import(
            n=networks[-1],
            year=modelyears[-1],
            scenario=scenario,
            savepath=snakemake.output.report + "/non_eu_import_temporal.png"
        )

        plot_non_eu_import_balance(
            networks=networks,
            years=modelyears,
            scenario=scenario,
            savepath=snakemake.output.report
        )

        non_eu_import = networks[-1].statistics.withdrawal(bus_carrier="export", groupby=["bus"]).droplevel("component")
        non_eu_import.index = non_eu_import.index.str.replace(" export", "", regex=False)
        
        non_eu_import = non_eu_import.div(1e6).sort_values()
        logger.info("Countries with the highest exports to Europe in 2050 [TWh]:")
        logger.info(non_eu_import[-5:])

    plot_h2_map(
            networks[-1],
            regions,
            savepath=snakemake.output.report + "/h2_transmission.svg",
        )

    # plot_h2_pipeline_loading(
    #         networks[-1],
    #         tech_colors=colors,
    #         scenario=scenario,
    #         savepath=snakemake.output.report + "/h2_transmission_loading.svg")

    # # TODO: add sequestration
    # plot_hydrogen_carbon_map(
    #     n=networks[-1],
    #     regions=regions,
    #     year=modelyears[-1],
    #     scenario=scenario,
    #     savepath=snakemake.output.map,
    # )
    # # TODO: stranded assets map