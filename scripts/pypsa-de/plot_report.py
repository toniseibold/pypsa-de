# -*- coding: utf-8 -*-
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import cartopy.crs as ccrs
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pypsa
from matplotlib.patches import Patch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pypsa.statistics import get_transmission_carriers
from scripts._helpers import configure_logging, mock_snakemake
from scripts.make_summary import assign_locations
from scripts.plot_power_network import load_projection
import textwrap
import yaml

logger = logging.getLogger(__name__)

scenario_dict = {
  "h2_network_27": "h2_network_27",
  "h2_network": "h2_network",
  "meoh_import": "meoh_import",
  "onshore_sequestration": "onshore_sequestration",
  "no_co2_network": "no_co2_network",
}

def plot_hydrogen_balance(networks, modelyears, scenario, savepath):
    H2_DE = pd.DataFrame()
    H2_EU = pd.DataFrame()

    for i, n in enumerate(networks):
        DE = (
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
        DE["import"] = n.links_t.p0[incoming].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6
        DE["export"] = -n.links_t.p0[outgoing].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6

        EU = (
            n.statistics.energy_balance(bus_carrier="H2")
            .groupby("carrier")
            .sum()
            .div(1e6)
        )

        all_idx = H2_DE.index.union(DE.index)
        H2_DE = H2_DE.reindex(all_idx)
        H2_DE.loc[DE.index, modelyears[i]] = DE
        all_idx = H2_EU.index.union(EU.index)
        H2_EU = H2_EU.reindex(all_idx)
        H2_EU.loc[EU.index, modelyears[i]] = EU

    # drop store & pipelines
    H2_DE = H2_DE[~(H2_DE.index.str.contains("pipeline")) & ~(H2_DE.index.str.contains("Store"))]
    H2_EU = H2_EU[~(H2_EU.index.str.contains("pipeline")) & ~(H2_EU.index.str.contains("Store"))]

    # reorder
    order = list(n.statistics.energy_balance(bus_carrier="H2").sort_values(ascending=False).index.get_level_values(1))
    order.insert(5, "import")
    order.insert(-1, "export")
    H2_DE = H2_DE.reindex(order)
    H2_EU = H2_EU.reindex(order)
    
    tech_colors["H2 OCGT"] = tech_colors["OCGT"]
    tech_colors["export"] = tech_colors["import H2"]
    tech_colors["import"] = tech_colors["import H2"]
    tech_colors["urban central H2 CHP"] = tech_colors["urban central gas CHP"]

    fig, ax = plt.subplots(2, 1, figsize=(14, 12))
    
    H2_DE.T.plot.barh(ax=ax[0], color=[tech_colors[col] for col in H2_DE.index], stacked=True, width=0.8, legend=False)
    ax[0].set_xlabel("TWh")
    ax[0].set_title("Germany")

    H2_EU.T.plot.barh(ax=ax[1], color=[tech_colors[col] for col in H2_DE.index], stacked=True, width=0.8, legend=False)
    ax[1].set_xlabel("TWh")
    ax[1].set_title("Europe")

    handles, labels = ax[0].get_legend_handles_labels()

    supply_handles = handles[:6]
    supply_labels = labels[:6]
    demand_handles = handles[6:]
    demand_labels = labels[6:]
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

    ax[0].legend(
        combined_handles,
        combined_labels,
        loc="upper center",
        bbox_to_anchor=(1.1, 0.3),
    )
    fig.suptitle(scenario)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_carbon_balance(networks, modelyears, scenario, savepath):
    CO2_DE = pd.DataFrame()
    CO2_EU = pd.DataFrame()
    for i, n in enumerate(networks):
        DE = (
            n.statistics.energy_balance(bus_carrier="co2 stored", groupby=["bus", "carrier"])
            .filter(like="DE")
            .groupby("carrier")
            .sum()
            .div(1e6)
        )
        # add import/export to Germany
        incoming = n.links[
            (n.links.bus0.str[:2] != "DE") & 
            (n.links.bus1.str[:2] == "DE") &
            (n.links.carrier.str.contains("CO2 pipeline"))
        ].index
        outgoing = n.links[
            (n.links.bus0.str[:2] == "DE") & 
            (n.links.bus1.str[:2] != "DE") &
            (n.links.carrier.str.contains("CO2 pipeline"))
        ].index
        DE["trade"] = n.links_t.p0[incoming].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6 - n.links_t.p0[outgoing].mul(n.snapshot_weightings.generators, axis=0).sum().sum() / 1e6

        EU = (
            n.statistics.energy_balance(bus_carrier="co2 stored")
            .groupby("carrier")
            .sum()
            .div(1e6)
        )

        all_idx = CO2_DE.index.union(DE.index)
        CO2_DE = CO2_DE.reindex(all_idx)
        CO2_DE.loc[DE.index, modelyears[i]] = DE
        all_idx = CO2_EU.index.union(EU.index)
        CO2_EU = CO2_EU.reindex(all_idx)
        CO2_EU.loc[EU.index, modelyears[i]] = EU

    # drop store & pipelines
    CO2_DE = CO2_DE[~(CO2_DE.index.str.contains("pipeline"))]
    CO2_EU = CO2_EU[~(CO2_EU.index.str.contains("pipeline"))]
    # manipulate CC in 2025
    CO2_DE.loc["steel emission CC", "2025"] += CO2_DE.loc["co2 vent", "2025"]
    CO2_DE.loc["co2 vent", "2025"] = 0
    CO2_EU.loc["steel emission CC", "2025"] += CO2_EU.loc["co2 vent", "2025"]
    CO2_EU.loc["co2 vent", "2025"] = 0

    # reorder
    order = list(n.statistics.energy_balance(bus_carrier="co2 stored").sort_values(ascending=False).index.get_level_values(1))
    order.insert(-1, "trade")
    CO2_DE = CO2_DE.reindex(order)
    CO2_EU = CO2_EU.reindex(order)
    tech_colors["trade"] = '#f0d071'
    fig, ax = plt.subplots(2, 1, figsize=(14, 12))
    
    CO2_DE.T.plot.barh(ax=ax[0], color=[tech_colors[col] for col in CO2_DE.index], stacked=True, width=0.8, legend=False)
    ax[0].set_xlabel("Mt")
    ax[0].set_title("Germany")

    CO2_EU.T.plot.barh(ax=ax[1], color=[tech_colors[col] for col in CO2_EU.index], stacked=True, width=0.8, legend=False)
    ax[1].set_xlabel("Mt")
    ax[1].set_title("Europe")
    # adjust plot
    ax[0].legend(
            loc="upper center",
            bbox_to_anchor=(1.1, 1.0),
        )
    fig.suptitle(scenario)

    fig.savefig(savepath, bbox_inches="tight")
    CO2_DE.to_csv(snakemake.output.report + f"/co2_balance_DE_{modelyears[i]}.csv")
    CO2_EU.to_csv(snakemake.output.report + f"/co2_balance_Europe_{modelyears[i]}.csv")
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


def plot_hydrogen_capacities(networks, modelyears, scenario, regions, savepath):
    logger.info("Plotting hydrogen map")
    map_opts = snakemake.params.plotting["map"]
    snakemake.params.plotting["projection"] = {"name": "EqualEarth"}
    proj = load_projection(map_opts)
    fig, ax = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={"projection": proj})
    regions = regions.to_crs(proj.proj4_init)
    for i, n in enumerate(networks):
        n = n.copy()
        
        assign_locations(n)
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

        carriers = ["H2 Electrolysis", "SMR", "SMR CC", "ammonia cracker"]
        production = n.links[n.links.carrier.isin(carriers)].groupby(["bus1", "carrier"]).p_nom_opt.sum()

        # make a fake MultiIndex so that area is correct for legend
        production.rename(index=lambda x: x.replace(" H2", ""), level=0, inplace=True)
        production /= bus_size_factor
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

        color_h2_pipe = "#b3f3f4"
        color_retrofit = "#499a9c"
        color_kern = "#6b3161"

        bus_colors = {"H2 Electrolysis": "#ff29d9",
                    "H2 Fuel Cell": "#805394",
                    "SMR": '#870c71',
                    "SMR CC": '#4f1745',
                    "Sabatier": '#9850ad',
                    "ammonia cracker": '#87d0e6',
                    }
        n.plot.map(
            geomap=True,
            bus_sizes=production,
            bus_colors=bus_colors,
            link_colors=color_h2_pipe,
            link_widths=link_widths_total,
            branch_components=["Link"],
            ax=ax[i],
            **map_opts,
        )

        n.plot.map(
            geomap=True,
            bus_sizes=0,
            link_colors=color_retrofit,
            link_widths=link_widths_retro,
            branch_components=["Link"],
            ax=ax[i],
            **map_opts,
        )

        n.plot.map(
            geomap=True,
            bus_sizes=0,
            link_colors=color_kern,
            link_widths=link_widths_kern,
            branch_components=["Link"],
            ax=ax[i],
            **map_opts,
        )

        regions.plot(
            ax=ax[i],
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
        ax[i].set_facecolor("white")
        ax[i].set_title(modelyears[i])

    sizes = [50, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / bus_size_factor * 1e3 for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0, -0.5),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_circles(
        ax[0],
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
        loc="lower left",
        bbox_to_anchor=(0.5, -0.5),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    pypsa.plot.maps.static.add_legend_lines(
        ax[0],
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

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0, -0.5),
        ncol=3,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_patches(ax[1], colors, labels, legend_kw=legend_kw)
    fig.suptitle(scenario)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_hydrogen_volumes(networks, modelyears, scenario, regions, savepath):

    map_opts = snakemake.params.plotting["map"]
    snakemake.params.plotting["projection"] = {"name": "EqualEarth"}
    proj = load_projection(map_opts)
    fig, ax = plt.subplots(1, 3, figsize=(12, 5), subplot_kw={"projection": proj})
    regions = regions.to_crs(proj.proj4_init)
    for i, n in enumerate(networks):
        n = n.copy()
        assign_locations(n)

        prices_t = n.buses_t.marginal_price[n.buses[n.buses.carrier=="H2"].index]
        demand = n.statistics.withdrawal(bus_carrier="H2", aggregate_time=False, groupby="bus").groupby("bus").sum().mul(n.snapshot_weightings.generators, axis=1)
        prices = prices_t.mul(demand.T).sum().div(demand.T.sum())
        prices.index = prices.index.str.replace(" H2", "")
        regions["H2"] = prices # EUR/MWh

        bus_size_factor = 5e7

        energy_balance = n.statistics.energy_balance(bus_carrier="H2", groupby=["bus", "carrier"]).droplevel("component")
        energy_balance = energy_balance[~energy_balance.index.get_level_values(1).str.contains("pipeline")]

        n.links.drop(
            n.links.index[~n.links.carrier.str.contains("H2 pipeline")], inplace=True
        )

        bus_colors = {carrier: tech_colors[carrier] for carrier in energy_balance.index.get_level_values(1).unique()}

        n.plot.map(
            geomap=True,
            bus_sizes=energy_balance/bus_size_factor,
            bus_split_circles=True,
            bus_colors=bus_colors,
            link_widths=0,
            line_widths=0,
            ax=ax[i],
            **map_opts,
        )

        regions.plot(
            ax=ax[i],
            column="H2",
            cmap="Blues",
            linewidths=0,
            legend=False,
            vmax=130,
            vmin=60,
            legend_kwds={
                "label": "H2 Price [EUR/MWh]",
                "shrink": 0.7,
                "extend": "max",
            },
        )
        ax[i].set_facecolor("white")
        ax[i].set_title(modelyears[i])
    # Colorbar axis: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.04])  # center it
    # Set up colorbar
    sm = cm.ScalarMappable(
        cmap="Blues",
        norm=mcolors.Normalize(vmin=60, vmax=130)
    )
    sm.set_array([])

    # Draw horizontal colorbar
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal",
        extend="max",
    )
    cbar.set_label("€/MWh")

    sizes = [50, 10]
    labels = [f"{s} TWh" for s in sizes]
    sizes = [s / bus_size_factor * 1e6 for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0, -0.2),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_circles(
        ax[0],
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = list(bus_colors.values())
    labels = list(bus_colors.keys())

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(-0.5, -0.3),
        ncol=4,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_patches(ax[1], colors, labels, legend_kw=legend_kw)
    fig.suptitle(f"{scenario} H2 Price and Volumes", fontsize=16)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_carbon_network(networks, modelyears, scenario, regions, savepath):
    # TONI TODO: pipeline is still a bit shaky
    map_opts = snakemake.params.plotting["map"]
    snakemake.params.plotting["projection"] = {"name": "EqualEarth"}
    proj = load_projection(map_opts)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={"projection": proj})
    regions = regions.to_crs(proj.proj4_init)
    for i, n in enumerate(networks):
        n = n.copy()
        assign_locations(n)

        prices_t = n.buses_t.marginal_price[n.buses[n.buses.carrier=="co2 stored"].index].mean()
        prices = prices_t[~prices_t.index.str.contains("offshore")] - n.global_constraints.loc["CO2Limit", "mu"]
        prices.index = prices.index.str.replace(" co2 stored", "")
        regions["CO2"] = prices # EUR/t
        regions.to_csv(snakemake.output.report + f"/co2_stored_prices_{modelyears[i]}.csv")

        bus_size_factor = 5e7
        linewidth_factor = 1e3
        line_lower_threshold = 0

        co2_balance = n.statistics.energy_balance(bus_carrier="co2 stored", groupby=["bus", "carrier"]).droplevel("component")
        co2_balance = co2_balance[~co2_balance.index.get_level_values(1).str.contains("pipeline")]

        sequestration = co2_balance[co2_balance.index.get_level_values(0).str.contains("offshore")]
        co2_supply = co2_balance[~co2_balance.index.get_level_values(0).str.contains("offshore")]
        co2_supply = co2_supply.rename(lambda x: x.replace(" offshore 0 co2 stored", ""), level=0)
        
        bus_colors = {carrier: tech_colors[carrier] for carrier in co2_supply.index.get_level_values(1).unique()}
        bus_colors["co2 sequestered"] = tech_colors["co2 sequestered"]

        n.links.drop(
            n.links.index[~n.links.carrier.str.contains("CO2 pipeline")], inplace=True
        )
        # pipelines
        co2_pipes = n.links[n.links.carrier == "CO2 pipeline"]
        co2_pipes = group_pipes(co2_pipes)

        co2_pipes = co2_pipes.p_nom_opt.groupby(level=0).sum()
        link_widths_total = co2_pipes / linewidth_factor

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

        carriers_pipe = ["CO2 pipeline"]
        total = n.links.p_nom_opt.where(n.links.carrier.isin(carriers_pipe), other=0.0)

        link_widths_total = total / linewidth_factor
        link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

        n.links.bus0 = n.links.bus0.str.replace(" co2 stored", "")
        n.links.bus1 = n.links.bus1.str.replace(" co2 stored", "")
        # save data
        co2_supply.to_csv(snakemake.output.report + f"/co2_supply_{modelyears[i]}.csv")
        total.to_csv(snakemake.output.report + f"/co2_pipelines_{modelyears[i]}.csv")
        sequestration.to_csv(snakemake.output.report + f"/co2_sequestration_{modelyears[i]}.csv")

        n.plot.map(
            geomap=True,
            bus_sizes=co2_supply/bus_size_factor,
            bus_split_circles=True,
            bus_colors=bus_colors,
            link_colors=tech_colors["CO2 pipeline"],
            link_widths=link_widths_total,
            branch_components=["Link"],
            line_widths=0,
            ax=ax[i],
            **map_opts,
        )
        n.plot.map(
            geomap=True,
            bus_sizes=sequestration/bus_size_factor,
            bus_split_circles=True,
            bus_colors={"co2 sequestered": "#f2682f"},
            link_widths=0,
            line_widths=0,
            ax=ax[i],
            **map_opts,
        )

        regions.plot(
            ax=ax[i],
            column="CO2",
            cmap="Purples",
            linewidths=0,
            legend=False,
            vmax=70,
            vmin=0,
        )
        ax[i].set_facecolor("white")
        ax[i].set_title(modelyears[i])

    # Colorbar axis: [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.04])  # center it
    # Set up colorbar
    sm = cm.ScalarMappable(
        cmap="Purples",
        norm=mcolors.Normalize(vmin=0, vmax=70)
    )
    sm.set_array([])

    # Draw horizontal colorbar
    cbar = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal",
        extend="max",
    )
    cbar.set_label("€/t")

    sizes = [50, 10]
    labels = [f"{s} Mt" for s in sizes]
    sizes = [s / bus_size_factor * 1e6 for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0, -0.2),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_circles(
        ax[0],
        sizes,
        labels,
        srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )
    sizes = [30, 10]
    labels = [f"{s} kt" for s in sizes]
    scale = 1e3 / linewidth_factor
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0.5, -0.5),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    pypsa.plot.maps.static.add_legend_lines(
        ax[0],
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )

    colors = list(bus_colors.values())
    labels = list(bus_colors.keys())

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(-0, -0.3),
        ncol=4,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_patches(ax[0], colors, labels, legend_kw=legend_kw)
    fig.suptitle(f"{scenario} CO2 Emission and Sequestration", fontsize=16)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def post_combustion_cc(n, suffix, sw):
    cc_ind = n.links[(n.links.index.str.contains(f"{suffix} emission CC")) & (n.links.index.str[:2] == "DE")].index
    cc = n.links_t.p0[cc_ind].mul(sw, axis=0).sum().sum()

    co2_ind = n.links[(n.links.index.str.contains(f"{suffix} emission")) & (n.links.index.str[:2] == "DE") & ~(n.links.index.str.contains("CC"))].index
    co2 = n.links_t.p0[co2_ind].mul(sw, axis=0).sum().sum()

    return cc/(cc+co2)


def plot_industry_capacities(networks, modelyears, scenario, savepath):
    index = pd.Index(modelyears)
    steel_vol = pd.DataFrame(index=index, columns=["BOF", "BOF CC", "gas EAF", "gas EAF CC", "H2 EAF"])
    cement_vol = pd.DataFrame(index=index, columns=["cement", "cement CC"])
    ethy_vol = pd.DataFrame(index=index, columns=["oil", "MeOH"])
    for i, n in enumerate(networks):
        ## steel
        share = post_combustion_cc(n, "BOF", sw)
        bof_links = n.links[(n.links.carrier=="BOF") & (n.links.index.str[:2] == "DE")].index
        steel_vol.loc[index[i], "BOF"] = (1-share) * abs(n.links_t.p1[bof_links].mul(sw, axis=0).sum().sum())
        steel_vol.loc[index[i], "BOF CC"] = share * abs(n.links_t.p1[bof_links].mul(sw, axis=0).sum().sum())

        share = post_combustion_cc(n, "gas DRI", sw)
        gas_eaf_links = n.links[(n.links.carrier=="gas DRI") & (n.links.index.str[:2] == "DE")].index
        steel_vol.loc[index[i], "gas EAF"] = (1-share) * abs(n.links_t.p1[gas_eaf_links].mul(sw, axis=0).sum().sum())
        steel_vol.loc[index[i], "gas EAF CC"] = share * abs(n.links_t.p1[gas_eaf_links].mul(sw, axis=0).sum().sum())

        h2_eaf_links = n.links[(n.links.carrier=="H2 DRI") & (n.links.index.str[:2] == "DE")].index
        steel_vol.loc[index[i], "H2 EAF"] = abs(n.links_t.p1[h2_eaf_links].mul(sw, axis=0).sum().sum())

        ## cement
        cement_links = n.links[(n.links.carrier.str.contains("cement")) & (n.links.index.str[:2] == "DE")].index
        cement_vol.loc[index[i], "cement"] = abs(n.links_t.p1[cement_links].mul(sw, axis=0).sum().sum())

        cement_cc_links = n.links[(n.links.carrier=="cement CC") & (n.links.index.str[:2] == "DE")].index
        cement_vol.loc[index[i], "cement CC"] = abs(n.links_t.p1[cement_cc_links].mul(sw, axis=0).sum().sum())

        ## ethylene
        meoh2oa_links = n.links[(n.links.carrier=="methanol-to-olefins/aromatics") & (n.links.index.str[:2] == "DE")].index
        ethy_vol.loc[index[i], "MeOH"] = abs(n.links_t.p1[meoh2oa_links].mul(sw, axis=0).sum().sum())

        naphtha_links = n.links[(n.links.carrier=="naphtha for industry") & (n.links.index.str[:2] == "DE")].index
        ethy_vol.loc[index[i], "oil"] = abs(n.links_t.p1[naphtha_links].mul(sw, axis=0).sum().sum())
    
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    plt.xticks(rotation=30, ha="right")

    steel_vol.div(1e6).plot.bar(ax=ax[0], stacked=True, width=0.8, legend=True)
    ax[0].set_title("Steel Production")
    ax[0].set_ylabel("Mt/a")
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3)

    cement_vol.div(1e6).plot.bar(ax=ax[1], stacked=True, width=0.8, legend=True)
    ax[1].set_title("Cement Production")
    ax[1].set_ylabel("Mt/a")
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2)

    ethy_vol.div(1e6).plot.bar(ax=ax[2], stacked=True, width=0.8, legend=True)
    ax[2].set_title("Ethylene Production")
    ax[2].set_ylabel("TWh/a")
    ax[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=2)

    for axis in ax:
        axis.set_xticklabels(labels=index, rotation=30, ha="right")
    fig.suptitle(scenario)

    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def plot_commodity_prices(networks, modelyears, scenario, savepath):
    prices = pd.DataFrame(index=modelyears, columns=["steel", "meoh", "cement"])
    for i, n in enumerate(networks):
        co2_price = (-n.global_constraints.loc["CO2Limit", "mu"]-n.global_constraints.loc["co2_limit-DE", "mu"]).round(decimals=2)
        steel_bus = n.buses[(n.buses.carrier=="steel") & (n.buses.index.str[:2] == "DE")].index
        prices.loc[modelyears[i], "steel"] = n.buses_t.marginal_price[steel_bus].mean().mean()
        meoh_bus = n.buses[(n.buses.carrier=="methanol") & (n.buses.index.str[:2] == "DE")].index
        prices.loc[modelyears[i], "meoh"] = n.buses_t.marginal_price[meoh_bus].mean().values[0] + co2_price*0.248
        cement_bus = n.buses[(n.buses.carrier=="cement") & (n.buses.index.str[:2] == "DE")].index
        prices.loc[modelyears[i], "cement"] = n.buses_t.marginal_price[cement_bus].mean().mean()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    plt.xticks(rotation=30, ha="right")

    ax[0].bar(prices.index, prices["steel"])
    ax[0].set_title("Steel Price")
    ax[0].set_ylabel("€/t")

    ax[1].bar(prices.index, prices["meoh"])
    ax[1].set_title("MeOH Price")
    ax[1].set_ylabel("€/MWh")

    cement_bar = ax[2].bar(prices.index, prices["cement"])
    ax[2].set_title("Cement Price")
    ax[2].set_ylabel("€/t")
    
    fig.suptitle(scenario)
    fig.savefig(savepath, bbox_inches="tight")
    plt.close()


def print_flh(n, sw):
    threshold = 1
    carriers=[
        "BOF",
        "gas DRI",
        "H2 DRI0",
        "cement kiln",
        "Haber-Bosch",
        "grey methanol",
        "blue methanol",
        "methanolisation",
    ]
    for carrier in carriers:
        index = n.links[(n.links.carrier==carrier) &
                        (n.links.index.str[:2]=="DE") &
                        (n.links.p_nom_opt>threshold)
                    ].index
        if index.empty:
            continue
        p_nom = n.links.loc[index, "p_nom_opt"]
        output = n.links_t.p0[index].mul(sw, axis=0).sum()
        cap_factor = output / (p_nom*8760)
        logger.info(f"Capacity Factor for {carrier}")
        logger.info(cap_factor)


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_report",
            simpl="",
            clusters=68,
            planning_horizons=2035,
            opts="",
            ll="vopt",
            sector_opts="None",
            run="h2_network_27",
        )

    os.makedirs(snakemake.output.report, exist_ok=True)

    configure_logging(snakemake)

    # collect general info
    scenario = scenario_dict[snakemake.wildcards.run]
    modelyears = [fn[-7:-3] for fn in snakemake.input.networks]

    # read in networks
    networks = [pypsa.Network(fn) for fn in snakemake.input.networks]

    regions = regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    regions["country"] = regions.index.str[:2]

    with open("/home/toni-seibold/dev/pypsa-de-co2/config/plotting.default.yaml") as stream:
        plotting = yaml.safe_load(stream)
    tech_colors = plotting["plotting"]["tech_colors"]
    sw = networks[0].snapshot_weightings.generators

    plot_hydrogen_capacities(
        networks,
        modelyears,
        scenario,
        regions,
        savepath=snakemake.output.report + f"/hydrogen_capacities.png",
        )

    plot_hydrogen_volumes(
        networks,
        modelyears,
        scenario,
        regions,
        savepath=snakemake.output.hydrogen_volumes,
        )

    plot_carbon_network(
        networks[1:],
        modelyears[1:],
        scenario,
        regions,
        savepath=snakemake.output.report + f"/carbon_network.png",
        )

    plot_hydrogen_balance(
        networks,
        modelyears,
        scenario,
        savepath=snakemake.output.report + f"/hydrogen_balance.png",
        )

    plot_carbon_balance(
        networks,
        modelyears,
        scenario,
        savepath=snakemake.output.report + f"/carbon_balance.png",
        )
    
    plot_industry_capacities(
        networks,
        modelyears,
        scenario,
        savepath=snakemake.output.report + f"/industry_production.png"
        )
    
    plot_commodity_prices(
        networks,
        modelyears,
        scenario,
        savepath=snakemake.output.report + f"/commodity_prices.png"
        )

    for i, n in enumerate(networks):
        h2_pipeline_carrier=["H2 pipeline", "H2 pipeline (Kernnetz)", "H2 pipeline retrofitted"]
        h2_pipes = n.links[
            (n.links.carrier.isin(h2_pipeline_carrier)) &
            ~(n.links.carrier.str.contains("reversed"))
            ].index
        TWkm = (n.links.loc[h2_pipes].length * n.links.loc[h2_pipes].p_nom_opt).sum() / 1e6
        logger.info(f"H2 Network {modelyears[i]}: {round(TWkm)} TWkm")
        # print Mtkm CO2 grid
        co2_pipes = n.links[
            (n.links.carrier == "CO2 pipeline") &
            ~(n.links.carrier.str.contains("reversed"))
            ].index
        Mtkm = (n.links.loc[co2_pipes].length * n.links.loc[co2_pipes].p_nom_opt).sum() / 1e6
        logger.info(f"CO2 Network {modelyears[i]}: {round(Mtkm)} Mtkm")

        print_flh(n, sw)