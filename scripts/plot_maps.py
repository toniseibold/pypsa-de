import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import pypsa
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scripts.plot_power_network import load_projection
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import yaml
from matplotlib.patches import Patch

from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)

scenario_dict = {
    "no_co2_network": "No CO\u2082 network",
    "frozen_H2_28": "H\u2082 + CO\u2082 (PCI 2028) frozen",
    "endo_H2": "H\u2082 + CO\u2082 endogenous",
    "pcipmi_H2_+": "H\u2082 + CO\u2082 (PCI 2028)",
    "onshore_sequestration": "Onshore Seq.",
    "no_early_retirement": "No early BF-BOF retirement",
    "seq_50": "Low Seq Potential",
    "northern_lights": "No North Sea Seq",
    "onshore_sequestration_endo": "Onshore Seq endogenous",
    "no_north_sea_endo": "No North Sea endogenous",
}

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
carriers = ["H2 Electrolysis", "SMR", "SMR CC", "ammonia cracker"]

def group_pipes_co2(df, drop_direction=False):
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
        lambda x: f"CO2 pipeline {x.bus0.replace(' CO2', '')} -> {x.bus1.replace(' CO2', '')}",
        axis=1,
    )
    return df.groupby(level=0).agg(
        {"p_nom_opt": "sum", "bus0": "first", "bus1": "first", "index_orig": "first"}
    )


def group_pipes_h2(df, drop_direction=False):
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


def plot_co2_map(
        n: pypsa.Network,
        regions: gpd.GeoDataFrame,
        ax: plt.axes,
        title:str,
):
    # Take care of canvas with prices
    regions = regions.to_crs(proj.proj4_init)

    prices_t = n.buses_t.marginal_price[n.buses[n.buses.carrier=="co2 stored"].index].mean()
    prices = prices_t[~prices_t.index.str.contains("offshore")] - n.global_constraints.loc["CO2Limit", "mu"]
    prices.index = prices.index.str.replace(" co2 stored", "")
    regions["CO2"] = prices # EUR/t

    bus_size_factor = 5e7
    linewidth_factor = 1e3
    line_lower_threshold = 0

    co2_balance = n.statistics.energy_balance(bus_carrier="co2 stored", groupby=["bus", "carrier"]).droplevel("component")
    co2_balance = co2_balance[~co2_balance.index.get_level_values(1).str.contains("pipeline")]

    sequestration = co2_balance[(co2_balance.index.get_level_values(0).str.contains("offshore")) | (co2_balance.index.get_level_values(0).str.contains("PCI"))]
    co2_supply = co2_balance[~(co2_balance.index.get_level_values(0).str.contains("offshore")) | ~(co2_balance.index.get_level_values(0).str.contains("PCI"))]
    co2_supply = co2_supply.rename(lambda x: x.replace(" offshore 0 co2 stored", ""), level=0)
    co2_supply = co2_supply.drop("co2 sequestered", level=1)

    bus_colors = {carrier: tech_colors[carrier] for carrier in co2_supply.index.get_level_values(1).unique()}
    bus_colors["co2 sequestered"] = tech_colors["co2 sequestered"]

    n.links.drop(
        n.links.index[~(n.links.carrier.str.contains("CO2 pipeline")) | ~(n.links.active)], inplace=True
    )
    n.links.drop(
        n.links[n.links.p_nom_opt <= 0].index, inplace=True
    )
    # pipelines
    co2_pipes = n.links[(n.links.carrier == "CO2 pipeline")].p_nom_opt
    co2_pipes_pcipmi = n.links[(n.links.carrier == "CO2 pipeline pcipmi")].p_nom_opt

    link_widths_endo = co2_pipes / linewidth_factor
    link_widths_pcipmi = co2_pipes_pcipmi / linewidth_factor

    carriers_pipe = ["CO2 pipeline", "CO2 pipeline pcipmi"]

    n.links.bus0 = n.links.bus0.str.replace(" co2 stored", "")
    n.links.bus1 = n.links.bus1.str.replace(" co2 stored", "")

    # co2_supply[co2_supply.index.get_level_values(0).str.startswith("DE")]
    current = co2_supply[co2_supply > 1e3].index.get_level_values(1).unique()

    n.plot.map(
        geomap=True,
        bus_sizes=co2_supply/bus_size_factor,
        bus_split_circles=True,
        bus_colors=bus_colors,
        link_colors=tech_colors["CO2 pipeline"],
        link_widths=link_widths_endo,
        branch_components=["Link"],
        line_widths=0,
        ax=ax,
        **map_opts,
    )
    n.plot.map(
        geomap=True,
        bus_sizes=sequestration/bus_size_factor,
        bus_split_circles=True,
        bus_colors={"co2 sequestered": "#f2682f"},
        link_widths=link_widths_pcipmi,
        branch_components=["Link"],
        link_colors="#57232d",
        line_widths=0,
        ax=ax,
        **map_opts,
    )

    regions.plot(
        ax=ax,
        facecolor="white",
        edgecolor="grey",
        linewidth=0.1,
        legend=False,
    )

    ax.set_title(title)

    return current


def plot_h2_map(
        n: pypsa.Network,
        regions: gpd.GeoDataFrame,
        ax: plt.axis,
        title: str,
):
    # Take care of canvas with prices
    regions = regions.to_crs(proj.proj4_init)

    prices_t = n.buses_t.marginal_price[n.buses[n.buses.carrier=="H2"].index]
    demand = n.statistics.withdrawal(bus_carrier="H2", aggregate_time=False, groupby="bus").groupby("bus").sum().mul(n.snapshot_weightings.generators, axis=1)
    prices = prices_t.mul(demand.T).sum().div(demand.T.sum())
    prices.index = prices.index.str.replace(" H2", "")
    regions["H2"] = prices # EUR/MWh

    bus_size_factor = 1e5
    linewidth_factor = 4e3
    line_lower_threshold = 0

    # Drop non-electric buses so they don't clutter the plot
    n.buses.drop(n.buses.index[n.buses.carrier != "AC"], inplace=True)

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
    h2_new = group_pipes_h2(h2_new)

    if not h2_retro.empty:
        h2_retro = (
            group_pipes_h2(h2_retro, drop_direction=True).reindex(h2_new.index).fillna(0)
        )

    if not h2_kern.empty:
        h2_kern = (
            group_pipes_h2(h2_kern, drop_direction=True).reindex(h2_new.index).fillna(0)
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

    n.plot.map(
        geomap=True,
        bus_sizes=production,
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
        legend=False,
        vmax=120,
        vmin=50,
    )


    # axes[i].set_extent([5.9, 15.0, 47.3, 55.1], crs=ccrs.PlateCarree())
    ax.set_title(title)



def plot_maps(
        paths: list,
        regions=gpd.GeoDataFrame,
        year=int,
):
    """"
    Plot the carbon network for Europe and Germany.
    """
    # TODO: fig subplots
    fig1, ax1 = plt.subplots(2, 4, figsize=(len(paths)*2, 8), subplot_kw={"projection": proj})
    fig2, ax2 = plt.subplots(2, 4, figsize=(len(paths)*2, 8), subplot_kw={"projection": proj})
    acc_carrier = pd.Index([])
    axes1 = ax1.flatten()
    axes2 = ax2.flatten()
    for i, path in enumerate(paths):
        network = pypsa.Network(path)
        n = network.copy()
        current = plot_co2_map(n, regions, axes1[i], scenario_dict[scenarios[i]])
        acc_carrier = acc_carrier.union(current)
        del n
        n = network.copy()
        plot_h2_map(n, regions, axes2[i], scenario_dict[scenarios[i]])
        del network, n
    
    # ### co2 map
    # fig1.suptitle("2045", fontsize=16)
    # # Colorbar axis: [left, bottom, width, height]
    # cbar_ax = fig1.add_axes([0.3, 0.02, 0.4, 0.04])  # center it
    # # Set up colorbar
    # sm = cm.ScalarMappable(
    #     cmap="Purples",
    #     norm=mcolors.Normalize(vmin=30, vmax=100)
    # )
    # sm.set_array([])

    # # Draw horizontal colorbar
    # cbar = fig1.colorbar(
    #     sm,
    #     cax=cbar_ax,
    #     orientation="horizontal",
    #     extend="max",
    # )
    # cbar.set_label("€/t")

    sizes = [10, 5]
    labels = [f"{s} Mt" for s in sizes]
    sizes = [s / 5e7 * 1e6 for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0, -0.5),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_circles(
        ax1[1,0],
        sizes,
        labels,
        # srid=n.srid,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )
    sizes = [10, 5]
    labels = [f"{s} kt" for s in sizes]
    scale = 1e3 / 1e3
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0.5, -0.5),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    pypsa.plot.maps.static.add_legend_lines(
        ax1[1,0],
        sizes,
        labels,
        patch_kw=dict(color="lightgrey"),
        legend_kw=legend_kw,
    )
    plot_colors = {carrier: tech_colors[carrier] for carrier in acc_carrier}
    colors = list(plot_colors.values())
    labels = list(plot_colors.keys())

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(-0, -0.3),
        ncol=4,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_patches(ax1[1,0], colors, labels, legend_kw=legend_kw)

    plot_colors = {"CO2 Pipeline": tech_colors["CO2 pipeline"], "PCIPMI CO2 Pipeline": "#57232d"}
    colors = list(plot_colors.values())
    labels = list(plot_colors.keys())
    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0, -0.5),
        frameon=False,
    )
    pypsa.plot.maps.static.add_legend_patches(axes1[6], colors, labels, legend_kw=legend_kw)

    ### h2 map
    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    sizes = [s / 1e5 * 1e3 for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0, -0.5),
        labelspacing=0.8,
        handletextpad=0,
        frameon=False,
    )

    pypsa.plot.maps.static.add_legend_circles(
        ax2[1,0],
        sizes,
        labels,
        patch_kw=dict(facecolor="lightgrey"),
        legend_kw=legend_kw,
    )

    sizes = [30, 10]
    labels = [f"{s} GW" for s in sizes]
    scale = 1e3 / 4e3
    sizes = [s * scale for s in sizes]

    legend_kw = dict(
        loc="lower left",
        bbox_to_anchor=(0.4, -0.5),
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    pypsa.plot.maps.static.add_legend_lines(
        ax2[1,0],
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
        bbox_to_anchor=(0, -0.22),
        ncol=7,
        frameon=False,
        labelspacing=0.8,
        handletextpad=1,
    )

    pypsa.plot.maps.static.add_legend_patches(
        ax2[1,0],
        colors,
        labels,
        legend_kw=legend_kw)

    cbar_ax = fig2.add_axes([0.3, 0.02, 0.4, 0.04])  # center it

    sm = cm.ScalarMappable(
        cmap="Blues",
        norm=mcolors.Normalize(vmin=50, vmax=120)
    )
    sm.set_array([])

    cbar = fig2.colorbar(
        sm,
        cax=cbar_ax,
        orientation="horizontal",
        extend="max",
    )
    cbar.set_label("€/MWh")

    fig2.suptitle(year, fontsize=16)

    fig1.savefig(save_dir + f"/EU_co2_stored_map_{year}.pdf", bbox_inches="tight")
    fig2.savefig(save_dir + f"/EU_h2_map_{year}.pdf", bbox_inches="tight")

    for axis in axes1:
        axis.set_extent([5.9, 15.0, 47.3, 55.1], crs=ccrs.PlateCarree())
    for axis in axes2:
        axis.set_extent([5.9, 15.0, 47.3, 55.1], crs=ccrs.PlateCarree())

    fig1.savefig(save_dir + f"/DE_co2_stored_map_{year}.pdf", bbox_inches="tight")
    fig2.savefig(save_dir + f"/DE_h2_map_{year}.pdf", bbox_inches="tight")



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_maps",
            simpl="",
            clusters=89,
            planning_horizons=2035,
            opts="",
            ll="vopt",
            sector_opts="none",
            run="endo_H2",
        )

    configure_logging(snakemake)

    scenarios = snakemake.params.scenarios

    n_paths = snakemake.input.networks

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    regions["country"] = regions.index.str[:2]
    year = 2035
    path = snakemake.input.networks[0]
    save_dir = path.split("results/")[0] + "results/" + path.split("results/")[1].split("/")[0]

    tech_colors = snakemake.params.plotting["tech_colors"]

    map_opts = snakemake.params.plotting["map"]
    proj = load_projection(map_opts)

    plot_maps(
        paths=n_paths,
        regions=regions,
        year=year,
    )
