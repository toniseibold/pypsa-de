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


def plot_carbon_network_de(modelyears, scenarios, regions):
    map_opts = snakemake.params.plotting["map"]
    snakemake.params.plotting["projection"] = {"name": "EqualEarth"}
    proj = load_projection(map_opts)
    fig, ax = plt.subplots(len(modelyears), len(scenarios), figsize=(12, 5), subplot_kw={"projection": proj})
    regions = regions.to_crs(proj.proj4_init)
    
    bus_size_factor = 5e7
    linewidth_factor = 1e3
    line_lower_threshold = 0
    
    for i, m in enumerate(modelyears):
        for j, n in enumerate(scenarios):
            prices = pd.read_csv(snakemake.input.reports[j] + f"/co2_stored_prices_{modelyears[i]}.csv")
            regions["CO2"] = prices["CO2"]

            # n.plot.map(
            #     geomap=True,
            #     bus_sizes=co2_supply/bus_size_factor,
            #     bus_split_circles=True,
            #     bus_colors=bus_colors,
            #     link_colors=tech_colors["CO2 pipeline"],
            #     link_widths=link_widths_total,
            #     branch_components=["Link"],
            #     line_widths=0,
            #     ax=ax[i],
            #     **map_opts,
            # )
            # n.plot.map(
            #     geomap=True,
            #     bus_sizes=sequestration/bus_size_factor,
            #     bus_split_circles=True,
            #     bus_colors={"co2 sequestered": "#f2682f"},
            #     link_widths=0,
            #     line_widths=0,
            #     ax=ax[i],
            #     **map_opts,
            # )
            regions.plot(
                ax=ax[i,j],
                column="CO2",
                cmap="Purples",
                linewidths=0,
                legend=False,
                vmax=70,
                vmin=0,
            )
            

    #     co2_balance = n.statistics.energy_balance(bus_carrier="co2 stored", groupby=["bus", "carrier"]).droplevel("component")
    #     co2_balance = co2_balance[~co2_balance.index.get_level_values(1).str.contains("pipeline")]

    #     sequestration = co2_balance[co2_balance.index.get_level_values(0).str.contains("offshore")]
    #     co2_supply = co2_balance[~co2_balance.index.get_level_values(0).str.contains("offshore")]
    #     co2_supply = co2_supply.rename(lambda x: x.replace(" offshore 0 co2 stored", ""), level=0)
        
    #     bus_colors = {carrier: tech_colors[carrier] for carrier in co2_supply.index.get_level_values(1).unique()}
    #     bus_colors["co2 sequestered"] = tech_colors["co2 sequestered"]

    #     n.links.drop(
    #         n.links.index[~n.links.carrier.str.contains("CO2 pipeline")], inplace=True
    #     )
    #     # pipelines
    #     co2_pipes = n.links[n.links.carrier == "CO2 pipeline"]
    #     co2_pipes = group_pipes(co2_pipes)

    #     co2_pipes = co2_pipes.p_nom_opt.groupby(level=0).sum()
    #     link_widths_total = co2_pipes / linewidth_factor

    #     # drop all reversed pipe
    #     n.links.drop(n.links.index[n.links.index.str.contains("reversed")], inplace=True)
    #     n.links.rename(index=lambda x: x.split("-2")[0], inplace=True)
    #     n.links = n.links.groupby(level=0).agg(
    #         {
    #             **{
    #                 col: "first" for col in n.links.columns if col != "p_nom_opt"
    #             },  # Take first value for all columns except 'p_nom_opt'
    #             "p_nom_opt": "sum",  # Sum values for 'p_nom_opt'
    #         }
    #     )
    #     link_widths_total = link_widths_total.reindex(n.links.index).fillna(0.0)
    #     link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    #     carriers_pipe = ["CO2 pipeline"]
    #     total = n.links.p_nom_opt.where(n.links.carrier.isin(carriers_pipe), other=0.0)

    #     link_widths_total = total / linewidth_factor
    #     link_widths_total[n.links.p_nom_opt < line_lower_threshold] = 0.0

    #     n.links.bus0 = n.links.bus0.str.replace(" co2 stored", "")
    #     n.links.bus1 = n.links.bus1.str.replace(" co2 stored", "")

    #     n.plot.map(
    #         geomap=True,
    #         bus_sizes=co2_supply/bus_size_factor,
    #         bus_split_circles=True,
    #         bus_colors=bus_colors,
    #         link_colors=tech_colors["CO2 pipeline"],
    #         link_widths=link_widths_total,
    #         branch_components=["Link"],
    #         line_widths=0,
    #         ax=ax[i],
    #         **map_opts,
    #     )
    #     n.plot.map(
    #         geomap=True,
    #         bus_sizes=sequestration/bus_size_factor,
    #         bus_split_circles=True,
    #         bus_colors={"co2 sequestered": "#f2682f"},
    #         link_widths=0,
    #         line_widths=0,
    #         ax=ax[i],
    #         **map_opts,
    #     )

    #     regions.plot(
    #         ax=ax[i],
    #         column="CO2",
    #         cmap="Purples",
    #         linewidths=0,
    #         legend=False,
    #         vmax=70,
    #         vmin=0,
    #     )
    #     ax[i].set_facecolor("white")
    #     ax[i].set_title(modelyears[i])

    # # Colorbar axis: [left, bottom, width, height]
    # cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.04])  # center it
    # # Set up colorbar
    # sm = cm.ScalarMappable(
    #     cmap="Purples",
    #     norm=mcolors.Normalize(vmin=0, vmax=70)
    # )
    # sm.set_array([])

    # # Draw horizontal colorbar
    # cbar = fig.colorbar(
    #     sm,
    #     cax=cbar_ax,
    #     orientation="horizontal",
    #     extend="max",
    # )
    # cbar.set_label("€/t")

    # sizes = [50, 10]
    # labels = [f"{s} Mt" for s in sizes]
    # sizes = [s / bus_size_factor * 1e6 for s in sizes]

    # legend_kw = dict(
    #     loc="lower left",
    #     bbox_to_anchor=(0, -0.2),
    #     labelspacing=0.8,
    #     handletextpad=0,
    #     frameon=False,
    # )

    # pypsa.plot.maps.static.add_legend_circles(
    #     ax[0],
    #     sizes,
    #     labels,
    #     srid=n.srid,
    #     patch_kw=dict(facecolor="lightgrey"),
    #     legend_kw=legend_kw,
    # )
    # sizes = [30, 10]
    # labels = [f"{s} kt" for s in sizes]
    # scale = 1e3 / linewidth_factor
    # sizes = [s * scale for s in sizes]

    # legend_kw = dict(
    #     loc="lower left",
    #     bbox_to_anchor=(0.5, -0.5),
    #     frameon=False,
    #     labelspacing=0.8,
    #     handletextpad=1,
    # )

    # pypsa.plot.maps.static.add_legend_lines(
    #     ax[0],
    #     sizes,
    #     labels,
    #     patch_kw=dict(color="lightgrey"),
    #     legend_kw=legend_kw,
    # )

    # colors = list(bus_colors.values())
    # labels = list(bus_colors.keys())

    # legend_kw = dict(
    #     loc="lower left",
    #     bbox_to_anchor=(-0, -0.3),
    #     ncol=4,
    #     frameon=False,
    # )

    # pypsa.plot.maps.static.add_legend_patches(ax[0], colors, labels, legend_kw=legend_kw)
    # fig.suptitle(f"{scenario} CO2 Emission and Sequestration", fontsize=16)
    # fig.savefig(savepath, bbox_inches="tight")
    # plt.close()



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "plot_comparison",
            simpl="",
            clusters=68,
            planning_horizons=2035,
            opts="",
            ll="vopt",
            sector_opts="None",
            run="h2_network_27",
        )

    # os.makedirs(snakemake.output.report, exist_ok=True)

    configure_logging(snakemake)

    # collect general info
    scenarios = snakemake.config["run"]["name"]
    modelyears = snakemake.config["scenario"]["planning_horizons"]

    regions = regions = gpd.read_file(snakemake.input.regions_onshore).set_index("name")
    regions["country"] = regions.index.str[:2]

    with open("/home/toni-seibold/dev/pypsa-de-co2/config/plotting.default.yaml") as stream:
        plotting = yaml.safe_load(stream)
    tech_colors = plotting["plotting"]["tech_colors"]

    plot_carbon_network_de(
            modelyears[1:],
            scenarios,
            regions,
            # savepath=snakemake.output.report + f"/carbon_network_de.png",
            )