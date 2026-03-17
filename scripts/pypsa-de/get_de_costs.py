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


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "get_de_costs",
            simpl="",
            clusters=49,
            planning_horizons=2035,
            opts="",
            ll="vopt",
            sector_opts="none",
            run="no_co2_network",
            column="no_co2_network",
        )

    configure_logging(snakemake)

    n = pypsa.Network(snakemake.input.network)

    revenue = n.statistics.revenue(groupby=["carrier", "bus"]).loc["Load", :, :]
    de_rev = revenue[revenue.index.get_level_values(1).str.startswith("DE")].groupby("carrier").sum()

    de_rev.to_csv(snakemake.output.cost)
