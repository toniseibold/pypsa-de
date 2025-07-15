# -*- coding: utf-8 -*-

import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import pandas as pd

from scripts._helpers import mock_snakemake

# scenario_index = pd.Index(["FT-Import-EU No Reloc", "FT-Import-World No Reloc", "All-Import-EU No Reloc", "All-Import-World No Reloc", "All-Import-EU Reloc", "All-Import-World Reloc"])
scenario_index = pd.Index(["Base", "EH", "WH", "EHP", "WHP"])
# scenario_index = pd.Index(["100 Mt Seq.", "150 Mt Seq.", "200 Mt Seq.", "250 Mt Seq.", "300 Mt Seq.", "350 Mt Seq.", "400 Mt Seq."])
suffix = "" # -50% Electrolysis CAPEX

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
    }
legend={
    'electricity EU': '#ccbb95',
    'hydrogen EU': '#d9b8d3',
    'hydrogen non EU': '#db8ccd',
    'ammonia EU': '#87d0e6',
    'ammonia non EU': '#598896',
    'methanol EU': '#b7dbdb',
    'methanol non EU': '#6a8080',
    'Fischer-Tropsch EU': '#95decb',
    'Fischer-Tropsch non EU': '#598579',
    'hbi EU': '#4682B4',
    'hbi non EU': '#738399',
    'fossil gas': '#edaf1c',
    'fossil oil': '#de571d',
    }

order = [
        'electricity non EU',
        'electricity EU in',
        'electricity EU out',
        'hydrogen EU in',
        'hydrogen EU out',
        'hydrogen non EU',
        'ammonia EU in',
        'ammonia EU out',
        'ammonia non EU',
        'methanol EU in',
        'methanol EU out',
        'methanol non EU',
        'Fischer-Tropsch EU in',
        'Fischer-Tropsch EU out',
        'Fischer-Tropsch non EU',
        'hbi EU in',
        'hbi EU out',
        'hbi non EU',
        'fossil gas',
        'fossil oil',
    ]

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


def plot_consumer_costs(scenarios, savepath):
    # set up dataframe
    data = pd.DataFrame(index=scenarios, columns=['electricity', 'land transport EV', 'land transport fuel cell',
       'land transport oil', 'urban central heat',
       'solid biomass for industry', 'gas for industry', 'H2 for industry',
       'industry methanol', 'shipping methanol', 'shipping oil',
       'ethylene for industry', 'kerosene for aviation',
       'low-temperature heat for industry', 'industry electricity',
       'process emissions', 'NH3', 'coal for industry',
       'agriculture electricity', 'agriculture heat',
       'agriculture machinery oil', 'steel', 'rural heat',
       'urban decentral heat', 'coal for steel'])
    # iterate over all scenarios
    for i, fn in enumerate(snakemake.input.revenues):
        df = pd.read_csv(fn, index_col=0)
        data.loc[scenarios[i], :] = df.loc[2050]
    
    # drop NaN columns - 0 in 2050
    data.drop(columns=["land transport oil", "coal for steel", "shipping oil", "coal for industry"], inplace=True)

    # group together heat, electricity
    data.loc[:, 'electricity'] = data[['electricity', 'industry electricity', 'agriculture electricity']].sum(axis=1)
    data.drop(columns=["industry electricity", "agriculture electricity"], inplace=True)

    data["land transport"] = data[['land transport EV', 'land transport fuel cell']].sum(axis=1)
    data.drop(columns=['land transport EV', 'land transport fuel cell'], inplace=True)
    colors["land transport"] = colors["land transport EV"]

    data["heat"] = data[data.columns[data.columns.str.contains("heat")]].sum(axis=1)
    data.drop(columns=data.columns[data.columns.str.contains(" heat")], inplace=True)

    data = data[data.sum(axis=0).sort_values(ascending=False).index]

    data.index = scenario_index

    fig, ax = plt.subplots(4, 1, figsize=(len(scenarios)*1.7, 8))

    data.abs().div(1e9).plot.bar(ax=ax[0], stacked=True, width=0.8, legend=True, color=[colors[col] for col in data.columns])
    ax[0].set_xticklabels([])
    ax[0].legend(loc="upper center", bbox_to_anchor=(1.2, 1))
    ax[0].set_title(f"Consumer Costs DE 2050 across scenarios{suffix}")
    ax[0].set_ylabel("Bill €")
    ax[0].grid(visible=True, axis="y")

    data[['steel', 'industry methanol', 'NH3']].abs().div(1e9).plot.bar(ax=ax[2], stacked=True, width=0.8, legend=False, color=[colors[col] for col in ['steel', 'industry methanol', 'NH3']])
    ax[2].set_xticklabels([])
    ax[2].set_title("Consumer Costs Industry")
    ax[2].set_ylabel("Bill €")
    ax[2].grid(visible=True, axis="y")

    # savings
    summe = data.sum(axis=1)
    savings = (summe - summe.iloc[0]).div(summe.iloc[0]).mul(100)
    savings.round(2).plot.bar(ax=ax[1], width=0.8)
    ax[1].set_xticklabels([])
    ax[1].set_title(f"Savings")
    ax[1].set_ylabel("[%]")
    ax[1].grid(visible=True, axis="y")


    # industry savings
    summe_ind = data[["steel", "industry methanol", "NH3"]].sum(axis=1)
    savings_ind = (summe_ind - summe_ind.iloc[0]).div(summe_ind.iloc[0]).mul(100)
    savings_ind.round(2).plot.bar(ax=ax[3], width=0.8)
    plt.xticks(rotation=30, ha="right")
    ax[3].set_title(f"Savings")
    ax[3].set_ylabel("[%]")
    ax[3].grid(visible=True, axis="y")

    plt.subplots_adjust(bottom=0.15)
    fig.savefig(savepath+"consumer_costs.svg", bbox_inches="tight")
    plt.close()

def plot_consumer_costs_combined(scenarios, savepath):
    # set up dataframe
    data = pd.DataFrame(index=scenarios, columns=['electricity', 'land transport EV', 'land transport fuel cell','land transport oil', 'urban central heat',
       'solid biomass for industry', 'gas for industry', 'H2 for industry',
       'industry methanol', 'shipping methanol', 'shipping oil',
       'ethylene for industry', 'kerosene for aviation',
       'low-temperature heat for industry', 'industry electricity',
       'process emissions', 'NH3', 'coal for industry',
       'agriculture electricity', 'agriculture heat',
       'agriculture machinery oil', 'steel', 'rural heat',
       'urban decentral heat', 'coal for steel'])
    # iterate over all scenarios
    for i, fn in enumerate(snakemake.input.revenues):
        df = pd.read_csv(fn, index_col=0)
        data.loc[scenarios[i], :] = df.loc[2050]
    
    # drop NaN columns - 0 in 2050
    data.drop(columns=["land transport oil", "coal for steel", "shipping oil", "coal for industry"], inplace=True)

    # group together heat, electricity
    data.loc[:, 'electricity'] = data[['electricity', 'industry electricity', 'agriculture electricity']].sum(axis=1)
    data.drop(columns=["industry electricity", "agriculture electricity"], inplace=True)

    data["land transport"] = data[['land transport EV', 'land transport fuel cell']].sum(axis=1)
    data.drop(columns=['land transport EV', 'land transport fuel cell'], inplace=True)
    colors["land transport"] = colors["land transport EV"]

    data["bunkers"] = data[['kerosene for aviation', 'shipping methanol']].sum(axis=1)
    data.drop(columns=['kerosene for aviation', 'shipping methanol'], inplace=True)
    colors["bunkers"] = colors["kerosene for aviation"]

    data["other"] = data[['process emissions', 'gas for industry', 'agriculture machinery oil', 'H2 for industry', 'solid biomass for industry']].sum(axis=1)
    data.drop(columns=['process emissions', 'gas for industry', 'agriculture machinery oil', 'H2 for industry', 'solid biomass for industry'], inplace=True)
    colors["other"] = "black"

    data["heat"] = data[data.columns[data.columns.str.contains("heat")]].sum(axis=1)
    data.drop(columns=data.columns[data.columns.str.contains(" heat")], inplace=True)

    data = data.rename(columns={'ethylene for industry': 'HVC'})
    colors["HVC"] = colors["ethylene for industry"]

    pref_order = ["electricity", "heat", "land transport", "bunkers", "other", "steel", "HVC", "NH3", "industry methanol"]
    data = data[pref_order]
    data.index = scenario_index
    bottom = data.iloc[:, :5].abs().div(1e9).sum(axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(len(scenarios)*1.7, 8))
    data.iloc[:, :5].abs().div(1e9).plot.bar(ax=ax[0], stacked=True, width=0.8, 
                                        legend=True, alpha=1,
                                        color=[colors[col] for col in data.iloc[:, :5].columns])
    for i in range(len(data)):
        ax[0].plot([i - 0.4, i + 0.4], [bottom.iloc[i]] * 2, color='black', linestyle='dotted', linewidth=1.5)

    data.iloc[:, 5:].abs().div(1e9).plot.bar(ax=ax[0], stacked=True, width=0.8, bottom=bottom,legend=True, color=[colors[col] for col in data.iloc[:, 5:].columns])
    ax[0].set_xticklabels([])
    ax[0].legend(loc="upper center", bbox_to_anchor=(1.2, 1))
    ax[0].set_title(f"Consumer Costs DE 2050 across scenarios{suffix}")
    ax[0].set_ylabel("Bill €")
    ax[0].grid(visible=True, axis="y")

    # savings total
    summe = data.sum(axis=1)
    savings = -(summe - summe.iloc[0]).div(summe.iloc[0]).mul(100)
    savings.round(2).plot(ax=ax[1], label="total")

    # industry savings
    summe_ind = data[["steel", "industry methanol", "NH3", "HVC"]].sum(axis=1)
    savings_ind = -(summe_ind - summe_ind.iloc[0]).div(summe_ind.iloc[0]).mul(100)
    savings_ind.round(2).plot(ax=ax[1], label="industry")
    plt.xticks(rotation=30, ha="right")
    ax[1].set_title(f"Savings")
    ax[1].set_ylabel("[%]")
    ax[1].grid(visible=True, axis="y")

    plt.subplots_adjust(bottom=0.15)
    fig.savefig(savepath+"consumer_costs.svg", bbox_inches="tight")
    plt.close()


def plot_carrier_prices(scenarios, savepath):
    multi_ind = [scenarios, ["mean", "min", "max"]]
    # set up dataframe
    data = pd.DataFrame(index=pd.MultiIndex.from_product(multi_ind, names=["scenario", "price"]), columns=['steel', 'NH3', 'methanol'])
    # iterate over all scenarios
    for i, fn in enumerate(snakemake.input.prices):
        df = pd.read_csv(fn, index_col=0)
        data.loc[scenarios[i], :] = df.T.values

    mean_data = data.xs("mean", level="price")
    mean_data.index = scenario_index

    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    for i, carrier in enumerate(['steel', 'NH3', 'methanol']):
        mean_data[carrier].plot.bar(ax=ax[i], title=carrier+suffix)
        ax[i].grid(visible=True, axis="y")
    ax[0].set_ylabel("€/MWh(t)")


def plot_carbon_balances(scenarios, savepath):
    data = pd.DataFrame(index=scenarios, columns=['CCGT methanol CC', 'DAC', 'Fischer-Tropsch',
       'Methanol steam reforming CC', 'SMR CC', 'Sabatier', 'allam methanol',
       'biogas to gas CC', 'biomass to liquid CC', 'biomass-to-methanol CC',
       'co2 sequestered', 'gas for industry CC', 'methanolisation',
       'process emissions CC', 'solid biomass for industry CC',
       'urban central gas CHP CC', 'urban central solid biomass CHP CC',
       'waste CHP CC', 'Trade'])
    for i, fn in enumerate(snakemake.input.co2):
        df = pd.read_csv(fn, index_col=0)
        data.loc[scenarios[i], :] = df.loc[0]
    data.index = scenario_index
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    # iterate over all scenarios
    data.plot.barh(ax=ax, stacked=True, width=0.8, legend=True, color=[colors[col] for col in data.columns])
    ax.legend(loc="upper center", bbox_to_anchor=(1.25, 1))
    ax.set_title(f"CO2 Balance DE 2050 across scenarios{suffix}")
    ax.set_xlabel("Mt CO2")
    ax.grid(visible=True, axis="x")

    fig.savefig(savepath+"co2_balance.pdf", bbox_inches="tight")
    plt.close()


def plot_hydrogen_balances(scenarios, savepath):
    data = pd.DataFrame(index=scenarios, columns=["Electrolysis", "SMR", "SMR CC", "Storage discharge", "Import", "Non EU Import", "H2 for industry", "DRI", "Haber-Bosch", "methanolisation", "Fischer-Tropsch", "H2 OCGT", "H2 CHP", "Storage charge", "Export"])
    for i, fn in enumerate(snakemake.input.h2):
        df = pd.read_csv(fn, index_col=0)
        data.loc[scenarios[i], :] = df.T.iloc[0]
    
    data.index = scenario_index

    data = data.fillna(0)
    data.drop(columns=["Storage discharge", "Storage charge"], inplace=True)
    h2_order = [
        'Electrolysis', 'SMR', 'SMR CC', 'Import', 'Non EU Import', 'H2 for industry',
       'DRI', 'Haber-Bosch', 'methanolisation', 'Fischer-Tropsch', 'H2 OCGT',
       'H2 CHP', 'Export'
    ]
    colors["H2 for industry"] = "#ad539e"
    data = data[[col for col in h2_order if col in data.columns]]
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 6))
    # iterate over all scenarios
    data.plot.barh(ax=ax, stacked=True, width=0.8, legend=True, color=[colors[col] for col in data.columns])
    ax.legend(loc="upper center", bbox_to_anchor=(1.1, 1))
    ax.set_title(f"H2 Balance DE 2050 across scenarios{suffix}")
    ax.set_xlabel("TWh H2")
    ax.grid(visible=True, axis="x")

    fig.savefig(savepath+"h2_balance.pdf", bbox_inches="tight")
    plt.close()


def plot_import_balances(scenarios, savepath):
    # set up dataframe
    data = pd.DataFrame(columns=scenarios)
    # iterate over all scenarios
    for i, fn in enumerate(snakemake.input.import_vol):
        df = pd.read_csv(fn, index_col=0)
        # Ensure data contains all indices from df
        data = data.reindex(data.index.union(df.index), fill_value=0)
        
        data[scenarios[i]] = df["2050"]

    data.columns = scenario_index
    data = data.fillna(0).T
    
    data = data[[col for col in order if col in data.columns]]
    data = data.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, len(scenarios)*1.25))
    data.div(1e6).plot.barh(ax=ax, stacked=True, width=0.8, legend=False, color=[color_palette[col] for col in data.columns])


    colors = list(legend.values())
    labels = list(legend.keys())

    legend_handles = [Patch(color=color, label=label) for color, label in zip(colors, labels)]

    fig.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(0.9, 0.65), ncol=1)

    
    ax.set_title(f"Import DE 2050 across scenarios{suffix}")
    ax.set_xlabel("TWh")
    ax.set_xlim([-200, 900])
    ax.grid(visible=True, axis="x")

    fig.savefig(savepath+"import_balance.svg", bbox_inches="tight")
    plt.close()



if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "import_all",
        )

    scenarios = snakemake.config["run"]["name"]

    colors = prepare_colors()

    prefix = snakemake.config["run"]["prefix"]
    savepath = f"results/{prefix}/comparison/"
    os.makedirs(savepath, exist_ok=True)


    plot_consumer_costs(scenarios, savepath)

    plot_carrier_prices(scenarios, savepath)

    plot_carbon_balances(scenarios, savepath)

    plot_hydrogen_balances(scenarios, savepath)

    plot_import_balances(scenarios, savepath)