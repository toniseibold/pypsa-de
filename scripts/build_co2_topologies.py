# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build graph-based CO2 network topology variants focused on Germany.

The script creates multiple candidate topologies for a sweep of total network
length budgets. Topologies are rooted in at least one available sink region:
adjacent NL/DK regions and the German North Sea exit regions DE0 6 / DE0 8.

Node importance is weighted by capturable CO2 potential from industrial demand
plus waste-incineration potential derived from naphtha demand and distributed by
population.
"""

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import unary_union

from scripts._helpers import configure_logging, set_scenario_config, load_costs

logger = logging.getLogger(__name__)

DISTANCE_CRS = "EPSG:3035"

CO2_FACTORS = {
    "process emission": 1.0,
    "process emission from feedstock": 1.0,
    # "methane": 0.20,
    # "gas": 0.20,
    "solid biomass": 0.18,
    "biomass": 0.18,
    # "coal": 0.34,
    # "coke": 0.39,
}

START_POINT_PREFIXES = (
    "DE0 6",
    "DE0 8",
    "NL0 0",
    "NL0 4",
    "DK0 0",
)

DEFAULT_ALPHA_VALUES = (0.5, 2.0)
DEFAULT_LENGTH_LIMIT_KM = 4000.0


def _find_matching_column(df: pd.DataFrame, key: str) -> str | None:
    """Find a demand column by case-insensitive exact/substring matching."""
    lower_to_original = {column.lower(): column for column in df.columns}
    key_lower = key.lower()
    if key_lower in lower_to_original:
        return lower_to_original[key_lower]

    contains = [column for column in df.columns if key_lower in column.lower()]
    if len(contains) == 1:
        return contains[0]

    begins = [column for column in contains if column.lower().startswith(key_lower)]
    if begins:
        return begins[0]

    return contains[0] if contains else None


def _get_scalar_by_year(value: float | dict, year: int) -> float:
    """Resolve either a scalar or a year-indexed mapping."""
    if isinstance(value, dict):
        if year in value:
            return float(value[year])
        if str(year) in value:
            return float(value[str(year)])

        available_years = sorted(int(item) for item in value)
        fallback_year = max(item for item in available_years if item <= year)
        if str(fallback_year) in value:
            return float(value[str(fallback_year)])
        return float(value[fallback_year])

    return float(value)


def _get_population_series(pop_layout: pd.DataFrame) -> pd.Series:
    """Extract total population per node from the clustered population layout."""
    if "total" in pop_layout.columns:
        return pd.to_numeric(pop_layout["total"], errors="coerce").fillna(0.0)

    numeric_columns = pop_layout.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        return pd.to_numeric(
            pop_layout[numeric_columns[0]], errors="coerce"
        ).fillna(0.0)

    raise ValueError("Population layout has no usable numeric population column.")


def _compute_industrial_co2_potential(industrial_demand: pd.DataFrame) -> pd.Series:
    """Estimate capturable industrial CO2 potential in MtCO2/a for each region."""
    potential = pd.Series(0.0, index=industrial_demand.index)
    matched_columns: set[str] = set()

    for key, factor in CO2_FACTORS.items():
        column = _find_matching_column(industrial_demand, key)
        if column is None or column in matched_columns:
            continue
        matched_columns.add(column)
        potential = potential.add(
            pd.to_numeric(industrial_demand[column], errors="coerce").fillna(0.0)
            * factor,
            fill_value=0.0,
        )

    logger.info("Industrial CO2 potential uses columns: %s", sorted(matched_columns))
    return potential.clip(lower=0.0)


def _compute_hvc_waste_co2_potential(
    industrial_demand: pd.DataFrame,
    pop_layout: pd.DataFrame,
    costs: pd.Series,
    config: dict,
    planning_horizon: int,
) -> pd.Series:
    """Estimate capturable CO2 potential from non-sequestered HVC waste incineration."""
    potential = pd.Series(0.0, index=industrial_demand.index)

    naphtha_column = _find_matching_column(industrial_demand, "naphtha")
    feedstock_column = _find_matching_column(
        industrial_demand, "process emission from feedstock"
    )
    if naphtha_column is None or feedstock_column is None:
        logger.info("Skipping HVC waste potential because required columns are missing.")
        return potential

    de_nodes = industrial_demand.index[
        industrial_demand.index.astype(str).str.startswith("DE")
    ]
    if len(de_nodes) == 0:
        return potential

    naphtha_demand = (
        pd.to_numeric(industrial_demand.loc[de_nodes, naphtha_column], errors="coerce")
        .fillna(0.0)
        .sum()
    )
    if naphtha_demand <= 0:
        return potential

    feedstock_process_emissions = (
        pd.to_numeric(
            industrial_demand.loc[de_nodes, feedstock_column], errors="coerce"
        )
        .fillna(0.0)
        .sum()
    )
    process_co2_per_naphtha = feedstock_process_emissions / naphtha_demand

    oil_co2_intensity = float(costs.loc[("oil", "CO2 intensity")])
    hvc_per_naphtha = (
        oil_co2_intensity - process_co2_per_naphtha
    ) / oil_co2_intensity

    non_sequestered_fraction = 1.0 - _get_scalar_by_year(
        config["industry"].get("HVC_environment_sequestration_fraction", 0.0),
        planning_horizon,
    )
    hvc_energy = naphtha_demand * non_sequestered_fraction * hvc_per_naphtha
    hvc_co2_total = hvc_energy * oil_co2_intensity
    if hvc_co2_total <= 0:
        return potential

    population = _get_population_series(pop_layout).reindex(de_nodes).fillna(0.0)
    population = population[population > 0]
    if population.empty:
        logger.warning("Skipping HVC waste potential because population shares are empty.")
        return potential

    shares = population / population.sum()
    potential.loc[shares.index] = shares * hvc_co2_total

    logger.info(
        "Added %.3f MtCO2/a HVC waste potential using HVC_per_naphtha %.3f and non-sequestered fraction %.3f.",
        hvc_co2_total,
        hvc_per_naphtha,
        non_sequestered_fraction,
    )
    return potential.clip(lower=0.0)


def _select_regions(regions: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep DE regions and adjacent NL/DK interconnector regions."""
    idx = regions.index.astype(str)

    de = regions[idx.str.startswith("DE")].copy()

    nldk = regions[regions.index.isin(["NL0 0", "DK0 0"])].copy()

    de_proj = de.to_crs(DISTANCE_CRS)
    nldk_proj = nldk.to_crs(DISTANCE_CRS)

    de_union = unary_union(de_proj.geometry)
    adjacent_mask = nldk_proj.geometry.intersects(de_union.buffer(1000.0))
    interconnectors = nldk.loc[adjacent_mask.values].copy()

    nldk_proj = nldk_proj.copy()
    nldk_proj["dist_to_de"] = nldk_proj.geometry.distance(de_union)
    for prefix in ["NL", "DK"]:
        has_prefix = interconnectors.index.astype(str).str.startswith(prefix).any()
        if has_prefix:
            continue
        grp = nldk_proj[nldk_proj.index.astype(str).str.startswith(prefix)]
        if not grp.empty:
            nearest = grp["dist_to_de"].idxmin()
            interconnectors = pd.concat([interconnectors, nldk.loc[[nearest]]])

    selected = pd.concat([de, interconnectors])
    selected = selected.loc[~selected.index.duplicated()]
    logger.info(
        "Selected %s DE regions and %s interconnector regions (NL/DK).",
        len(de),
        len(selected) - len(de),
    )
    return selected


def _build_candidate_edges(
    regions_selected: gpd.GeoDataFrame,
    node_potential: pd.Series,
    alpha: float,
    start_nodes: set[str],
    penalty_buses: set[str] | None = None,
) -> pd.DataFrame:
    """Build Gabriel-filtered Delaunay candidate edges with weighted cost for one alpha."""
    from scripts.build_transmission_topology import (
        delaunay_triangulation,
        prepare_candidate_edges,
    )

    gabriel_filter_enabled = True
    min_degree = 1
    length_factor = 1.25

    nodes = regions_selected.index.astype(str)
    geo = regions_selected.to_crs("EPSG:4326")
    meter = regions_selected.to_crs(DISTANCE_CRS)

    centroids_geo = geo.geometry.centroid
    centroids_meter = meter.geometry.centroid

    coords_geo = np.column_stack(
        [centroids_geo.x.to_numpy(), centroids_geo.y.to_numpy()]
    ).astype(float)
    coords_meter = np.column_stack(
        [centroids_meter.x.to_numpy(), centroids_meter.y.to_numpy()]
    ).astype(float)

    delaunay_graph = delaunay_triangulation(
        bus_ids=nodes,
        coords_geo=coords_geo,
        coords_meter=coords_meter,
        length_factor=length_factor,
    )

    selected = prepare_candidate_edges(
        delaunay_graph=delaunay_graph,
        gabriel_filter_enabled=gabriel_filter_enabled,
        node_count=len(nodes),
        min_degree=min_degree,
    )

    potential = node_potential.copy()
    if penalty_buses:
        for penalty_bus in penalty_buses:
            if penalty_bus in nodes and penalty_bus.startswith("DE"):
                potential.loc[penalty_bus] = potential.get(penalty_bus, 0.0) * 0.2

    max_potential = max(float(potential.max()), 1e-9)
    node_potential_norm = potential / max_potential

    def weighted_cost(row: pd.Series) -> float:
        pot_sum = float(
            node_potential_norm.get(row["bus0"], 0.0)
            + node_potential_norm.get(row["bus1"], 0.0)
        )
        return float(row["length"]) / ((1.0 + pot_sum) ** alpha)

    edges = selected[["bus0", "bus1", "length"]].copy().rename(
        columns={"length": "length_km"}
    )
    edges["weighted_cost"] = selected.apply(weighted_cost, axis=1)
    edges["alpha"] = alpha
    edges["is_interconnector"] = (
        edges["bus0"].str.startswith(("NL", "DK"))
        | edges["bus1"].str.startswith(("NL", "DK"))
    )
    edges = edges.reset_index(drop=True)

    # remove exit edges that are not in start_nodes
    if "DE0 6" in start_nodes:
        edges = edges[~edges.is_interconnector]
    elif "NL0 0" in start_nodes:
        edges = edges[~(edges.bus0=="DK0 0") & ~(edges.bus1=="DK0 0")]
    elif "DK0 0" in start_nodes:
        edges = edges[~(edges.bus0=="NL0 0") & ~(edges.bus1=="NL0 0")]

    return edges


def _identify_start_cases(nodes: list[str]) -> dict[str, set[str]]:
    """Build start-point cases from required exit-point prefixes."""
    by_prefix: dict[str, set[str]] = {
        prefix: {node for node in nodes if node.startswith(prefix)}
        for prefix in START_POINT_PREFIXES
    }

    start_cases: dict[str, set[str]] = {
        "de06_only": by_prefix["DE0 6"],
        "nl00_only": by_prefix["NL0 0"],
        "dk00_only": by_prefix["DK0 0"],
    }

    logger.info(
        "Using start cases: %s",
        {name: sorted(values) for name, values in start_cases.items()},
    )
    return start_cases


def _select_rooted_tree(
    edges: pd.DataFrame,
    ordered_index: list[int],
    start_nodes: set[str],
    max_total_length_km: float,
    node_potential: pd.Series | None = None,
) -> list[int]:
    """Grow a rooted tree from the selected start-point set and length budget."""
    if not start_nodes:
        return []

    eligible_index = ordered_index
    active_start_nodes = set(start_nodes)

    # Rank active sink nodes by potential (descending) to prefer best exit point.
    if node_potential is not None:
        ranked_starts = sorted(
            active_start_nodes,
            key=lambda node: node_potential.get(node, 0.0),
            reverse=True,
        )
    else:
        ranked_starts = sorted(active_start_nodes)

    # Start from the best available start point (highest potential).
    connected = {ranked_starts[0]}
    used_edges: set[int] = set()
    selected: list[int] = []
    total_length = 0.0

    # Force at least one edge from the selected start point to the rest.
    root_start = ranked_starts[0]
    if root_start.startswith(("NL", "DK")):
        # For international exits, force the nearest direct DE connection first.
        anchor_candidates: list[tuple[float, int, str, str]] = []
        for edge_idx in eligible_index:
            edge = edges.loc[edge_idx]
            u = edge["bus0"]
            v = edge["bus1"]
            if u == root_start and v.startswith("DE"):
                anchor_candidates.append((float(edge["length_km"]), edge_idx, u, v))
            elif v == root_start and u.startswith("DE"):
                anchor_candidates.append((float(edge["length_km"]), edge_idx, u, v))

        if not anchor_candidates:
            return []

        _, edge_idx, u, v = min(anchor_candidates, key=lambda item: item[0])
        length = float(edges.at[edge_idx, "length_km"])
        if total_length + length > max_total_length_km:
            return []

        connected.update([u, v])
        used_edges.add(edge_idx)
        selected.append(edge_idx)
        total_length += length
    else:
        for target_start in ranked_starts:
            for edge_idx in eligible_index:
                edge = edges.loc[edge_idx]
                u = edge["bus0"]
                v = edge["bus1"]
                if u == target_start and v not in connected:
                    edge_connects_target = True
                elif v == target_start and u not in connected:
                    edge_connects_target = True
                else:
                    edge_connects_target = False

                if not edge_connects_target:
                    continue

                length = float(edge["length_km"])
                if total_length + length > max_total_length_km:
                    continue

                connected.update([u, v])
                used_edges.add(edge_idx)
                selected.append(edge_idx)
                total_length += length
                break
            if selected:
                break

    while True:
        added = False
        for edge_idx in eligible_index:
            if edge_idx in used_edges:
                continue

            edge = edges.loc[edge_idx]
            u = edge["bus0"]
            v = edge["bus1"]
            u_in = u in connected
            v_in = v in connected
            if u_in == v_in:
                continue

            length = float(edge["length_km"])
            if total_length + length > max_total_length_km:
                continue

            connected.update([u, v])
            used_edges.add(edge_idx)
            selected.append(edge_idx)
            total_length += length
            added = True
            break

        if not added:
            break

    return selected


def _topology_metrics(
    nodes: list[str],
    edge_df: pd.DataFrame,
    sink_nodes: set[str],
) -> tuple[int, float, int, int]:
    """Return topology metrics for selected edges."""
    graph = {node: set() for node in nodes}
    for row in edge_df.itertuples(index=False):
        graph[row.bus0].add(row.bus1)
        graph[row.bus1].add(row.bus0)

    visited = set()
    n_components = 0
    sinked_components = 0
    reached_nodes = 0

    for node in nodes:
        if node in visited:
            continue
        n_components += 1
        stack = [node]
        component = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            stack.extend(graph[current] - visited)

        if component & sink_nodes:
            sinked_components += 1
        if len(component) > 1 or (component & sink_nodes):
            reached_nodes += len(component)

    total_length_km = float(edge_df["length_km"].sum()) if not edge_df.empty else 0.0
    return n_components, total_length_km, sinked_components, reached_nodes


def _write_topology(
    topologies_dir: Path,
    topology_name: str,
    algorithm: str,
    start_case: str,
    alpha: float,
    length_limit_km: float,
    start_nodes: set[str],
    all_nodes: set[str],
    edges: pd.DataFrame,
    node_components: pd.DataFrame,
) -> dict:
    """Write one topology edge list and return summary stats."""
    out = edges.copy()
    out.insert(0, "topology", topology_name)
    out.insert(1, "algorithm", algorithm)
    out.insert(2, "start_case", start_case)
    out.insert(3, "length_limit_km", length_limit_km)
    out["bus0_co2_potential_mtco2"] = out["bus0"].map(
        node_components["total_co2_potential_mtco2"]
    )
    out["bus1_co2_potential_mtco2"] = out["bus1"].map(
        node_components["total_co2_potential_mtco2"]
    )

    out_file = topologies_dir / f"{topology_name}.csv"
    out.to_csv(out_file, index=False)

    metrics = _topology_metrics(sorted(all_nodes), out, start_nodes)
    n_components, total_length_km, sinked_components, reached_nodes = metrics
    included_nodes = set(out["bus0"]).union(out["bus1"]).union(start_nodes)

    return {
        "topology": topology_name,
        "algorithm": algorithm,
        "start_case": start_case,
        "alpha": alpha,
        "length_limit_km": length_limit_km,
        "n_edges": int(len(out)),
        "n_nodes_included": int(len(included_nodes)),
        "n_reached_nodes": int(reached_nodes),
        "total_length_km": total_length_km,
        "n_connected_components": int(n_components),
        "n_sinked_components": int(sinked_components),
        "n_interconnector_edges": (
            int(out["is_interconnector"].sum()) if not out.empty else 0
        ),
        "n_start_nodes_available": int(len(start_nodes)),
        "start_nodes": ";".join(sorted(start_nodes)),
        "file": out_file.name,
    }



if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "build_co2_topologies",
            clusters=89,
            planning_horizons=2035,
            run="endogenous",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    planning_horizon = 2035

    co2_topology_cfg = snakemake.params.co2_topology
    alpha_values = co2_topology_cfg.get("alpha_values", list(DEFAULT_ALPHA_VALUES))
    alpha_values = [float(value) for value in alpha_values]
    alpha_values = [0.5, 2.0]
    length_limit_km = float(co2_topology_cfg.get("length_limit_km", DEFAULT_LENGTH_LIMIT_KM))

    logger.info(
        "Building CO2 topologies with length limit %s km and alpha values %s.",
        length_limit_km,
        alpha_values,
    )

    regions = gpd.read_file(snakemake.input.regions).set_index("name")
    industrial_demand = pd.read_csv(snakemake.input.industrial_demand, index_col=0)
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    costs = load_costs(snakemake.input.costs)

    selected_regions = _select_regions(regions)
    selected_nodes = selected_regions.index.astype(str)

    industrial_co2 = _compute_industrial_co2_potential(industrial_demand)
    hvc_waste_co2 = _compute_hvc_waste_co2_potential(
        industrial_demand=industrial_demand,
        pop_layout=pop_layout,
        costs=costs,
        config=snakemake.config,
        planning_horizon=planning_horizon,
    )

    node_components = pd.DataFrame(index=selected_nodes)
    node_components["industrial_co2_potential_mtco2"] = industrial_co2.reindex(
        selected_nodes
    ).fillna(0.0)
    node_components["waste_incineration_co2_potential_mtco2"] = hvc_waste_co2.reindex(
        selected_nodes
    ).fillna(0.0)
    node_components["total_co2_potential_mtco2"] = (
        node_components["industrial_co2_potential_mtco2"]
        + node_components["waste_incineration_co2_potential_mtco2"]
    )

    population = _get_population_series(pop_layout).reindex(selected_nodes).fillna(0.0)
    node_components["population_total"] = population

    output_dir = Path(snakemake.output.topologies)
    topologies_dir = output_dir / "topologies"
    manifest_dir = output_dir / "manifest"
    topologies_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    start_cases = _identify_start_cases(selected_nodes.tolist())
    start_union = set().union(*start_cases.values())
    node_components["is_start_point"] = node_components.index.isin(start_union)
    node_components.to_csv(manifest_dir / "node_co2_potential.csv")

    start_case_items = list(start_cases.items())
    all_nodes_set = set(selected_nodes.tolist())
    definitions: list[dict] = []
    rng = np.random.default_rng(seed=20260423)

    for start_case, start_nodes in start_case_items:
        for alpha in alpha_values:
            edges = _build_candidate_edges(
                regions_selected=selected_regions,
                node_potential=node_components["total_co2_potential_mtco2"],
                alpha=alpha,
                start_nodes=start_nodes,
                penalty_buses=None,
            )

            weighted_order = (
                edges.sort_values("weighted_cost", ascending=True).index.tolist()
            )
            selected_idx = _select_rooted_tree(
                edges,
                weighted_order,
                start_nodes=start_nodes,
                max_total_length_km=length_limit_km,
                node_potential=node_components["total_co2_potential_mtco2"],
            )

            selected_edges = edges.loc[selected_idx].copy()
            
            definitions.append(
                {
                    "algorithm": "weighted_mst",
                    "start_case": start_case,
                    "alpha": float(alpha),
                    "length_limit_km": float(length_limit_km),
                    "start_nodes": set(start_nodes),
                    "all_nodes": all_nodes_set,
                    "edges": selected_edges,
                    "penalty_buses": [],
                }
            )

            # get all connected buses
            connected_buses = set()
            for row in selected_edges.itertuples(index=False):
                connected_buses.add(row.bus0)
                connected_buses.add(row.bus1)
            # select only the ones starting with DE
            connected_buses = {bus for bus in connected_buses if bus.startswith("DE")}

            connected_buses_sorted = sorted(connected_buses)
            if not connected_buses_sorted:
                continue

            base_edge_set = frozenset(
                zip(selected_edges["bus0"], selected_edges["bus1"])
            )
            seen_edge_sets = {base_edge_set}

            sample_size = min(3, len(connected_buses_sorted))
            n_penalized_target = 3
            n_penalized_added = 0
            max_attempts = 15
            attempt = 0
            while n_penalized_added < n_penalized_target and attempt < max_attempts:
                attempt += 1
                penalized_bus_list = sorted(
                    rng.choice(connected_buses_sorted, size=sample_size, replace=False).tolist()
                )
                penalized_bus_set = set(penalized_bus_list)
                edges_p = _build_candidate_edges(
                    regions_selected=selected_regions,
                    node_potential=node_components["total_co2_potential_mtco2"],
                    alpha=alpha,
                    start_nodes=start_nodes,
                    penalty_buses=penalized_bus_set,
                )
                weighted_order = (
                    edges_p.sort_values("weighted_cost", ascending=True).index.tolist()
                )
                selected_idx = _select_rooted_tree(
                    edges_p,
                    weighted_order,
                    start_nodes=start_nodes,
                    max_total_length_km=length_limit_km,
                    node_potential=node_components["total_co2_potential_mtco2"],
                )
                if not selected_idx:
                    continue

                penalized_edges = edges_p.loc[selected_idx].copy()
                penalized_edge_set = frozenset(
                    zip(penalized_edges["bus0"], penalized_edges["bus1"])
                )
                if penalized_edge_set in seen_edge_sets:
                    logger.info(
                        "Penalized topology (%s, alpha=%s, penalty=%s) is identical to "
                        "an existing topology — choosing different penalized regions.",
                        start_case,
                        alpha,
                        penalized_bus_list,
                    )
                    continue

                seen_edge_sets.add(penalized_edge_set)
                n_penalized_added += 1
                definitions.append(
                    {
                        "algorithm": "weighted_mst",
                        "start_case": start_case,
                        "alpha": float(alpha),
                        "length_limit_km": float(length_limit_km),
                        "start_nodes": set(start_nodes),
                        "all_nodes": all_nodes_set,
                        "edges": penalized_edges,
                        "penalty_buses": penalized_bus_list,
                    }
                )

    logger.info(
        "Built %s CO2 topology variants for start cases %s and alphas %s.",
        len(definitions),
        [name for name, _ in start_case_items],
        alpha_values,
    )

    summary_rows = []
    for topology_id, definition in enumerate(definitions):
        topology_name = f"topology_{topology_id:03d}"
        penalty_suffix = ""
        if definition.get("penalty_buses"):
            penalty_suffix = "__penalty_" + "-".join(
                bus.replace(" ", "_") for bus in definition["penalty_buses"]
            )
        variant_name = (
            f"{definition['algorithm']}__{definition['start_case']}"
            f"__alpha{definition['alpha']:g}__{int(definition['length_limit_km'])}km"
            f"{penalty_suffix}"
        )

        summary_row = _write_topology(
            topologies_dir=topologies_dir,
            topology_name=topology_name,
            algorithm=definition["algorithm"],
            start_case=definition["start_case"],
            alpha=definition["alpha"],
            length_limit_km=definition["length_limit_km"],
            start_nodes=definition["start_nodes"],
            all_nodes=definition["all_nodes"],
            edges=definition["edges"],
            node_components=node_components,
        )
        summary_row["variant_name"] = variant_name
        summary_row["penalty_buses"] = ";".join(definition.get("penalty_buses", []))
        summary_rows.append(summary_row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(manifest_dir / "summary.csv", index=False)

    topology_index = summary[["topology", "file"]].copy()
    topology_index["topology_file"] = topology_index["file"].map(
        lambda file_name: str((Path("topologies") / file_name).as_posix())
    )
    topology_index.to_csv(manifest_dir / "topology_index.csv", index=False)

    with open(manifest_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary_rows, file, indent=2)

    with open(manifest_dir / "start_cases.json", "w", encoding="utf-8") as file:
        json.dump(
            {name: sorted(nodes) for name, nodes in start_cases.items()},
            file,
            indent=2,
        )

    (manifest_dir / ".ready").touch()

    logger.info(
        "Wrote %s CO2 topology variants to %s at %s km (alpha values=%s).",
        len(definitions),
        output_dir,
        int(length_limit_km),
        alpha_values,
    )
