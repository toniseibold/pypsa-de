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
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import unary_union
from shapely.geometry import MultiPoint
from shapely.ops import triangulate

from scripts._helpers import configure_logging, set_scenario_config, load_costs

logger = logging.getLogger(__name__)

DISTANCE_CRS = "EPSG:3035"

CO2_FACTORS = {
    "process emission": 1.0,
    "process emission from feedstock": 1.0,
    "methane": 0.20,
    "gas": 0.20,
    "solid biomass": 0.18,
    "biomass": 0.18,
    "coal": 0.34,
    "coke": 0.39,
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
DEFAULT_PRUNING_ROUNDS = 3


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
    if de.empty:
        raise ValueError("No German regions found (expected index starting with 'DE').")

    nldk = regions[idx.str.startswith("NL") | idx.str.startswith("DK")].copy()
    if nldk.empty:
        logger.warning("No NL/DK regions found in clustered regions input.")
        return de

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
) -> pd.DataFrame:
    """Build candidate graph with weighted edge metrics for one alpha value."""
    nodes = regions_selected.index.astype(str)
    nodes_df = regions_selected.copy()
    nodes_df.index = nodes

    projected = nodes_df.to_crs(DISTANCE_CRS)
    centroids = projected.geometry.centroid
    xy = pd.DataFrame({"x": centroids.x, "y": centroids.y}, index=nodes)

    max_potential = max(float(node_potential.max()), 1e-9)
    node_potential_norm = node_potential / max_potential

    coord_to_nodes: dict[tuple[float, float], list[str]] = {}
    for node in nodes:
        coord = (float(xy.at[node, "x"]), float(xy.at[node, "y"]))
        coord_to_nodes.setdefault(coord, []).append(node)

    unique_coords = list(coord_to_nodes)
    delaunay_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()

    for triangle in triangulate(MultiPoint(unique_coords)):
        tri_coords = [
            (float(x), float(y)) for x, y in list(triangle.exterior.coords)[:3]
        ]
        for i in range(3):
            a = tri_coords[i]
            b = tri_coords[(i + 1) % 3]
            if a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            delaunay_edges.add(edge)

    rows = []
    seen_bus_pairs: set[tuple[str, str]] = set()
    for coord_u, coord_v in sorted(delaunay_edges):
        for u in coord_to_nodes[coord_u]:
            for v in coord_to_nodes[coord_v]:
                if u == v:
                    continue

                bus0, bus1 = (u, v) if u < v else (v, u)
                pair = (bus0, bus1)
                if pair in seen_bus_pairs:
                    continue
                seen_bus_pairs.add(pair)

                dx = xy.at[bus0, "x"] - xy.at[bus1, "x"]
                dy = xy.at[bus0, "y"] - xy.at[bus1, "y"]
                length_km = float(np.hypot(dx, dy) / 1000.0)

                pot_sum = float(
                    node_potential_norm.at[bus0] + node_potential_norm.at[bus1]
                )
                weighted_cost = length_km / ((1.0 + pot_sum) ** alpha)
                rows.append(
                    {
                        "bus0": bus0,
                        "bus1": bus1,
                        "length_km": length_km,
                        "weighted_cost": weighted_cost,
                        "alpha": alpha,
                        "is_interconnector": (
                            bus0.startswith(("NL", "DK"))
                            or bus1.startswith(("NL", "DK"))
                        ),
                    }
                )

    edges = pd.DataFrame(rows)
    to_drop = edges[~(edges.bus0.str.startswith("DE")) & ~(edges.bus1.str.startswith("DE"))].index
    edges = edges.drop(to_drop)
    edges = edges.reset_index(drop=True)
    if edges.empty:
        raise ValueError("Not enough regions to build a topology graph.")
    return edges


def _identify_start_cases(nodes: list[str]) -> dict[str, set[str]]:
    """Build start-point cases from required exit-point prefixes."""
    by_prefix: dict[str, set[str]] = {
        prefix: {node for node in nodes if node.startswith(prefix)}
        for prefix in START_POINT_PREFIXES
    }

    start_cases: dict[str, set[str]] = {
        "de06_only": by_prefix["DE0 6"],
        "de08_only": by_prefix["DE0 8"],
        "nl00_only": by_prefix["NL0 0"],
        "nl04_only": by_prefix["NL0 4"],
        "dk00_only": by_prefix["DK0 0"],
    }
    start_cases["all_start_points"] = set().union(*start_cases.values())

    missing_cases = [name for name, case_nodes in start_cases.items() if not case_nodes]
    if missing_cases:
        raise ValueError(
            "Missing required start-point nodes for cases: "
            + ", ".join(missing_cases)
        )

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
    out.insert(3, "alpha", alpha)
    out.insert(4, "length_limit_km", length_limit_km)
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

    pruning_rounds = int(co2_topology_cfg.get("pruning_rounds", DEFAULT_PRUNING_ROUNDS))

    length_limit_km = float(co2_topology_cfg.get("length_limit_km", DEFAULT_LENGTH_LIMIT_KM))

    logger.info(
        "Building CO2 topologies with length limit %s km, alpha values %s, and pruning rounds %s.",
        length_limit_km,
        alpha_values,
        pruning_rounds,
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
    expected_per_round = len(start_case_items) * len(alpha_values)

    rng = np.random.default_rng(seed=20260422)

    for round_id in range(1, pruning_rounds + 1):
        round_entries: list[dict] = []

        if round_id == 2:
            n_to_remove = 5
        elif round_id >= 3:
            n_to_remove = 7
        else:
            n_to_remove = 0

        candidate_edges: list[int] = []
        if n_to_remove > 0:
            edge_counter: Counter[int] = Counter()
            for defn in definitions:
                edge_counter.update(defn["edges"].index)

            all_candidates = edge_counter.most_common()
            if not all_candidates:
                raise ValueError(
                    f"Cannot prune edges in round {round_id}: no prior edges available."
                )

            threshold_index = min(n_to_remove - 1, len(all_candidates) - 1)
            threshold_count = all_candidates[threshold_index][1]
            candidate_edges = [
                edge_idx for edge_idx, count in all_candidates if count >= threshold_count
            ]

            if len(candidate_edges) < n_to_remove:
                candidate_edges = [edge_idx for edge_idx, _count in all_candidates]

        for start_case, start_nodes in start_case_items:
            for alpha in alpha_values:
                edges = _build_candidate_edges(
                    regions_selected=selected_regions,
                    node_potential=node_components["total_co2_potential_mtco2"],
                    alpha=alpha,
                )

                removed_edge_ids: list[int] = []
                if n_to_remove > 0:
                    if len(candidate_edges) <= n_to_remove:
                        removed_edge_ids = list(candidate_edges)
                    else:
                        removed_edge_ids = list(
                            rng.choice(candidate_edges, size=n_to_remove, replace=False)
                        )

                    removed_set = set(removed_edge_ids)
                    keep_mask = [idx not in removed_set for idx in edges.index]
                    edges = edges.loc[keep_mask].copy()

                if edges.empty:
                    continue

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
                if not selected_idx:
                    continue

                selected_edges = edges.loc[selected_idx].copy()

                round_entries.append(
                    {
                        "algorithm": "weighted_mst",
                        "start_case": start_case,
                        "alpha": float(alpha),
                        "length_limit_km": float(length_limit_km),
                        "start_nodes": set(start_nodes),
                        "all_nodes": all_nodes_set,
                        "edges": selected_edges,
                        "removed_edge_ids": sorted(removed_edge_ids),
                        "round_id": round_id,
                    }
                )

        if not round_entries:
            logger.warning("No more feasible topologies after pruning edges.")
            break

        if len(round_entries) != expected_per_round:
            raise ValueError(
                "Expected "
                f"{expected_per_round} topologies in round {round_id} "
                f"(start_cases={len(start_case_items)}, alpha_values={len(alpha_values)}), "
                f"but generated {len(round_entries)}."
            )

        definitions.extend(round_entries)
        logger.info(
            "Round %s built %s topologies (removed %s edges per topology).",
            round_id,
            len(round_entries),
            n_to_remove,
        )

    summary_rows = []
    for topology_id, definition in enumerate(definitions):
        topology_name = f"topology_{topology_id:03d}"
        variant_name = (
            f"{definition['algorithm']}__{definition['start_case']}"
            f"__alpha{definition['alpha']:g}__{int(definition['length_limit_km'])}km"
            f"__round{definition['round_id']:02d}"
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
        summary_row["round_id"] = definition["round_id"]
        summary_row["n_removed_edges"] = len(definition["removed_edge_ids"])
        summary_row["removed_edge_ids"] = ";".join(
            str(edge_idx) for edge_idx in definition["removed_edge_ids"]
        )
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
        "Wrote %s CO2 topology variants to %s at %s km (alpha values=%s, pruning_rounds=%s).",
        len(definitions),
        output_dir,
        int(length_limit_km),
        alpha_values,
        pruning_rounds,
    )
