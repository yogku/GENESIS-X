"""Knowledge graph utilities for the Space Biology dashboard.

This module constructs a simple graph from the experiment metadata without
relying on external graph libraries such as NetworkX. Each node is
represented by a dictionary storing its label and category, and edges are
stored as pairs of node identifiers. Positions for plotting are generated
using a deterministic circular layout for experiments and radial layouts for
their neighbours. Plotly is used to visualise the graph.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import plotly.graph_objects as go


# Type aliases for clarity
NodeID = str
Nodes = Dict[NodeID, Dict[str, str]]  # maps node id -> attributes
Edges = List[Tuple[NodeID, NodeID]]


def _extract_concepts(text: str) -> Set[str]:
    """Extract a set of simple concept keywords from a text string.

    A simple keyword search is used to identify recurring themes in the
    experiment descriptions. Keywords were hand‑selected to reflect common
    biological and technical concepts relevant to space biology.

    Args:
        text: A sentence or paragraph describing the experiment.

    Returns:
        A set of lower‑cased concept keywords found in the text.
    """
    keywords = [
        "microgravity",
        "gene",
        "photosynthesis",
        "hydroponic",
        "aeroponic",
        "epigenetic",
        "calcium",
        "nutritional",
        "production",
        "transplant",
        "microbe",
        "flavour",
        "epigenetics",
        "radiation",
    ]
    text_lower = text.lower()
    return {kw for kw in keywords if kw in text_lower}


class KnowledgeGraph:
    """A simple container for nodes and edges representing experiment relations."""

    def __init__(self, nodes: Optional[Nodes] = None, edges: Optional[Edges] = None) -> None:
        self.nodes: Nodes = nodes or {}
        self.edges: Edges = edges or []

    def add_node(self, node_id: NodeID, label: str, category: str) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {"label": label, "category": category}

    def add_edge(self, source: NodeID, target: NodeID) -> None:
        self.edges.append((source, target))

    def neighbors(self, node_id: NodeID) -> List[NodeID]:
        neigh = []
        for u, v in self.edges:
            if u == node_id:
                neigh.append(v)
            elif v == node_id:
                neigh.append(u)
        return neigh

    def subgraph(self, nodes_to_keep: Iterable[NodeID]) -> 'KnowledgeGraph':
        sub_nodes: Nodes = {n: self.nodes[n] for n in nodes_to_keep}
        sub_edges: Edges = [e for e in self.edges if e[0] in nodes_to_keep and e[1] in nodes_to_keep]
        return KnowledgeGraph(sub_nodes, sub_edges)


def build_graph(df: pd.DataFrame) -> KnowledgeGraph:
    """Construct a KnowledgeGraph from a DataFrame.

    Args:
        df: DataFrame containing experiment metadata with columns: id, name,
            organisms, objective, summary.

    Returns:
        A KnowledgeGraph instance populated with nodes and edges.
    """
    graph = KnowledgeGraph()
    for _, row in df.iterrows():
        exp_node = f"exp_{row['id']}"
        graph.add_node(exp_node, label=row['name'], category='experiment')

        # Split organisms by comma or semicolon
        orgs = re.split(r"\s*;\s*|\s*,\s*", str(row['organisms']))
        for org in orgs:
            org = org.strip()
            if not org:
                continue
            org_node = f"org_{org.lower().replace(' ', '_')}"
            graph.add_node(org_node, label=org, category='organism')
            graph.add_edge(exp_node, org_node)

        # Extract concepts from objective and summary
        concepts = _extract_concepts(f"{row['objective']} {row['summary']}")
        for concept in concepts:
            concept_node = f"concept_{concept}"
            graph.add_node(concept_node, label=concept.capitalize(), category='concept')
            graph.add_edge(exp_node, concept_node)

    return graph


def filter_graph_by_experiment(graph: KnowledgeGraph, experiment_id: str) -> KnowledgeGraph:
    """Extract a subgraph containing an experiment and its neighbours.

    Args:
        graph: The full KnowledgeGraph.
        experiment_id: ID of the experiment (e.g. 'PESTO').

    Returns:
        A KnowledgeGraph containing the experiment node and all nodes directly
        connected to it.
    """
    exp_node = f"exp_{experiment_id}"
    if exp_node not in graph.nodes:
        raise KeyError(f"Experiment {experiment_id} not found in graph")
    keep_nodes = [exp_node] + graph.neighbors(exp_node)
    return graph.subgraph(keep_nodes)


def _compute_positions(graph: KnowledgeGraph) -> Dict[NodeID, Tuple[float, float]]:
    """Compute deterministic node positions for plotting.

    Experiments are placed evenly around a circle. Each experiment's neighbours
    (organisms and concepts) are placed around it on a smaller circle. If a
    neighbour appears in multiple experiments, its position is determined by the
    first occurrence. This simple layout is adequate for small graphs and
    avoids external dependencies.

    Args:
        graph: The KnowledgeGraph for which positions should be computed.

    Returns:
        A mapping from node IDs to (x, y) coordinates.
    """
    # Identify experiment nodes
    exp_nodes = [n for n, attrs in graph.nodes.items() if attrs['category'] == 'experiment']
    n_exp = len(exp_nodes)
    positions: Dict[NodeID, Tuple[float, float]] = {}

    # Radius for placing experiment nodes
    R_exp = 3.0 if n_exp > 1 else 0.0

    # Precompute angles for experiments
    for idx, exp in enumerate(exp_nodes):
        angle = 2 * math.pi * idx / max(n_exp, 1)
        x = R_exp * math.cos(angle)
        y = R_exp * math.sin(angle)
        positions[exp] = (x, y)

        # Determine neighbours (organisms and concepts)
        neigh = graph.neighbors(exp)
        if not neigh:
            continue
        k = len(neigh)
        R_neigh = 1.0  # distance from experiment node
        for j, node in enumerate(neigh):
            # Skip if position already assigned (from another experiment)
            if node in positions:
                continue
            sub_angle = 2 * math.pi * j / k
            nx_pos = x + R_neigh * math.cos(sub_angle)
            ny_pos = y + R_neigh * math.sin(sub_angle)
            positions[node] = (nx_pos, ny_pos)

    # Ensure all nodes have a position; place any isolated nodes at the origin
    for node in graph.nodes:
        if node not in positions:
            positions[node] = (0.0, 0.0)

    return positions


def graph_to_plotly(graph: KnowledgeGraph) -> go.Figure:
    """Convert a KnowledgeGraph into a Plotly figure.

    Nodes are coloured by category to help differentiate experiments,
    organisms, and concepts. Edges are drawn as simple line segments.

    Args:
        graph: The KnowledgeGraph to visualise.

    Returns:
        A Plotly Figure object containing the graph.
    """
    positions = _compute_positions(graph)

    # Prepare edge traces
    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v in graph.edges:
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Prepare node traces
    node_x: List[float] = []
    node_y: List[float] = []
    node_text: List[str] = []
    node_color: List[str] = []
    colour_map = {
        'experiment': '#1f77b4',  # blue
        'organism': '#ff7f0e',    # orange
        'concept': '#2ca02c',     # green
    }
    for node_id, attrs in graph.nodes.items():
        x, y = positions[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(attrs['label'])
        node_color.append(colour_map.get(attrs['category'], '#17becf'))
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(
            showscale=False,
            color=node_color,
            size=14,
            line=dict(width=1, color='black'),
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ))
    fig.update_layout(
        template='simple_white',
        title_text='Space Biology Knowledge Graph',
        title_x=0.5
    )
    return fig