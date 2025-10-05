"""FastAPI application for the Space Biology Knowledge Engine.

This web service exposes a simple dashboard for exploring NASA space‑biology
experiments. It provides HTML pages rendered with Jinja2 templates and
serves an interactive knowledge graph using Plotly. FastAPI is used as the
framework because Flask is not available in the current environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.io as pio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Use a relative import for the knowledge_graph module. When this file is
# executed within the ``space_biology_app`` package (for example via
# ``uvicorn space_biology_app.app:app``), relative imports ensure that
# Python correctly resolves the module from the package directory.
from .knowledge_graph import build_graph, filter_graph_by_experiment, graph_to_plotly


def load_data(data_path: str) -> Dict[str, Dict]:
    """Load experiment metadata from a CSV and build a lookup dict.

    Args:
        data_path: Path to the CSV file containing experiment records.

    Returns:
        A dictionary mapping experiment IDs to row dictionaries.
    """
    df = pd.read_csv(data_path)
    return {row['id']: row for _, row in df.iterrows()}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Space Biology Knowledge Engine")

    base_path = Path(__file__).resolve().parent
    data_path = base_path / 'data' / 'experiments.csv'
    experiments = load_data(str(data_path))
    full_graph = build_graph(pd.DataFrame(experiments.values()))

    # Mount static files
    static_path = base_path / 'static'
    app.mount('/static', StaticFiles(directory=str(static_path)), name='static')
    # Setup template directory
    templates = Jinja2Templates(directory=str(base_path / 'templates'))

    @app.get('/', response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Homepage showing a list of all experiments."""
        exp_list = sorted(experiments.values(), key=lambda r: r['name'])
        return templates.TemplateResponse('index.html', {
            'request': request,
            'experiments': exp_list
        })

    @app.get('/experiment/{exp_id}', response_class=HTMLResponse)
    async def experiment_page(request: Request, exp_id: str) -> HTMLResponse:
        """Show details for a specific experiment."""
        exp = experiments.get(exp_id)
        if exp is None:
            raise HTTPException(status_code=404, detail='Experiment not found')
        return templates.TemplateResponse('experiment.html', {
            'request': request,
            'experiment': exp
        })

    @app.get('/graph', response_class=HTMLResponse)
    async def graph_page(request: Request, id: Optional[str] = None) -> HTMLResponse:
        """Render the knowledge graph or a subgraph for a given experiment."""
        if id:
            try:
                subgraph = filter_graph_by_experiment(full_graph, id)
            except KeyError:
                raise HTTPException(status_code=404, detail='Experiment not found')
            fig = graph_to_plotly(subgraph)
            title = f"Knowledge graph for experiment {id}"
        else:
            fig = graph_to_plotly(full_graph)
            title = "Knowledge graph of space‑biology experiments"
        graph_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
        return templates.TemplateResponse('graph.html', {
            'request': request,
            'graph_html': graph_html,
            'title': title
        })

    return app


app = create_app()