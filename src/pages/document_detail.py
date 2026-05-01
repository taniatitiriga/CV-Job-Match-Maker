"""Detail view for a single matched document.

Renders the original processed JSON file from disk so the user can inspect the
full content of a matched CV / job posting. Match metadata (score, document
key, etc.) is read from the in-memory ``match-results-store``.
"""

from __future__ import annotations

from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html

from ..extraction import load_processed_json


def layout(index: int) -> Any:
    return dbc.Container(
        [
            dcc.Store(id="detail-index-store", data=index),
            html.Div(
                dcc.Link("← Back to results", href="/", className="btn btn-link"),
                className="mb-2",
            ),
            html.Div(id="detail-content"),
        ],
        fluid=True,
    )


def _render_resume(payload: dict) -> Any:
    title = payload.get("title", "Resume")
    skills = payload.get("skills") or []
    positions = payload.get("positions") or []

    return html.Div(
        [
            html.H3(title),
            html.H5("Skills", className="mt-3"),
            html.Div(
                [
                    dbc.Badge(s, color="secondary", className="me-1 mb-1")
                    for s in skills
                ]
                if skills
                else html.P("—", className="text-muted"),
            ),
            html.H5("Positions", className="mt-4"),
            html.Div(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    pos.get("job_title", "Position"),
                                    className="card-title",
                                ),
                                html.Div(
                                    [
                                        html.Strong(pos.get("company", "")),
                                        html.Span(
                                            f" · {pos.get('start_date', '')} – "
                                            f"{pos.get('end_date', '')}",
                                            className="text-muted",
                                        ),
                                    ],
                                    className="mb-2",
                                ),
                                html.Ul(
                                    [
                                        html.Li(d)
                                        for d in (pos.get("description") or [])
                                    ]
                                ),
                            ]
                        ),
                        className="mb-3",
                    )
                    for pos in positions
                ]
            ),
        ]
    )


def _render_job_posting(payload: dict) -> Any:
    return html.Div(
        [
            html.H3(payload.get("job_title", "Job Posting")),
            html.Div(
                [
                    html.Strong(payload.get("company", "")),
                    html.Span(
                        f" · domain {payload.get('domain', '')}",
                        className="text-muted",
                    ),
                ],
                className="mb-3",
            ),
            html.H5("Description"),
            html.P(payload.get("description", "—")),
            html.H5("Responsibilities", className="mt-3"),
            html.Ul(
                [html.Li(r) for r in (payload.get("responsibilities") or [])]
            ),
            html.H5("Required qualifications", className="mt-3"),
            html.Ul(
                [
                    html.Li(q)
                    for q in (payload.get("required_qualifications") or [])
                ]
            ),
        ]
    )


def _render_payload(kind: str, payload: dict) -> Any:
    if kind == "resume":
        return _render_resume(payload)
    if kind == "job-posting":
        return _render_job_posting(payload)
    # Unknown kind → show pretty JSON.
    import json as _json

    return html.Pre(_json.dumps(payload, indent=2, ensure_ascii=False))


def _match_summary(match: dict) -> Any:
    payload = match.get("payload") or {}
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(match.get("document_key", ""), className="mb-2"),
                html.Div(
                    [
                        html.Strong("Score: "),
                        html.Span(f"{match.get('score', 0):.4f}"),
                        html.Span("   "),
                        html.Strong("Kind: "),
                        html.Span(payload.get("kind", "")),
                        html.Span("   "),
                        html.Strong("Domain: "),
                        html.Span(payload.get("domain", "")),
                        html.Span("   "),
                        html.Strong("Pooling: "),
                        html.Code(payload.get("pooling_method", "")),
                    ],
                    className="text-muted small mb-1",
                ),
                html.Div(
                    [
                        html.Strong("Source file: "),
                        html.Code(match.get("source_path", "")),
                    ],
                    className="text-muted small",
                ),
            ]
        ),
        className="mb-3",
    )


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("detail-content", "children"),
        Input("detail-index-store", "data"),
        State("match-results-store", "data"),
    )
    def render(idx, results):
        if results is None:
            return dbc.Alert(
                [
                    html.P(
                        "No match results in memory. Run a search first to "
                        "view document details."
                    ),
                    dcc.Link("Go to upload", href="/"),
                ],
                color="warning",
            )

        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return dbc.Alert("Invalid match index.", color="danger")

        if idx < 0 or idx >= len(results):
            return dbc.Alert(
                f"Match #{idx} is out of range (have {len(results)}).",
                color="warning",
            )

        match = results[idx]
        source_path = match.get("source_path") or ""

        try:
            doc = load_processed_json(source_path)
        except Exception as e:
            return dbc.Alert(
                [
                    html.P(
                        f"Could not load the processed JSON at "
                        f"{source_path!r}."
                    ),
                    html.Pre(str(e), className="small"),
                ],
                color="danger",
            )

        kind = (match.get("payload") or {}).get("kind") or match.get("payload", {}).get(
            "kind"
        ) or ""
        return html.Div(
            [
                _match_summary(match),
                _render_payload(kind, doc),
            ]
        )
