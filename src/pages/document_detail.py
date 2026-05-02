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
                dcc.Link(
                    "Back to match",
                    href="/",
                    className="btn btn-outline-secondary btn-sm px-3",
                ),
                className="cjm-back-row",
            ),
            html.Div(id="detail-content"),
        ],
        fluid=True,
        className="px-0",
    )


def _render_resume(payload: dict) -> Any:
    title = payload.get("title", "Resume")
    skills = payload.get("skills") or []
    positions = payload.get("positions") or []

    return html.Div(
        [
            html.Div(
                [
                    html.H3(title, className="mb-0"),
                    html.Div(
                        "Resume",
                        className="badge bg-primary bg-opacity-10 text-primary mt-2",
                        style={"fontWeight": "600", "fontSize": "0.75rem"},
                    ),
                ],
                className="d-flex flex-wrap align-items-center gap-2 mb-1",
            ),
            html.H5("Skills"),
            html.Div(
                [
                    dbc.Badge(
                        s,
                        color="light",
                        className="me-1 mb-1 cjm-badge-skill border text-dark",
                    )
                    for s in skills
                ]
                if skills
                else html.P("—", className="text-muted mb-0"),
            ),
            html.H5("Positions"),
            html.Div(
                [
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6(
                                    pos.get("job_title", "Position"),
                                    className="card-title fw-semibold mb-2",
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
                                    className="mb-2 small",
                                ),
                                html.Ul(
                                    [
                                        html.Li(d)
                                        for d in (pos.get("description") or [])
                                    ],
                                    className="mb-0 ps-3",
                                ),
                            ]
                        ),
                        className="mb-3",
                    )
                    for pos in positions
                ]
            ),
        ],
        className="cjm-doc-shell",
    )


def _render_job_posting(payload: dict) -> Any:
    return html.Div(
        [
            html.Div(
                [
                    html.H3(
                        payload.get("job_title", "Job Posting"),
                        className="mb-0",
                    ),
                    html.Div(
                        "Job posting",
                        className="badge bg-primary bg-opacity-10 text-primary mt-2",
                        style={"fontWeight": "600", "fontSize": "0.75rem"},
                    ),
                ],
                className="d-flex flex-wrap align-items-center gap-2 mb-3",
            ),
            html.Div(
                [
                    html.Strong(payload.get("company", "")),
                    html.Span(
                        f" · {payload.get('domain', '')}",
                        className="text-muted",
                    ),
                ],
                className="small text-body mb-0 pb-3",
                style={"borderBottom": "1px solid var(--cjm-border)"},
            ),
            html.H5("Description"),
            html.P(
                payload.get("description", "—"),
                className="mb-0",
                style={"lineHeight": "1.65"},
            ),
            html.H5("Responsibilities"),
            html.Ul(
                [html.Li(r) for r in (payload.get("responsibilities") or [])],
                className="mb-0",
            ),
            html.H5("Required qualifications"),
            html.Ul(
                [
                    html.Li(q)
                    for q in (payload.get("required_qualifications") or [])
                ],
                className="mb-0",
            ),
        ],
        className="cjm-doc-shell",
    )


def _render_payload(kind: str, payload: dict) -> Any:
    if kind == "resume":
        return _render_resume(payload)
    if kind == "job-posting":
        return _render_job_posting(payload)
    # Unknown kind → show pretty JSON.
    import json as _json

    return html.Div(
        html.Pre(
            _json.dumps(payload, indent=2, ensure_ascii=False),
            className="mb-0 small",
            style={"whiteSpace": "pre-wrap"},
        ),
        className="cjm-doc-shell",
    )


def _match_summary(match: dict) -> Any:
    payload = match.get("payload") or {}
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.H5(
                            match.get("document_key", ""),
                            className="mb-1 fw-semibold",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    f"{match.get('score', 0):.4f}",
                                    className="fw-semibold text-primary",
                                ),
                                html.Span(
                                    " similarity",
                                    className="text-muted small",
                                ),
                            ],
                            className="mb-3",
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Div(
                                    "Kind",
                                    className="small text-muted text-uppercase fw-semibold",
                                    style={"fontSize": "0.7rem", "letterSpacing": "0.04em"},
                                ),
                                html.Div(
                                    payload.get("kind", "—"),
                                    className="small",
                                ),
                            ],
                            md=3,
                            sm=6,
                            className="mb-3",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    "Domain",
                                    className="small text-muted text-uppercase fw-semibold",
                                    style={"fontSize": "0.7rem", "letterSpacing": "0.04em"},
                                ),
                                html.Div(
                                    payload.get("domain", "—"),
                                    className="small",
                                ),
                            ],
                            md=3,
                            sm=6,
                            className="mb-3",
                        ),
                        dbc.Col(
                            [
                                html.Div(
                                    "Pooling",
                                    className="small text-muted text-uppercase fw-semibold",
                                    style={"fontSize": "0.7rem", "letterSpacing": "0.04em"},
                                ),
                                html.Div(
                                    html.Code(
                                        payload.get("pooling_method", ""),
                                        className="small",
                                    ),
                                    className="small",
                                ),
                            ],
                            md=3,
                            sm=6,
                            className="mb-3",
                        ),
                    ],
                    className="g-0",
                ),
                html.Div(
                    [
                        html.Div(
                            "Source file",
                            className="small text-muted text-uppercase fw-semibold",
                            style={"fontSize": "0.7rem", "letterSpacing": "0.04em"},
                        ),
                        html.Code(
                            match.get("source_path", ""),
                            className="small text-break d-block mt-1",
                        ),
                    ],
                    className="pt-2",
                    style={"borderTop": "1px solid var(--cjm-border)"},
                ),
            ]
        ),
        className="mb-3 cjm-meta-card",
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
            ],
            className="cjm-detail-body",
        )
