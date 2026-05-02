"""Upload + match page.

The user picks whether they're uploading a CV/resume or a job posting, drops a
PDF in the upload zone, optionally tweaks the pooling metric and the number of
results to return, and submits. The PDF is parsed via BAML, the structured
output is embedded with the same chunking/pooling pipeline used for indexing,
and Qdrant is queried with kind + domain + pooling_method filters.

Matching is cross-kind by design:
- uploaded resume -> match against indexed job postings
- uploaded job posting -> match against indexed resumes
"""

from __future__ import annotations

import base64
import json
import traceback
from typing import Any

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update

from ..config import (
    DEFAULT_POOLING_METHOD,
    DEFAULT_TOP_K_RESULTS,
    MAX_TOP_K_RESULTS,
    POOLING_METHODS,
    POOLING_METHOD_LABELS,
)
from ..embeddings import embed_document_text
from ..extraction import (
    extract_job_posting,
    extract_resume,
    payload_to_text,
)
from ..search import search_similar_documents


KIND_RESUME = "resume"
KIND_JOB_POSTING = "job-posting"


def layout() -> Any:
    return dbc.Container(
        [
            dcc.Store(id="upload-kind-store", storage_type="memory"),
            html.Div(
                [
                    html.H1(
                        "Match CVs and job postings",
                        className="cjm-hero-title",
                    ),
                    html.P(
                        "Upload a resume or a job posting as PDF. We parse it with "
                        "your local LLM, embed it like the indexed corpus, and "
                        "surface the closest documents of the opposite type.",
                        className="cjm-hero-lead",
                    ),
                ],
                className="cjm-hero",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            [
                                html.Div(
                                    "Resume",
                                    className="fw-semibold fs-5 mb-1",
                                ),
                                html.Div(
                                    "Match to job postings",
                                    className="small text-muted",
                                ),
                            ],
                            id="kind-resume-btn",
                            color="primary",
                            outline=True,
                            className="w-100",
                        ),
                        md=6,
                        className="cjm-kind-card mb-3 mb-md-0",
                    ),
                    dbc.Col(
                        dbc.Button(
                            [
                                html.Div(
                                    "Job posting",
                                    className="fw-semibold fs-5 mb-1",
                                ),
                                html.Div(
                                    "Match to resumes",
                                    className="small text-muted",
                                ),
                            ],
                            id="kind-job-btn",
                            color="primary",
                            outline=True,
                            className="w-100",
                        ),
                        md=6,
                        className="cjm-kind-card",
                    ),
                ],
                className="mb-3 g-md-3",
            ),
            html.Div(id="kind-indicator", className="mb-3"),
            dcc.Upload(
                id="pdf-upload",
                children=html.Div(
                    [
                        "Drop a PDF here or ",
                        html.Span("browse", className="fw-semibold cjm-upload-link"),
                    ],
                    className="text-center",
                ),
                className="cjm-upload-zone",
                multiple=False,
                accept="application/pdf",
            ),
            html.Div(id="upload-filename", className="text-muted small mt-2"),
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label(
                                        "Pooling metric",
                                        className="fw-bold",
                                    ),
                                    dcc.Dropdown(
                                        id="pooling-method",
                                        options=[
                                            {
                                                "label": POOLING_METHOD_LABELS.get(
                                                    m, m
                                                ),
                                                "value": m,
                                            }
                                            for m in POOLING_METHODS
                                        ],
                                        value=DEFAULT_POOLING_METHOD,
                                        clearable=False,
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    html.Label(
                                        "Number of results",
                                        className="fw-bold",
                                    ),
                                    dcc.Slider(
                                        id="top-k",
                                        min=1,
                                        max=MAX_TOP_K_RESULTS,
                                        step=1,
                                        value=DEFAULT_TOP_K_RESULTS,
                                        marks={
                                            i: str(i)
                                            for i in range(
                                                1, MAX_TOP_K_RESULTS + 1
                                            )
                                        },
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="g-3 cjm-settings align-items-end",
                    ),
                ),
                className="mt-4 border-0 shadow-sm",
                style={
                    "borderRadius": "12px",
                    "border": "1px solid var(--cjm-border)",
                },
            ),
            html.Div(
                dbc.Button(
                    "Find similar documents",
                    id="search-btn",
                    color="success",
                    size="lg",
                    disabled=True,
                ),
                className="d-grid gap-2 mt-4 cjm-search-btn",
            ),
            dcc.Loading(
                id="search-loading",
                type="default",
                color="var(--cjm-accent)",
                children=html.Div(
                    id="search-output",
                    className="mt-4 cjm-results-wrap",
                ),
            ),
        ],
        fluid=True,
        className="px-0",
    )


def _kind_indicator(kind: str | None):
    if not kind:
        return dbc.Alert(
            [
                html.Strong("Step 1 · ", className="me-1"),
                "Select whether you are uploading a resume or a job posting.",
            ],
            color="light",
            className="cjm-step-hint mb-0",
        )
    label = "Resume" if kind == KIND_RESUME else "Job posting"
    return dbc.Alert(
        [
            html.Span(
                "Matching ",
                className="text-muted",
            ),
            html.Strong(label),
            html.Span(
                " → ",
                className="text-muted mx-1",
            ),
            html.Strong(
                "job postings" if kind == KIND_RESUME else "resumes"
            ),
            html.Span(
                " in the same inferred domain.",
                className="text-muted ms-1",
            ),
        ],
        color="primary",
        className="cjm-kind-selected mb-0",
    )


def _format_results(
    hits: list[dict],
    *,
    target_kind: str,
    domain: str,
    pooling: str,
) -> Any:
    if not hits:
        return dbc.Alert(
            (
                "No matching documents found in collection for kind="
                f"{target_kind!r}, domain={domain!r}, pooling={pooling!r}. "
                "Make sure the dataset has been indexed for this domain."
            ),
            color="warning",
        )

    rows = []
    for i, hit in enumerate(hits):
        rows.append(
            html.Tr(
                [
                    html.Td(str(i + 1)),
                    html.Td(f"{hit['score']:.4f}"),
                    html.Td(hit.get("document_key") or hit.get("document_id", "")),
                    html.Td(hit.get("domain", "")),
                    html.Td(hit.get("json_filename", "")),
                    html.Td(
                        dcc.Link(
                            "View",
                            href=f"/match/{i}",
                            className="btn btn-sm btn-outline-primary px-3",
                        )
                    ),
                ]
            )
        )

    table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("#"),
                        html.Th("Cosine score"),
                        html.Th("Document"),
                        html.Th("Domain"),
                        html.Th("Source file"),
                        html.Th(""),
                    ]
                )
            ),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
    )

    return html.Div(
        [
            dbc.Alert(
                [
                    html.Span("Results · ", className="text-muted"),
                    html.Strong(
                        "CVs" if target_kind == KIND_RESUME else "job postings"
                    ),
                    html.Span(" in domain ", className="text-muted"),
                    html.Strong(domain),
                    html.Span(" · metric ", className="text-muted"),
                    html.Code(pooling, className="ms-1"),
                ],
                color="success",
                className="cjm-results-banner border-0",
            ),
            html.Div(table, className="cjm-results-table"),
        ]
    )


def _decode_upload_contents(contents: str) -> bytes:
    """Decode the base64 ``data:`` URI returned by ``dcc.Upload``."""

    try:
        _, b64 = contents.split(",", 1)
    except ValueError as e:
        raise ValueError("uploaded file payload is malformed") from e
    return base64.b64decode(b64)


def register_callbacks(app: dash.Dash) -> None:
    @app.callback(
        Output("upload-kind-store", "data"),
        Output("kind-indicator", "children"),
        Output("kind-resume-btn", "outline"),
        Output("kind-job-btn", "outline"),
        Input("kind-resume-btn", "n_clicks"),
        Input("kind-job-btn", "n_clicks"),
        prevent_initial_call=False,
    )
    def select_kind(_n_resume, _n_job):
        triggered = dash.callback_context.triggered_id
        if triggered == "kind-resume-btn":
            kind = KIND_RESUME
        elif triggered == "kind-job-btn":
            kind = KIND_JOB_POSTING
        else:
            kind = None
        return (
            kind,
            _kind_indicator(kind),
            kind != KIND_RESUME,
            kind != KIND_JOB_POSTING,
        )

    @app.callback(
        Output("upload-filename", "children"),
        Output("search-btn", "disabled"),
        Input("pdf-upload", "filename"),
        Input("upload-kind-store", "data"),
    )
    def show_upload_filename(filename, kind):
        if not filename:
            return "", True
        ready = bool(kind)
        message = f"Selected file: {filename}"
        if not ready:
            message += " — choose Resume or Job Posting first."
        return message, not ready

    @app.callback(
        Output("search-output", "children"),
        Output("match-results-store", "data"),
        Output("query-doc-store", "data"),
        Input("search-btn", "n_clicks"),
        State("pdf-upload", "contents"),
        State("pdf-upload", "filename"),
        State("upload-kind-store", "data"),
        State("pooling-method", "value"),
        State("top-k", "value"),
        prevent_initial_call=True,
    )
    def run_search(
        _n_clicks,
        contents,
        filename,
        kind,
        pooling_method,
        top_k,
    ):
        if not contents or not kind:
            return (
                dbc.Alert(
                    "Pick a kind and upload a PDF first.",
                    color="warning",
                ),
                no_update,
                no_update,
            )

        try:
            pdf_bytes = _decode_upload_contents(contents)
        except Exception as e:
            return (
                dbc.Alert(f"Could not read the uploaded file: {e}", color="danger"),
                no_update,
                no_update,
            )

        try:
            if kind == KIND_RESUME:
                _model, payload, domain = extract_resume(pdf_bytes)
            else:
                _model, payload, domain = extract_job_posting(pdf_bytes)
        except Exception as e:
            traceback.print_exc()
            return (
                dbc.Alert(
                    [
                        html.P("Failed to extract structured content via BAML."),
                        html.Pre(str(e), className="small"),
                    ],
                    color="danger",
                ),
                no_update,
                no_update,
            )

        try:
            text_for_embedding = payload_to_text(payload)
            pooled_vectors = embed_document_text(text_for_embedding)
        except Exception as e:
            traceback.print_exc()
            return (
                dbc.Alert(
                    [
                        html.P("Embedding the uploaded document failed."),
                        html.Pre(str(e), className="small"),
                    ],
                    color="danger",
                ),
                no_update,
                no_update,
            )

        if pooling_method not in pooled_vectors:
            return (
                dbc.Alert(
                    f"Unknown pooling method {pooling_method!r}.",
                    color="danger",
                ),
                no_update,
                no_update,
            )

        target_kind = (
            KIND_JOB_POSTING if kind == KIND_RESUME else KIND_RESUME
        )

        try:
            hits = search_similar_documents(
                query_vector=pooled_vectors[pooling_method],
                kind=target_kind,
                domain=domain,
                pooling_method=pooling_method,
                top_k=int(top_k or 1),
            )
        except Exception as e:
            traceback.print_exc()
            return (
                dbc.Alert(
                    [
                        html.P("Qdrant search failed."),
                        html.Pre(str(e), className="small"),
                    ],
                    color="danger",
                ),
                no_update,
                no_update,
            )

        results = [
            {
                "score": h.score,
                "payload": h.payload,
                "point_id": h.point_id,
                "document_key": h.document_key,
                "document_id": h.document_id,
                "domain": h.domain,
                "json_filename": h.json_filename,
                "source_path": h.source_path,
            }
            for h in hits
        ]

        query_doc = {
            "uploaded_kind": kind,
            "target_kind": target_kind,
            "domain": domain,
            "pooling_method": pooling_method,
            "filename": filename,
            "payload": payload,
        }

        return (
            _format_results(
                results,
                target_kind=target_kind,
                domain=domain,
                pooling=pooling_method,
            ),
            results,
            query_doc,
        )