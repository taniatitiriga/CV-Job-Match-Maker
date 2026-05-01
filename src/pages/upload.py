"""Upload + match page.

The user picks whether they're uploading a CV/resume or a job posting, drops a
PDF in the upload zone, optionally tweaks the pooling metric and the number of
results to return, and submits. The PDF is parsed via BAML, the structured
output is embedded with the same chunking/pooling pipeline used for indexing,
and Qdrant is queried with kind + domain + pooling_method filters.
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
            html.H2("Find similar documents", className="mb-3"),
            html.P(
                "Upload a CV/resume or a job posting (PDF). "
                "We will parse it, infer its domain via the local LLM, "
                "embed it with the same pipeline used for the indexed dataset, "
                "and show the most similar documents of the same kind.",
                className="text-muted",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            [
                                html.Div("Upload Resume", className="fw-bold fs-5"),
                                html.Div(
                                    "Match against indexed CVs",
                                    className="small",
                                ),
                            ],
                            id="kind-resume-btn",
                            color="primary",
                            outline=True,
                            className="w-100 py-4",
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Button(
                            [
                                html.Div(
                                    "Upload Job Posting", className="fw-bold fs-5"
                                ),
                                html.Div(
                                    "Match against indexed job postings",
                                    className="small",
                                ),
                            ],
                            id="kind-job-btn",
                            color="primary",
                            outline=True,
                            className="w-100 py-4",
                        ),
                        md=6,
                    ),
                ],
                className="mb-4",
            ),
            html.Div(id="kind-indicator", className="mb-3"),
            dcc.Upload(
                id="pdf-upload",
                children=html.Div(
                    [
                        "Drag and drop or ",
                        html.A("select a PDF", className="fw-bold"),
                    ]
                ),
                style={
                    "width": "100%",
                    "minHeight": "120px",
                    "lineHeight": "120px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "8px",
                    "textAlign": "center",
                    "background": "#f8f9fa",
                },
                multiple=False,
                accept="application/pdf",
            ),
            html.Div(id="upload-filename", className="text-muted small mt-2"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Pooling metric", className="fw-bold"),
                            dcc.Dropdown(
                                id="pooling-method",
                                options=[
                                    {
                                        "label": POOLING_METHOD_LABELS.get(m, m),
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
                                "Number of results", className="fw-bold"
                            ),
                            dcc.Slider(
                                id="top-k",
                                min=1,
                                max=MAX_TOP_K_RESULTS,
                                step=1,
                                value=DEFAULT_TOP_K_RESULTS,
                                marks={
                                    i: str(i)
                                    for i in range(1, MAX_TOP_K_RESULTS + 1)
                                },
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="mt-4",
            ),
            html.Div(
                dbc.Button(
                    "Find similar",
                    id="search-btn",
                    color="success",
                    size="lg",
                    disabled=True,
                ),
                className="d-grid gap-2 mt-4",
            ),
            dcc.Loading(
                id="search-loading",
                type="default",
                children=html.Div(id="search-output", className="mt-4"),
            ),
        ],
        fluid=True,
    )


def _kind_indicator(kind: str | None):
    if not kind:
        return dbc.Alert(
            "Choose what type of document you want to upload.",
            color="secondary",
        )
    label = "Resume" if kind == KIND_RESUME else "Job Posting"
    return dbc.Alert(
        [
            html.Span("Selected kind: "),
            html.Strong(label),
        ],
        color="info",
    )


def _format_results(
    hits: list[dict],
    *,
    kind: str,
    domain: str,
    pooling: str,
) -> Any:
    if not hits:
        return dbc.Alert(
            (
                "No matching documents found in collection for kind="
                f"{kind!r}, domain={domain!r}, pooling={pooling!r}. "
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
                            "Inspect",
                            href=f"/match/{i}",
                            className="btn btn-sm btn-outline-primary",
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
                    html.Span("Compared against "),
                    html.Strong(
                        "CVs" if kind == KIND_RESUME else "job postings"
                    ),
                    html.Span(" in domain "),
                    html.Strong(domain),
                    html.Span(" using metric "),
                    html.Code(pooling),
                    html.Span("."),
                ],
                color="success",
            ),
            table,
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

        try:
            hits = search_similar_documents(
                query_vector=pooled_vectors[pooling_method],
                kind=kind,
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
            "kind": kind,
            "domain": domain,
            "pooling_method": pooling_method,
            "filename": filename,
            "payload": payload,
        }

        return (
            _format_results(
                results,
                kind=kind,
                domain=domain,
                pooling=pooling_method,
            ),
            results,
            query_doc,
        )
