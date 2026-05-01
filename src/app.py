"""Entry point for the CV ↔ Job Posting matching Dash application.

Run from project root:

    uv run python -m src.app

Uses a manual multi-page routing approach: a single ``dcc.Location`` drives
which page layout is rendered into a shared ``page-content`` container, and a
``dcc.Store`` keeps the latest match results so the inspection page can render
without re-running expensive search/embedding work.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html

from .pages import document_detail, upload


app = dash.Dash(
    __name__,
    title="CV ↔ Job Match Maker",
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
)
server = app.server


def _navbar() -> dbc.NavbarSimple:
    return dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Match", href="/", active="exact")),
        ],
        brand="CV ↔ Job Match Maker",
        brand_href="/",
        color="primary",
        dark=True,
        className="mb-4",
    )


app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        # Stores the latest set of search results across pages.
        dcc.Store(id="match-results-store", storage_type="memory"),
        # Stores the most recent uploaded document's structured payload.
        dcc.Store(id="query-doc-store", storage_type="memory"),
        _navbar(),
        html.Div(id="page-content"),
    ],
    fluid=True,
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def _route(pathname: str | None):
    pathname = pathname or "/"

    if pathname == "/" or pathname == "":
        return upload.layout()

    if pathname.startswith("/match/"):
        try:
            idx = int(pathname.rsplit("/", 1)[-1])
        except ValueError:
            return _not_found(pathname)
        return document_detail.layout(idx)

    return _not_found(pathname)


def _not_found(pathname: str):
    return dbc.Alert(
        [
            html.H4("Page not found", className="alert-heading"),
            html.P(f"No view registered for path {pathname!r}."),
            dcc.Link("Go back to upload", href="/"),
        ],
        color="warning",
    )


# Register page-specific callbacks.
upload.register_callbacks(app)
document_detail.register_callbacks(app)


def main() -> None:
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
