"""
OWL Pipeline — Knowledge Graph Viewer (read-only)

Lightweight viewer for sharing pre-generated knowledge graphs with professors.
No pipeline execution, no API keys required.

Usage:
  python viewer.py                         # shows graph list
  Open: http://localhost:8050/?graph=demo_lecture

Graphs are loaded from the graphs/ directory.
Place graph JSON files as: graphs/{name}.json
"""

import json
import os
import re
from typing import Any, Optional

from dash import Dash, html, dcc, callback, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_cytoscape as cyto
cyto.load_extra_layouts()

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_GRAPHS_DIR = os.path.join(_PROJECT_ROOT, "graphs")


# ---------------------------------------------------------------------------
# edge colors
# ---------------------------------------------------------------------------
EDGE_COLORS: dict[str, str] = {
    "defines":    "#2196F3",
    "requires":   "#FF5722",
    "explains":   "#4CAF50",
    "elaborates": "#9C27B0",
    "example_of": "#FF9800",
    "contrasts":  "#E91E63",
    "motivates":  "#00BCD4",
    "precedes":   "#607D8B",
    "summarizes": "#795548",
}


# ---------------------------------------------------------------------------
# stylesheet
# ---------------------------------------------------------------------------
def _build_stylesheet() -> list[dict]:
    base = [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": "#BBDEFB",
                "border-color": "#42A5F5",
                "border-width": "2px",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "10px",
                "font-weight": "600",
                "width": "60px",
                "height": "46px",
                "text-wrap": "wrap",
                "text-max-width": "120px",
                "padding": "4px",
                "color": "#111827",
            },
        },
        {
            "selector": "node.backbone",
            "style": {
                "background-color": "#90CAF9",
                "border-color": "#0D47A1",
                "border-width": "4px",
                "color": "#0D1B2A",
                "font-weight": "bold",
                "width": "70px",
                "height": "54px",
            },
        },
        {
            "selector": "node.compound",
            "style": {
                "label": "data(label)",
                "background-color": "#EBF5FF",
                "border-color": "#97C2FC",
                "border-width": "1px",
                "font-size": "11px",
                "text-valign": "top",
                "text-halign": "center",
                "color": "#444",
            },
        },
        {
            "selector": "edge",
            "style": {
                "label": "data(label)",
                "curve-style": "bezier",
                "target-arrow-shape": "vee",
                "line-color": "#aaa",
                "target-arrow-color": "#aaa",
                "font-size": "8px",
                "text-rotation": "autorotate",
                "text-margin-y": -8,
                "opacity": 0.9,
            },
        },
        {
            "selector": "edge.inferred-edge",
            "style": {
                "line-style": "dashed",
                "opacity": 0.65,
            },
        },
        {
            "selector": ":selected",
            "style": {
                "background-color": "#F7A7A6",
                "border-width": 2,
                "border-color": "#E91E63",
                "line-color": "#E91E63",
                "target-arrow-color": "#E91E63",
            },
        },
    ]
    for edge_type, color in EDGE_COLORS.items():
        base.append({
            "selector": f"edge[label='{edge_type}']",
            "style": {"line-color": color, "target-arrow-color": color},
        })
    return base


STYLESHEET = _build_stylesheet()

DAGRE_LAYOUT = {
    "name": "dagre",
    "rankDir": "TB",
    "spacingFactor": 1.2,
    "nodeSep": 30,
    "rankSep": 60,
    "fit": True,
    "padding": 30,
}


# ---------------------------------------------------------------------------
# kG normalization + Cytoscape conversion
# ---------------------------------------------------------------------------
def normalize_kg(raw: dict) -> dict:
    nodes = []
    for n in raw.get("nodes", []):
        nid = n.get("id") or n.get("node_id", "")
        nodes.append({
            "id": nid,
            "label": n.get("label", ""),
            "node_type": n.get("node_type", "concept"),
            "is_backbone": bool(n.get("is_backbone", False)),
            "parent_id": n.get("parent_id") or "",
            "source_sentence": n.get("source_sentence") or n.get("label", ""),
            "slide_anchor_id": n.get("slide_anchor_id", ""),
            "slide_anchor_title": n.get("slide_anchor_title", ""),
            "description": n.get("description", ""),
            "why_it_matters": n.get("why_it_matters", ""),
            "lecture_order": n.get("lecture_order"),
            "section_order": n.get("section_order"),
        })
    edges = []
    for e in raw.get("edges", []):
        justification = (e.get("justification")
                         or e.get("evidence")
                         or e.get("justification_sentence")
                         or e.get("justification_span", ""))
        ed: dict[str, Any] = {
            "source": e.get("source") or e.get("from") or e.get("source_node", ""),
            "target": e.get("target") or e.get("to") or e.get("target_node", ""),
            "relation": e.get("relation") or e.get("edge_type", ""),
            "justification": justification,
            "reason": e.get("reason", ""),
            "evidence_section": e.get("evidence_section", ""),
            "edge_source": e.get("edge_source", ""),
            "edge_id": e.get("edge_id", ""),
        }
        if e.get("confidence_score") is not None:
            ed["confidence_score"] = e["confidence_score"]
        edges.append(ed)
    return {"nodes": nodes, "edges": edges}


def to_cytoscape_elements(kg: dict) -> list:
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    elements: list[dict] = []
    node_ids = {n["id"] for n in nodes}

    # compound nodes (PART groups)
    part_map: dict[str, str] = {}
    for n in nodes:
        sid = n.get("slide_anchor_id", "")
        if sid and sid not in part_map:
            part_map[sid] = n.get("slide_anchor_title", sid)

    for i, sid in enumerate(sorted(part_map.keys()), start=1):
        title = part_map[sid]
        part_num = int(sid.split("_")[-1]) if sid.split("_")[-1].isdigit() else i
        elements.append({
            "data": {"id": f"compound_{sid}", "label": f"PART {part_num} \u2014 {title}"},
            "classes": "compound",
        })

    # lecture order map for backbone numbering
    lecture_order_map: dict[str, int] = {}
    for n in nodes:
        lo = n.get("lecture_order")
        if lo is not None:
            lecture_order_map[n["id"]] = int(lo)

    # regular nodes
    for n in nodes:
        nid = n["id"]
        is_bb = n.get("is_backbone", False)
        base_label = n.get("label", nid)
        if is_bb and nid in lecture_order_map:
            display_label = f"{lecture_order_map[nid]}. {base_label}"
        else:
            display_label = base_label
        node_data: dict[str, Any] = {
            "id": nid,
            "label": display_label,
            "node_type": n.get("node_type", "concept"),
            "is_backbone": str(is_bb).lower(),
            "parent_id": n.get("parent_id", ""),
            "source_sentence": n.get("source_sentence", ""),
            "slide_anchor_id": n.get("slide_anchor_id", ""),
            "slide_anchor_title": n.get("slide_anchor_title", ""),
            "description": n.get("description", ""),
            "why_it_matters": n.get("why_it_matters", ""),
            "lecture_order": n.get("lecture_order"),
            "section_order": n.get("section_order"),
        }
        sid = n.get("slide_anchor_id", "")
        if sid and f"compound_{sid}" in {el["data"]["id"] for el in elements}:
            node_data["parent"] = f"compound_{sid}"
        cls = "node backbone" if is_bb else "node support"
        elements.append({"data": node_data, "classes": cls})

    # edges
    for i, e in enumerate(edges):
        src, tgt = e.get("source", ""), e.get("target", "")
        if src not in node_ids or tgt not in node_ids:
            continue
        rel = e.get("relation", "")
        edge_source = e.get("edge_source", "")
        is_inferred = edge_source == "inferred" or e.get("confidence_score") is not None
        edge_data: dict[str, Any] = {
            "id": f"{src}_{tgt}_{rel}_{i}",
            "source": src,
            "target": tgt,
            "label": rel,
            "justification": e.get("justification", ""),
            "reason": e.get("reason", ""),
            "evidence_section": e.get("evidence_section", ""),
            "edge_source": edge_source,
        }
        if e.get("confidence_score") is not None:
            edge_data["confidence_score"] = e["confidence_score"]
        elements.append({
            "data": edge_data,
            "classes": "edge inferred-edge" if is_inferred else "edge grounded-edge",
        })
    return elements


def filter_elements(elements: list, filter_kw: Optional[str]) -> list:
    if not filter_kw or not filter_kw.strip():
        return elements
    kw = filter_kw.lower().strip()
    compounds = [el for el in elements if el.get("classes") == "compound"]
    nodes = [el for el in elements if (el.get("classes") or "").startswith("node")]
    edges = [el for el in elements if "edge" in (el.get("classes") or "")]
    matching_ids = {n["data"]["id"] for n in nodes if kw in n["data"].get("label", "").lower()}
    if not matching_ids:
        return []
    parent_ids = {
        n["data"]["parent"] for n in nodes
        if n["data"]["id"] in matching_ids and "parent" in n["data"]
    }
    result: list[dict] = []
    result.extend(el for el in compounds if el["data"]["id"] in parent_ids)
    result.extend(el for el in nodes if el["data"]["id"] in matching_ids)
    result.extend(
        el for el in edges
        if el["data"]["source"] in matching_ids or el["data"]["target"] in matching_ids
    )
    return result


def _filter_edges_by_source(elements: list, edge_mode: str) -> list:
    if edge_mode == "all":
        return elements
    result = []
    for el in elements:
        classes = el.get("classes", "")
        if "edge" not in classes:
            result.append(el)
            continue
        src = el.get("data", {}).get("edge_source", "")
        if edge_mode == "explicit" and src in ("explicit", "enriched"):
            result.append(el)
        elif edge_mode == "inferred" and src == "inferred":
            result.append(el)
    return result


# ---------------------------------------------------------------------------
# kG helpers
# ---------------------------------------------------------------------------
def get_node_by_id(kg: dict, node_id: str) -> Optional[dict]:
    return next((n for n in kg.get("nodes", []) if n["id"] == node_id), None)


def get_edges_connected_to_node(kg: dict, node_id: str) -> list:
    return [e for e in kg.get("edges", []) if e["source"] == node_id or e["target"] == node_id]


def node_label(kg: dict, node_id: str) -> str:
    n = get_node_by_id(kg, node_id)
    return n.get("label", node_id) if n else node_id


def section_title(kg: dict, section_id: str) -> str:
    if not section_id or not kg:
        return section_id
    for n in kg.get("nodes", []):
        if n.get("slide_anchor_id") == section_id:
            title = n.get("slide_anchor_title", "")
            if title:
                return title
    return section_id


# ---------------------------------------------------------------------------
# graph loading
# ---------------------------------------------------------------------------
def _load_graph(graph_id: str) -> Optional[dict]:
    safe_id = re.sub(r"[^\w\-]", "_", graph_id)
    path = os.path.join(_GRAPHS_DIR, f"{safe_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _list_graphs() -> list[str]:
    if not os.path.isdir(_GRAPHS_DIR):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_GRAPHS_DIR)
        if f.endswith(".json")
    )


# ---------------------------------------------------------------------------
# edge legend
# ---------------------------------------------------------------------------
def _edge_legend() -> html.Div:
    items = []
    for etype, color in EDGE_COLORS.items():
        items.append(html.Span([
            html.Span("\u2501\u2501", style={"color": color, "fontWeight": "bold", "marginRight": "2px"}),
            html.Span(etype, style={"fontSize": "11px", "color": "#333", "marginRight": "10px"}),
        ]))
    items.append(html.Span([
        html.Span("\u254c\u254c", style={"color": "#888", "marginRight": "2px"}),
        html.Span("inferred", style={"fontSize": "11px", "color": "#888"}),
    ]))
    return html.Div(items, style={
        "display": "flex", "flexWrap": "wrap", "alignItems": "center",
        "marginLeft": "12px", "gap": "2px",
    })


# ---------------------------------------------------------------------------
# dash app
# ---------------------------------------------------------------------------
app = Dash(__name__, title="OWL Pipeline \u2013 Knowledge Graph Viewer")


def _graph_list_page() -> html.Div:
    """Landing page showing available graphs."""
    graphs = _list_graphs()
    if not graphs:
        return html.Div([
            html.H3("OWL Pipeline \u2013 Knowledge Graph Viewer",
                     style={"padding": "40px 40px 10px"}),
            html.P("No graphs available. Add graph JSON files to the graphs/ directory.",
                   style={"padding": "0 40px", "color": "#888"}),
        ])

    graph_links = []
    for g in graphs:
        kg = _load_graph(g)
        n_nodes = len(kg.get("nodes", [])) if kg else "?"
        n_edges = len(kg.get("edges", [])) if kg else "?"
        graph_links.append(
            html.A(
                html.Div([
                    html.Span(g.replace("_", " ").title(),
                              style={"fontSize": "16px", "fontWeight": "600", "color": "#1565C0"}),
                    html.Span(f"  {n_nodes} nodes, {n_edges} edges",
                              style={"fontSize": "12px", "color": "#888", "marginLeft": "12px"}),
                ], style={
                    "padding": "16px 20px", "border": "1px solid #e0e0e0",
                    "borderRadius": "8px", "marginBottom": "8px",
                    "backgroundColor": "#fafafa", "cursor": "pointer",
                }),
                href=f"/?graph={g}",
                style={"textDecoration": "none"},
            )
        )

    return html.Div([
        html.H3("OWL Pipeline \u2013 Knowledge Graph Viewer",
                 style={"padding": "30px 40px 10px 40px", "margin": "0"}),
        html.P("Select a lecture to explore its knowledge graph.",
               style={"padding": "0 40px 20px", "color": "#666", "margin": "0"}),
        html.Div(graph_links, style={"padding": "0 40px", "maxWidth": "600px"}),
    ])


def _viewer_page(graph_id: str) -> html.Div:
    """Graph viewer page."""
    kg = _load_graph(graph_id)
    if not kg:
        available = ", ".join(_list_graphs()) or "(none)"
        return html.Div([
            html.H3("Graph not found", style={"color": "#C62828", "padding": "40px"}),
            html.P(f'No graph found for "{graph_id}".',
                   style={"padding": "0 40px", "color": "#555"}),
            html.P(f"Available: {available}",
                   style={"padding": "0 40px", "color": "#888", "fontSize": "13px"}),
            html.A("Back to graph list", href="/",
                   style={"padding": "0 40px", "color": "#1565C0"}),
        ])

    kg = normalize_kg(kg)
    n_nodes = len(kg.get("nodes", []))
    n_edges = len(kg.get("edges", []))

    return html.Div([
        # header
        html.Div([
            html.A("\u2190 All graphs", href="/",
                   style={"fontSize": "12px", "color": "#888", "textDecoration": "none",
                          "marginRight": "16px"}),
            html.H3(graph_id.replace("_", " ").title(),
                     style={"margin": "0", "fontSize": "18px", "display": "inline-block"}),
            html.Span(f"  {n_nodes} nodes, {n_edges} edges",
                       style={"fontSize": "12px", "color": "#888", "marginLeft": "12px"}),
        ], style={"padding": "12px 16px", "borderBottom": "1px solid #ddd",
                  "backgroundColor": "#fafafa", "display": "flex", "alignItems": "center"}),

        # main: graph + inspector
        html.Div([
            # graph panel
            html.Div([
                html.Div([
                    dcc.Input(
                        id="node-filter", type="text",
                        placeholder="Filter nodes by keyword\u2026", debounce=True,
                        style={"padding": "5px 8px", "fontSize": "12px",
                               "border": "1px solid #ccc", "borderRadius": "4px", "width": "220px"},
                    ),
                    dcc.RadioItems(
                        id="edge-filter-toggle",
                        options=[
                            {"label": " All edges", "value": "all"},
                            {"label": " Explicit only", "value": "explicit"},
                            {"label": " Inferred only", "value": "inferred"},
                        ],
                        value="all", inline=True,
                        style={"marginLeft": "16px", "fontSize": "11px"},
                        inputStyle={"marginRight": "3px"}, labelStyle={"marginRight": "10px"},
                    ),
                    _edge_legend(),
                    html.Div([
                        html.Button("Collapse Groups", id="collapse-btn", n_clicks=0,
                                    style={"fontSize": "11px", "padding": "3px 8px",
                                           "cursor": "pointer", "marginRight": "6px"}),
                        dcc.RadioItems(
                            id="layout-direction",
                            options=[
                                {"label": " Top\u2192Bottom", "value": "TB"},
                                {"label": " Left\u2192Right", "value": "LR"},
                            ],
                            value="TB", inline=True, style={"fontSize": "11px"},
                            inputStyle={"marginRight": "3px"}, labelStyle={"marginRight": "10px"},
                        ),
                    ], style={"display": "flex", "alignItems": "center", "marginLeft": "16px"}),
                ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "marginBottom": "6px"}),

                cyto.Cytoscape(
                    id="cytoscape",
                    elements=to_cytoscape_elements(kg),
                    layout=DAGRE_LAYOUT, stylesheet=STYLESHEET,
                    style={"width": "100%", "height": "600px",
                           "border": "1px solid #eee", "borderRadius": "4px"},
                    responsive=True,
                ),
            ], style={"flex": "3", "padding": "12px", "minWidth": 0}),

            # inspector panel
            html.Div([
                html.H4("\U0001f50d Inspector",
                         style={"margin": "0 0 8px 0", "fontSize": "16px"}),
                html.Div(
                    id="inspector-content",
                    children=html.P("Click a node or edge to see details.",
                                    style={"color": "#888", "fontSize": "13px"}),
                ),
            ], style={
                "flex": "1", "minWidth": "320px", "maxWidth": "420px",
                "padding": "12px", "borderLeft": "1px solid #ddd",
                "backgroundColor": "#f9f9f9", "overflowY": "auto", "maxHeight": "650px",
            }),
        ], style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "minHeight": "620px"}),

        # stores
        dcc.Store(id="kg-store", data=kg),
    ], style={"fontFamily": "system-ui, -apple-system, sans-serif",
              "maxWidth": "1400px", "margin": "0 auto"})


def serve_layout():
    from flask import has_request_context, request
    if not has_request_context():
        # dash validates layout at import time — return a placeholder
        return html.Div("Loading...")
    graph_id = request.args.get("graph")
    if graph_id:
        return _viewer_page(graph_id)
    return _graph_list_page()


app.layout = serve_layout
app.config.suppress_callback_exceptions = True


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("cytoscape", "elements"),
    Input("kg-store", "data"),
    Input("node-filter", "value"),
    Input("edge-filter-toggle", "value"),
    Input("collapse-btn", "n_clicks"),
)
def update_elements(kg, filter_kw, edge_mode, collapse_clicks):
    if not kg:
        return []
    elements = to_cytoscape_elements(kg)
    elements = filter_elements(elements, filter_kw)
    elements = _filter_edges_by_source(elements, edge_mode or "all")

    collapsed = (collapse_clicks or 0) % 2 == 1
    if collapsed:
        visible_ids = set()
        for el in elements:
            d = el.get("data", {})
            cls = el.get("classes", "")
            if "compound" in cls or "backbone" in cls:
                visible_ids.add(d.get("id"))
        filtered = []
        for el in elements:
            d = el.get("data", {})
            cls = el.get("classes", "")
            if "edge" in cls:
                if d.get("source") in visible_ids and d.get("target") in visible_ids:
                    filtered.append(el)
            elif d.get("id") in visible_ids:
                filtered.append(el)
        elements = filtered
    return elements


@callback(
    Output("collapse-btn", "children"),
    Input("collapse-btn", "n_clicks"),
)
def update_collapse_label(n_clicks):
    collapsed = (n_clicks or 0) % 2 == 1
    return "Expand Groups" if collapsed else "Collapse Groups"


@callback(
    Output("cytoscape", "layout"),
    Input("layout-direction", "value"),
)
def update_layout(direction):
    return {**DAGRE_LAYOUT, "rankDir": direction or "TB"}


@callback(
    Output("inspector-content", "children"),
    Input("cytoscape", "tapNodeData"),
    Input("cytoscape", "tapEdgeData"),
    State("kg-store", "data"),
)
def update_inspector(node_data, edge_data, kg):
    triggered = ctx.triggered_id if ctx.triggered_id else None

    if triggered == "cytoscape" and ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"]
        if "tapEdgeData" in prop_id and edge_data:
            return _render_edge_card(edge_data, kg)
        if "tapNodeData" in prop_id and node_data:
            return _render_node_card(node_data, kg)

    if node_data and not edge_data:
        return _render_node_card(node_data, kg)
    if edge_data and not node_data:
        return _render_edge_card(edge_data, kg)

    return html.P("Click a node or edge to see details.",
                  style={"color": "#888", "fontSize": "13px"})


# ---------------------------------------------------------------------------
# inspector rendering
# ---------------------------------------------------------------------------

def _render_node_card(node_data: dict, kg: Optional[dict]) -> html.Div:
    nid = node_data.get("id", "")
    label = node_data.get("label", "")
    node_type = node_data.get("node_type", "concept")
    is_backbone = node_data.get("is_backbone", "false") == "true"
    parent_id = node_data.get("parent_id", "")
    description = node_data.get("description", "")
    why_it_matters = node_data.get("why_it_matters", "")

    _divider = html.Hr(style={"margin": "10px 0", "border": "none", "borderTop": "1px solid #e0e0e0"})
    children: list = []

    # badges
    badges = []
    type_color = "#1565C0" if node_type == "concept" else "#2E7D32"
    type_bg = "#E3F2FD" if node_type == "concept" else "#E8F5E9"
    badges.append(html.Span(
        node_type,
        style={"background": type_bg, "color": type_color,
               "padding": "2px 8px", "borderRadius": "10px",
               "fontSize": "11px", "fontWeight": "bold"},
    ))
    if is_backbone:
        badges.append(html.Span(
            "\u2b50 backbone",
            style={"background": "#FFF3E0", "color": "#E65100",
                   "padding": "2px 8px", "borderRadius": "10px",
                   "fontSize": "11px", "marginLeft": "6px"},
        ))
    children.append(html.Div(badges, style={"marginBottom": "6px"}))

    # label
    children.append(html.H3(label, style={"margin": "0 0 4px 0", "fontSize": "16px", "lineHeight": "1.3"}))

    # section
    sec_title = node_data.get("slide_anchor_title", "")
    sec_id = node_data.get("slide_anchor_id", "")
    if sec_title or sec_id:
        children.append(html.P([
            html.Span("\U0001f4cd ", style={"fontSize": "11px"}),
            html.Span(sec_title or sec_id, style={"color": "#666"}),
        ], style={"fontSize": "12px", "margin": "0 0 4px 0"}))

    # parent
    if parent_id:
        parent_lbl = node_label(kg, parent_id) if kg else parent_id
        children.append(html.P([
            html.Span("\u2191 Parent: ", style={"fontWeight": "bold", "color": "#888", "fontSize": "11px"}),
            html.Span(parent_lbl, style={"color": "#1565C0", "fontSize": "12px"}),
        ], style={"margin": "2px 0 0 0"}))

    # description
    children.append(_divider)
    if description:
        children.append(html.P(description,
                               style={"fontSize": "13px", "color": "#333", "lineHeight": "1.5", "margin": "0 0 8px 0"}))
    else:
        children.append(html.P("No description available.",
                               style={"fontSize": "12px", "color": "#aaa", "fontStyle": "italic", "marginBottom": "8px"}))

    # why it matters
    if why_it_matters:
        children.append(html.Div([
            html.P("\U0001f4a1 Why It Matters",
                   style={"fontSize": "11px", "color": "#E65100", "margin": "0 0 2px 0", "fontWeight": "bold"}),
            html.P(why_it_matters, style={"fontSize": "13px", "color": "#555", "lineHeight": "1.4", "margin": "0"}),
        ], style={"marginBottom": "4px", "borderLeft": "3px solid #FFA726", "paddingLeft": "8px"}))

    # relations
    children.append(_divider)
    children.append(_render_node_connections(nid, kg))

    return html.Div(children)


def _render_node_connections(nid: str, kg: Optional[dict]) -> html.Div:
    if not kg:
        return html.P("No graph loaded.", style={"fontSize": "11px", "color": "#888"})

    connected = get_edges_connected_to_node(kg, nid)
    if not connected:
        return html.P("No edges connected.", style={"fontSize": "11px", "color": "#888"})

    outgoing, incoming = [], []
    for e in connected:
        src_id, tgt_id = e["source"], e["target"]
        rel = e.get("relation", "")
        color = EDGE_COLORS.get(rel, "#888")
        is_inf = e.get("edge_source") == "inferred"
        conf = e.get("confidence_score")
        conf_str = f"  {float(conf):.2f}" if conf is not None and is_inf else ""
        style_tag = " \u22ef" if is_inf else ""
        item_style = {"marginBottom": "4px", "borderLeft": f"3px solid {color}", "paddingLeft": "8px"}

        if src_id == nid:
            lbl = node_label(kg, tgt_id)
            outgoing.append(html.Div([html.P([
                html.Span("\u2192 ", style={"color": color, "fontWeight": "bold"}),
                html.Strong(lbl),
                html.Span(f" [{rel}{conf_str}]{style_tag}", style={"color": color, "fontSize": "11px"}),
            ], style={"margin": "0"})], style=item_style))
        else:
            lbl = node_label(kg, src_id)
            incoming.append(html.Div([html.P([
                html.Span("\u2190 ", style={"color": color, "fontWeight": "bold"}),
                html.Strong(lbl),
                html.Span(f" [{rel}{conf_str}]{style_tag}", style={"color": color, "fontSize": "11px"}),
            ], style={"margin": "0"})], style=item_style))

    FOLD_LIMIT = 5
    sections: list = []

    def _make_section(title: str, items: list) -> None:
        if not items:
            return
        visible = items[:FOLD_LIMIT]
        hidden = items[FOLD_LIMIT:]
        section_children: list = [
            html.P(html.Strong(title), style={"margin": "0 0 4px 0", "fontSize": "12px", "color": "#555"}),
            html.Div(visible),
        ]
        if hidden:
            section_children.append(
                html.Details([
                    html.Summary(f"+ {len(hidden)} more",
                                 style={"fontSize": "11px", "color": "#888", "cursor": "pointer"}),
                    html.Div(hidden),
                ], style={"marginTop": "2px"})
            )
        sections.append(html.Div(section_children, style={"marginBottom": "8px"}))

    _make_section(f"Outgoing ({len(outgoing)})", outgoing)
    _make_section(f"Incoming ({len(incoming)})", incoming)

    return html.Div([
        html.P(html.Strong("Connected Relations"), style={"margin": "0 0 6px 0", "fontSize": "13px"}),
        html.Div(sections) if sections else html.P("No edges.", style={"fontSize": "11px", "color": "#888"}),
    ])


def _render_edge_card(edge_data: dict, kg: Optional[dict]) -> html.Div:
    rel = edge_data.get("label", "")
    color = EDGE_COLORS.get(rel, "#888")
    src_id = edge_data.get("source", "")
    tgt_id = edge_data.get("target", "")
    src_lbl = node_label(kg, src_id) if kg else src_id
    tgt_lbl = node_label(kg, tgt_id) if kg else tgt_id
    justification = edge_data.get("justification", "")
    reason = edge_data.get("reason", "")
    evidence_section = edge_data.get("evidence_section", "")
    edge_source = edge_data.get("edge_source", "")
    conf = edge_data.get("confidence_score")
    is_inferred = edge_source == "inferred" or conf is not None

    children: list = []

    # edge type
    line_label = "dashed" if is_inferred else "solid"
    children.append(html.Div([
        html.Span(rel, style={"color": color, "fontWeight": "bold", "fontSize": "15px"}),
        html.Span(f" \u2500\u2500 {line_label}", style={"fontSize": "11px", "color": "#888", "marginLeft": "8px"}),
    ]))

    # from → To
    children.append(html.P([
        html.Strong(src_lbl),
        html.Span(" \u2192 ", style={"color": "#888"}),
        html.Strong(tgt_lbl),
    ], style={"fontSize": "13px", "margin": "4px 0 12px 0"}))

    # justification
    if justification:
        children.append(html.Div([
            html.P("\U0001f4cc Justification",
                   style={"fontSize": "11px", "color": "#888", "margin": "0 0 2px 0", "fontWeight": "bold"}),
            html.P(f'"{justification}"',
                   style={"fontSize": "12px", "fontStyle": "italic", "color": "#555", "margin": "0"}),
        ], style={"marginBottom": "10px", "borderLeft": f"3px solid {color}", "paddingLeft": "8px"}))

    # reason
    if reason:
        children.append(html.Div([
            html.P("\U0001f4ac Reason",
                   style={"fontSize": "11px", "color": "#888", "margin": "0 0 2px 0", "fontWeight": "bold"}),
            html.P(reason, style={"fontSize": "13px", "color": "#333", "lineHeight": "1.4", "margin": "0"}),
        ], style={"marginBottom": "10px"}))
    else:
        children.append(html.P("No explanation available.",
                               style={"fontSize": "12px", "color": "#aaa", "fontStyle": "italic", "marginBottom": "10px"}))

    # section
    if evidence_section:
        sec_display = section_title(kg, evidence_section)
        children.append(html.P([
            html.Span("\U0001f4cd ", style={"fontSize": "11px"}),
            html.Strong("Section: "), sec_display,
        ], style={"fontSize": "12px", "marginBottom": "4px"}))

    # confidence
    if conf is not None and is_inferred:
        children.append(html.P([
            html.Strong("Confidence: "), f"{float(conf):.2f}",
        ], style={"fontSize": "12px", "marginBottom": "4px"}))

    return html.Div(children)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("HF_SPACE") is None
    app.run(debug=debug, host="0.0.0.0", port=port)
