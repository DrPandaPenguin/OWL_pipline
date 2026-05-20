# dash builder UI
import base64
import json
import os
import re
import sys
import time
from datetime import datetime

from dash import Dash, html, dcc, callback, Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate
import dash_cytoscape as cyto
cyto.load_extra_layouts()  # enables dagre, cola, etc.

# path setup
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from extract_edges import extract_edges
from build_graph import build_graph as build_graph_src
from hallucination_checker import check_hallucinations


# outputs directory
_OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "outputs")


# edge colors
EDGE_COLORS = {
    "defines":    "#2196F3",   # blue
    "requires":   "#FF5722",   # deep orange
    "explains":   "#4CAF50",   # green
    "details":    "#9C27B0",   # purple
    "example_of": "#FF9800",   # amber
    "contrasts":  "#E91E63",   # pink
    "drives":     "#00BCD4",   # cyan
    "precedes":   "#607D8B",   # blue-grey
    "summarizes": "#795548",   # brown
    "cross_lecture": "#FF6F00",  # dark amber (course map)
}

# short display labels for edge types (cleaner on graph)
EDGE_DISPLAY = {
    "defines":       "defines",
    "requires":      "requires",
    "explains":      "explains",
    "details":       "details",
    "example_of":    "example",
    "contrasts":     "vs",
    "drives":        "drives",
    "precedes":      "then",
    "summarizes":    "sums up",
    "cross_lecture":  "shared",
}


def _build_stylesheet() -> list[dict]:
    base = [
        # --- Defult node (support) light fill, dark text ---
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": "#BBDEFB",
                "border-color": "#42A5F5",
                "border-width": "2px",
                "text-valign": "center",
                "text-halign": "center",
                "font-size": "9px",
                "font-weight": "600",
                "width": "65px",
                "height": "50px",
                "text-wrap": "wrap",
                "text-max-width": "140px",
                "padding": "6px",
                "color": "#111827",
            },
        },
        # backbone node (덜 진한 fill + 굵은 border)
        {
            "selector": "node.backbone",
            "style": {
                "background-color": "#90CAF9",
                "border-color": "#0D47A1",
                "border-width": "4px",
                "color": "#0D1B2A",
                "font-weight": "bold",
                "font-size": "10px",
                "width": "80px",
                "height": "58px",
                "text-max-width": "160px",
            },
        },
        # --- Compound nodes (topic groups) ---
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
        # --- Defult edge ---
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
        # --- Inferred edge dashed line ---
        {
            "selector": "edge.inferred-edge",
            "style": {
                "line-style": "dashed",
                "opacity": 0.65,
            },
        },
        # --- Selected state ---
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
    # per-edge-type colors via raw_type data field
    for edge_type, color in EDGE_COLORS.items():
        base.append({
            "selector": f"edge[raw_type='{edge_type}']",
            "style": {"line-color": color, "target-arrow-color": color},
        })
    return base


STYLESHEET = _build_stylesheet()

DAGRE_LAYOUT = {
    "name": "dagre",
    "rankDir": "TB",          # top-to-bottom = lecture order
    "spacingFactor": 1.2,
    "nodeSep": 30,            # horizontal spacing between nodes
    "rankSep": 60,            # vertical spacing between ranks (PART groups)
    "fit": True,
    "padding": 30,
}

# fallback COSE layout (for graphs without PART structure)
COSE_LAYOUT = {
    "name": "cose",
    "idealEdgeLength": 120,
    "nodeOverlap": 20,
    "fit": True,
    "padding": 40,
    "randomize": False,
    "componentSpacing": 120,
    "nodeRepulsion": 500000,
    "edgeElasticity": 80,
    "nestingFactor": 12,
    "gravity": 60,
    "numIter": 1500,
    "initialTemp": 200,
    "coolingFactor": 0.95,
    "minTemp": 1.0,
}


# KG normalization + Cytoscape conversion
# 파이프라인마다 schema가 살짝 달라서 viewer가 쓰기 좋게 통일하는 함수
def normalize_kg(raw: dict) -> dict:
    nodes = []
    for n in raw.get("nodes", []):
        nid = n.get("id") or n.get("node_id", "")
        nodes.append({
            "id": nid,
            "label": n.get("label", ""),
            "node_type": n.get("node_type", "concept"),
            "is_backbone": bool(n.get("is_backbone", False)),
            "source_sentence": n.get("source_sentence") or n.get("label", ""),
            "slide_anchor_id": n.get("slide_anchor_id", ""),
            "slide_anchor_title": n.get("slide_anchor_title", ""),
            # enrichment fields (may be empty if enrich_graph() not run)
            "description": n.get("description", ""),
            "why_it_matters": n.get("why_it_matters", ""),
            # ordering fields
            "lecture_order": n.get("lecture_order"),
            "section_order": n.get("section_order"),
        })
    edges = []
    for e in raw.get("edges", []):
        evidence = (e.get("evidence") or e.get("justification")
                    or e.get("justification_sentence")
                    or e.get("justification_span", ""))
        ed = {
            "source": e.get("source") or e.get("from") or e.get("source_node", ""),
            "target": e.get("target") or e.get("to") or e.get("target_node", ""),
            "relation": e.get("relation") or e.get("edge_type", ""),
            "evidence": evidence,
            "explanation": e.get("explanation") or e.get("reason", ""),
            "evidence_section": e.get("evidence_section", ""),
            "edge_source": e.get("edge_source", ""),  # "explicit" | "inferred"
            "edge_id": e.get("edge_id", ""),           # preserve for enrich_graph() lookup
        }
        if e.get("confidence_score") is not None:
            ed["confidence_score"] = e["confidence_score"]
        edges.append(ed)
    out = {"nodes": nodes, "edges": edges}
    if raw.get("topics"):
        out["topics"] = raw["topics"]
    return out


# KG dict -> cytoscape에서 그릴 수 있는 elements list로 변환 (compound 그룹 포함)
def to_cytoscape_elements(kg: dict) -> list:
    nodes = kg.get("nodes", [])
    edges = kg.get("edges", [])
    elements: list[dict] = []
    node_ids = {n["id"] for n in nodes}

    # --- PART compound nodes (hierarchy Level 1) ---
    # collect all unique slide_anchor_id + title pairs
    part_map = {}  # section_id → title
    for n in nodes:
        sid = n.get("slide_anchor_id", "")
        if sid and sid not in part_map:
            part_map[sid] = n.get("slide_anchor_title", sid)

    # sort by section_id for stable ordering
    for i, sid in enumerate(sorted(part_map.keys()), start=1):
        title = part_map[sid]
        # extreact part number from sid (e.g. "part_005" → 5)
        part_num = int(sid.split("_")[-1]) if sid.split("_")[-1].isdigit() else i
        elements.append({
            "data": {"id": f"compound_{sid}", "label": f"PART {part_num} — {title}"},
            "classes": "compound",
        })

    # bild lecture_order map for backbone numbering
    lecture_order_map = {}
    for n in nodes:
        lo = n.get("lecture_order")
        if lo is not None:
            lecture_order_map[n["id"]] = int(lo)

    # --- Bild global node numbering (by section_order then lecture_order) ---
    sorted_nodes = sorted(nodes, key=lambda n: (
        n.get("section_order") or 999,
        n.get("lecture_order") or 999,
    ))
    node_number_map = {}
    for idx, n in enumerate(sorted_nodes, start=1):
        node_number_map[n["id"]] = idx

    # --- Regular nodes (assigned to PART compound parente) ---
    for n in nodes:
        nid = n["id"]
        is_bb = n.get("is_backbone", False)
        base_label = n.get("label", nid)
        num = node_number_map.get(nid, "")
        display_label = f"[{num}] {base_label}"
        node_data = {
            "id": nid,
            "label": display_label,
            "node_type": n.get("node_type", "concept"),
            "is_backbone": str(is_bb).lower(),
            "source_sentence": n.get("source_sentence", ""),
            "slide_anchor_id": n.get("slide_anchor_id", ""),
            "slide_anchor_title": n.get("slide_anchor_title", ""),
            "description": n.get("description", ""),
            "why_it_matters": n.get("why_it_matters", ""),
            "lecture_order": n.get("lecture_order"),
            "section_order": n.get("section_order"),
        }
        # assign node to its PART compound parente
        sid = n.get("slide_anchor_id", "")
        if sid and f"compound_{sid}" in {el["data"]["id"] for el in elements}:
            node_data["parent"] = f"compound_{sid}"
        # CSS classes for backbone styling
        cls = "node backbone" if is_bb else "node support"
        elements.append({"data": node_data, "classes": cls})

    # edges
    for i, e in enumerate(edges):
        src, tgt = e.get("source", ""), e.get("target", "")
        if src not in node_ids or tgt not in node_ids:
            continue
        rel = e.get("relation", "")
        display_rel = EDGE_DISPLAY.get(rel, rel)
        edge_source = e.get("edge_source", "")
        is_inferred = edge_source == "inferred" or e.get("confidence_score") is not None
        edge_data = {
            "id": f"{src}_{tgt}_{rel}_{i}",
            "source": src,
            "target": tgt,
            "label": display_rel,
            "raw_type": rel,  # keep original for color matching
            "evidence": e.get("evidence") or e.get("justification", ""),
            "explanation": e.get("explanation") or e.get("reason", ""),
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


def filter_elements(elements: list, filter_kw) -> list:
    """Keep nodes matching keyword, their compound parents, and connected edges"""
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
    """Filter edges by source type: 'all', 'explicit', or 'inferred'"""
    if edge_mode == "all":
        return elements

    result = []
    for el in elements:
        classes = el.get("classes", "")
        # keep anything that's not an edge
        if "edge" not in classes:
            result.append(el)
            continue
        # check edge_source in data
        src = el.get("data", {}).get("edge_source", "")
        if edge_mode == "explicit" and src == "explicit":
            result.append(el)
        elif edge_mode == "inferred" and src == "inferred":
            result.append(el)
        elif edge_mode == "explicit" and src == "enriched":
            # enriched edges are based on explicit include them
            result.append(el)
    return result


# background pipeline
# output saving
def _save_outputs(name, raw_nodes, raw_edges, kus, kg, transcript=""):
    """outputs/<name>/ 에 결과 저장"""
    safe_name = re.sub(r"[^\w\-]", "_", name)[:60] or "transcript"
    out_dir = os.path.join(_OUTPUTS_DIR, safe_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump({"nodes": raw_nodes}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "edges.json"), "w", encoding="utf-8") as f:
        json.dump({"edges": raw_edges}, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "graph.json"), "w", encoding="utf-8") as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)
    if transcript:
        with open(os.path.join(out_dir, "transcript.txt"), "w", encoding="utf-8") as f:
            f.write(transcript)

    return out_dir


# 파이프라인 동기 실행 (UI는 멈췄다가 결과 한 번에 받음)
def build_pipeline_sync(transcript, name, pipeline_name, slide_text, model_name, api_key):
    """transcript -> KG 한 번에 실행. (kg, save_path) 반환."""
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    t0 = time.perf_counter()
    kus = []

    from src.pipelines import get_pipeline, enrich_graph, DEFAULT_CONFIG

    print(f"[{pipeline_name}] 실행 중...")
    run_config = DEFAULT_CONFIG.copy()
    run_config["ku_model"] = model_name
    run_config["node_model"] = model_name
    run_config["strict_model"] = model_name
    run_config["soft_model"] = model_name
    if pipeline_name == "slide_anchored" and slide_text:
        run_config["slide_text"] = slide_text
    run_config["enrich_graph"] = True
    result = get_pipeline(pipeline_name)(transcript, run_config)
    raw_nodes = result.get("nodes", [])
    raw_edges = result.get("edges", [])
    kus = result.get("kus", [])
    if not raw_nodes:
        return None, None, None

    # graph 조립
    kg = normalize_kg({"nodes": raw_nodes, "edges": raw_edges})

    # direct는 enrich를 별도 호출 (slide_anchored는 안에서 처리)
    if pipeline_name == "direct" and kg.get("nodes") and kg.get("edges"):
        try:
            enrich_result = enrich_graph(kg["nodes"], kg["edges"], transcript, {
                "node_model": model_name, "node_temperature": 0.2,
            })
            if enrich_result:
                kg["nodes"] = enrich_result.get("nodes", kg["nodes"])
                kg["edges"] = enrich_result.get("edges", kg["edges"])
        except Exception as e:
            print(f"[enrich] skip ({e})")

    save_name = name.strip() if name and name.strip() else datetime.now().strftime("transcript_%Y%m%d_%H%M%S")
    save_path = _save_outputs(save_name, raw_nodes, raw_edges, kus, kg, transcript=transcript)

    print(f"완료: {len(kg.get('nodes', []))} nodes, {len(kg.get('edges', []))} edges  ({time.perf_counter() - t0:.1f}s)")
    return kg, save_path


# inspector helpers
def get_node_by_id(kg: dict, node_id: str):
    return next((n for n in kg.get("nodes", []) if n["id"] == node_id), None)


def get_edges_connected_to_node(kg: dict, node_id: str) -> list:
    return [e for e in kg.get("edges", []) if e["source"] == node_id or e["target"] == node_id]


def _build_node_number_map(kg: dict) -> dict:
    """Build node_id → number mapping (same logic as to_cytoscape_elements)"""
    nodes = kg.get("nodes", [])
    sorted_nodes = sorted(nodes, key=lambda n: (
        n.get("section_order") or 999,
        n.get("lecture_order") or 999,
    ))
    return {n["id"]: idx for idx, n in enumerate(sorted_nodes, start=1)}


def node_label(kg: dict, node_id: str) -> str:
    n = get_node_by_id(kg, node_id)
    if not n:
        return node_id
    base = n.get("label", node_id)
    num_map = _build_node_number_map(kg)
    num = num_map.get(node_id, "")
    return f"[{num}] {base}" if num else base


def section_title(kg: dict, section_id: str) -> str:
    """Look up section label from slide_anchor_id → slide_anchor_title via any node in the kg"""
    if not section_id or not kg:
        return section_id
    for n in kg.get("nodes", []):
        if n.get("slide_anchor_id") == section_id:
            title = n.get("slide_anchor_title", "")
            if title:
                return title
    return section_id




def _edge_legend() -> html.Div:
    items = []
    for etype, color in EDGE_COLORS.items():
        display = EDGE_DISPLAY.get(etype, etype)
        items.append(html.Span([
            html.Span("━━", style={"color": color, "fontWeight": "bold", "marginRight": "2px"}),
            html.Span(display, style={"fontSize": "11px", "color": "#333", "marginRight": "10px"}),
        ]))
    items.append(html.Span([
        html.Span("╌╌", style={"color": "#888", "marginRight": "2px"}),
        html.Span("implied", style={"fontSize": "11px", "color": "#888"}),
    ]))
    return html.Div(items, style={
        "display": "flex", "flexWrap": "wrap", "alignItems": "center",
        "marginLeft": "12px", "gap": "2px",
    })


# dash app + layout
app = Dash(__name__, title="OWL Pipeline – Knowledge Graph")
app.config.suppress_callback_exceptions = True


# ?graph=<id>로 들어왔을 때 보여주는 read-only viewer (제출/공유용)


# 메인 레이아웃: 입력 패널 + 실행 버튼 + viewer 통합
def _build_full_layout() -> html.Div:
    """Full builder layout with pipeline controls"""
    return html.Div([

    # ---- Header: input + controls ----
    html.Div([
        html.H3("OWL Pipeline – Knowledge Graph", style={"margin": "0 0 10px 0", "fontSize": "18px"}),

        # pipeline selector + Model selector (same row)
        html.Div([
            html.Span("Pipeline: ", style={"fontSize": "12px", "marginRight": "6px", "color": "#555"}),
            dcc.Dropdown(
                id="pipeline-select",
                options=[
                    {"label": "Slide-Anchored (slides + transcript)", "value": "slide_anchored"},
                    {"label": "Direct (transcript only)", "value": "direct"},
                ],
                value="slide_anchored",
                clearable=False,
                style={"width": "340px", "fontSize": "12px"},
            ),
            html.Span("Model: ", style={"fontSize": "12px", "marginLeft": "16px", "marginRight": "6px", "color": "#555"}),
            dcc.Dropdown(
                id="model-select",
                options=[
                    {"label": "gpt-5.2", "value": "gpt-5.2"},
                    {"label": "gpt-4o", "value": "gpt-4o"},
                    {"label": "gpt-4o-mini", "value": "gpt-4o-mini"},
                    {"label": "gpt-4-turbo", "value": "gpt-4-turbo"},
                ],
                value="gpt-5.2",
                clearable=False,
                style={"width": "160px", "fontSize": "12px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),

        # API Key input (required when no server-side key is set)
        html.Div([
            html.Span("API Key: ", style={"fontSize": "12px", "marginRight": "6px", "color": "#555"}),
            dcc.Input(
                id="api-key-input",
                type="password",
                placeholder="sk-... (Enter your OpenAI API key)" if not os.getenv("OPENAI_API_KEY") else "Server key active — override optional",
                style={"padding": "5px 8px", "fontSize": "12px", "width": "380px",
                       "border": "1px solid #ccc", "borderRadius": "4px"},
            ),
            html.Span(
                "Server key set" if os.getenv("OPENAI_API_KEY") else "required to build graphs",
                style={"fontSize": "11px", "marginLeft": "8px",
                       "color": "#4CAF50" if os.getenv("OPENAI_API_KEY") else "#FF9800"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),

        # transcript name input + file upload
        html.Div([
            dcc.Input(
                id="transcript-name-input",
                type="text",
                placeholder="Transcript name (e.g. lecture_01)",
                style={"padding": "5px 8px", "fontSize": "12px", "width": "220px",
                       "border": "1px solid #ccc", "borderRadius": "4px"},
            ),
            html.Span("  ", style={"marginRight": "8px"}),
            dcc.Upload(
                id="upload-transcript",
                children=html.Div([
                    "Upload .txt",
                    html.Br(),
                    html.Span("or drag & drop", style={"fontSize": "10px", "color": "#888"}),
                ]),
                style={
                    "display": "inline-block", "padding": "4px 10px",
                    "border": "1px dashed #aaa", "borderRadius": "4px",
                    "cursor": "pointer", "fontSize": "12px", "textAlign": "center",
                    "backgroundColor": "#fafafa",
                },
                accept=".txt",
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px"}),

        dcc.Textarea(
            id="transcript-input",
            placeholder="Paste lecture transcript here…",
            style={
                "width": "100%", "height": "110px", "boxSizing": "border-box",
                "fontFamily": "monospace", "fontSize": "12px",
            },
        ),

        # slide text input shown only when slide_structure pipeline is selected
        html.Div(
            id="slide-text-container",
            children=[
                html.P(
                    "Slide text (for slide_structure pipeline)",
                    style={"fontSize": "12px", "color": "#555", "margin": "6px 0 2px 0"},
                ),
                html.P(
                    "Paste slide content here — e.g. markdown from pptx2md, or text copied from GPT.",
                    style={"fontSize": "11px", "color": "#888", "margin": "0 0 4px 0"},
                ),
                dcc.Textarea(
                    id="slide-text-input",
                    placeholder="## Slide 1: Introduction\n- Key point A\n- Key point B\n\n## Slide 2: …",
                    style={
                        "width": "100%", "height": "90px", "boxSizing": "border-box",
                        "fontFamily": "monospace", "fontSize": "12px",
                        "border": "1px solid #b0b0e0", "borderRadius": "4px",
                    },
                ),
            ],
            style={"display": "none", "marginTop": "6px"},
        ),

        html.Div([
            html.Button("Build Graph", id="build-btn", n_clicks=0,
                        style={"padding": "6px 14px", "marginRight": "10px", "cursor": "pointer"}),
            html.Button("Hallucination Check", id="halluc-btn", n_clicks=0,
                        style={"padding": "6px 14px", "marginRight": "10px", "cursor": "pointer",
                               "backgroundColor": "#FFF3E0", "border": "1px solid #FF9800"}),
            html.Span(id="build-status", style={"fontSize": "13px", "color": "#555"}),
        ], style={"marginTop": "8px", "display": "flex", "alignItems": "center"}),

        html.Div(id="save-path-info", style={"fontSize": "11px", "color": "#4CAF50", "marginTop": "4px"}),

        html.Div(id="phase-log"),
        html.Div(id="halluc-result", style={"marginTop": "6px"}),

        # load existing KG JSON (재실행 없이 그래프만 보고 싶을 때)
        html.Details([
            html.Summary("Load existing KG from JSON",
                         style={"cursor": "pointer", "fontSize": "12px", "color": "#888", "marginTop": "8px"}),
            dcc.Textarea(
                id="kg-json-input",
                placeholder='{"nodes": [...], "edges": [...]}',
                style={"width": "100%", "height": "70px", "boxSizing": "border-box",
                       "fontFamily": "monospace", "fontSize": "11px", "marginTop": "6px"},
            ),
            html.Button("Load KG", id="load-json-btn", n_clicks=0,
                        style={"marginTop": "4px", "padding": "4px 10px", "cursor": "pointer"}),
        ]),

    ], style={"padding": "12px 16px", "borderBottom": "1px solid #ddd", "backgroundColor": "#fafafa"}),

    # ---- Main: graph (left) + inspector (right) ----
    html.Div([

        # graph panel
        html.Div([
            html.Div([
                dcc.Input(
                    id="node-filter",
                    type="text",
                    placeholder="Filter nodes by keyword…",
                    debounce=True,
                    style={
                        "padding": "5px 8px", "fontSize": "12px",
                        "border": "1px solid #ccc", "borderRadius": "4px",
                        "width": "220px",
                    },
                ),
                # --- Edge visibility toggle (T3-3: explicit-only vs full) ---
                dcc.RadioItems(
                    id="edge-filter-toggle",
                    options=[
                        {"label": " All edges", "value": "all"},
                        {"label": " Explicit only", "value": "explicit"},
                        {"label": " Inferred only", "value": "inferred"},
                    ],
                    value="all",
                    inline=True,
                    style={"marginLeft": "16px", "fontSize": "11px"},
                    inputStyle={"marginRight": "3px"},
                    labelStyle={"marginRight": "10px"},
                ),
                _edge_legend(),
                # --- Compound collapse/expand + layout direction ---
                html.Div([
                    html.Button(
                        "Collapse Groups",
                        id="collapse-btn",
                        n_clicks=0,
                        style={"fontSize": "11px", "padding": "3px 8px",
                               "cursor": "pointer", "marginRight": "6px"},
                    ),
                    dcc.RadioItems(
                        id="layout-direction",
                        options=[
                            {"label": " Top→Bottom", "value": "TB"},
                            {"label": " Left→Right", "value": "LR"},
                        ],
                        value="TB",
                        inline=True,
                        style={"fontSize": "11px"},
                        inputStyle={"marginRight": "3px"},
                        labelStyle={"marginRight": "10px"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "marginLeft": "16px"}),
            ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "marginBottom": "6px"}),

            cyto.Cytoscape(
                id="cytoscape",
                elements=[],
                layout=DAGRE_LAYOUT,
                stylesheet=STYLESHEET,
                style={"width": "100%", "height": "540px", "border": "1px solid #eee", "borderRadius": "4px"},
                responsive=True,
            ),
        ], style={"flex": "3", "padding": "12px", "minWidth": 0}),

        # inspector panel
        html.Div([
            html.Div([
                html.H4("Inspector", style={"margin": "0", "display": "inline-block"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
            html.Div(
                id="inspector-content",
                children=html.P("Click a node or edge to see details.",
                                style={"color": "#888", "fontSize": "13px"}),
            ),
        ], style={
            "flex": "1", "minWidth": "320px", "maxWidth": "420px",
            "padding": "12px", "borderLeft": "1px solid #ddd",
            "backgroundColor": "#f9f9f9", "overflowY": "auto", "maxHeight": "600px",
        }),

    ], style={"display": "flex", "flexDirection": "row", "flexWrap": "wrap", "minHeight": "580px"}),

    # ---- Stores + interval ----
    dcc.Store(id="kg-store", data=None),
    dcc.Store(id="transcript-name", data=None),   # resolved name used for saving

    ], style={"fontFamily": "system-ui, -apple-system, sans-serif", "maxWidth": "1400px", "margin": "0 auto"})




app.layout = _build_full_layout()


# callbacks

@callback(
    Output("kg-store", "data"),
    Output("build-status", "children"),
    Output("save-path-info", "children"),
    Input("build-btn", "n_clicks"),
    Input("load-json-btn", "n_clicks"),
    State("transcript-input", "value"),
    State("transcript-name-input", "value"),
    State("pipeline-select", "value"),
    State("slide-text-input", "value"),
    State("model-select", "value"),
    State("api-key-input", "value"),
    State("kg-json-input", "value"),
    prevent_initial_call=True,
)
def main_state_callback(n_build, n_load, transcript, name_input, pipeline_name,
                         slide_text, model_name, user_api_key, kg_json_str):
    triggered = ctx.triggered_id

    # load existing KG branch
    if triggered == "load-json-btn":
        if not kg_json_str or not kg_json_str.strip():
            return no_update, "empty JSON", no_update
        try:
            raw = json.loads(kg_json_str)
            kg = normalize_kg(raw)
        except json.JSONDecodeError:
            return no_update, "invalid JSON", no_update
        if not kg.get("nodes"):
            return no_update, "no nodes in JSON", no_update
        n, e = len(kg["nodes"]), len(kg.get("edges", []))
        return kg, f"loaded {n} nodes, {e} edges", ""

    # build branch (동기 실행, 끝날 때까지 페이지 멈춤)
    if not transcript or not transcript.strip():
        return no_update, "empty input", no_update
    api_key = (user_api_key or "").strip() or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return no_update, "no API key — enter your OpenAI key above", no_update

    name = (name_input or "").strip() or datetime.now().strftime("transcript_%Y%m%d_%H%M%S")
    selected = pipeline_name or "slide_anchored"
    try:
        kg, save_path = build_pipeline_sync(
            transcript.strip(), name, selected,
            (slide_text or "").strip(), model_name or "gpt-5.2", api_key,
        )
    except Exception as e:
        return no_update, f"error: {e}", no_update

    if kg is None or not kg.get("nodes"):
        return None, "pipeline produced no graph", no_update

    n, e = len(kg.get("nodes", [])), len(kg.get("edges", []))
    rel_path = os.path.relpath(save_path, _PROJECT_ROOT) if save_path else "—"
    save_info = f"saved: {rel_path}/" if save_path else ""
    return kg, f"{n} nodes, {e} edges", save_info


@callback(
    Output("slide-text-container", "style"),
    Input("pipeline-select", "value"),
)
def toggle_slide_text_container(pipeline_name):
    if pipeline_name == "slide_anchored":
        return {"display": "block", "marginTop": "6px"}
    return {"display": "none", "marginTop": "6px"}


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

    # collapse: hide child nodes + edges, show only compound + backbone
    collapsed = (collapse_clicks or 0) % 2 == 1
    if collapsed:
        visible_ids = set()
        # keep compound nodes and backbone nodes
        for el in elements:
            d = el.get("data", {})
            cls = el.get("classes", "")
            if "compound" in cls or "backbone" in cls:
                visible_ids.add(d.get("id"))
        # filter: keep compound, backbone, edges between visible nodes
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
    return {**DAGRE_LAYOUT, "rankDir": direction or "TB", "fit": True}


@callback(
    Output("inspector-content", "children"),
    Input("cytoscape", "tapNodeData"),
    Input("cytoscape", "tapEdgeData"),
    State("kg-store", "data"),
)
def update_inspector(node_data, edge_data, kg):

    # determine which input actually fired tapNodeData and tapEdgeData
    # both retain stale values, so we must check ctx.triggered_id.
    triggered = ctx.triggered_id if ctx.triggered_id else None

    if triggered == "cytoscape" and ctx.triggered:
        # figure out which property triggered the callback
        prop_id = ctx.triggered[0]["prop_id"]  # e.g. "cytoscape.tapEdgeData"
        if "tapEdgeData" in prop_id and edge_data:
            return _render_edge_card(edge_data, kg)
        if "tapNodeData" in prop_id and node_data:
            return _render_node_card(node_data, kg)

    # fallback: if triggered by mode toggle, show whichever was last clicked
    if node_data and not edge_data:
        return _render_node_card(node_data, kg)
    if edge_data and not node_data:
        return _render_edge_card(edge_data, kg)

    return html.P("Click a node or edge to see details.",
                  style={"color": "#888", "fontSize": "13px"})


def _render_node_card(node_data: dict, kg) -> html.Div:
    is_debug = False
    """Render Node Card — 3-block layout: Identity → Semantic → Relations"""
    nid = node_data.get("id", "")
    label = node_data.get("label", "")
    node_type = node_data.get("node_type", "concept")
    is_backbone = node_data.get("is_backbone", "false") == "true"
    description = node_data.get("description", "")
    why_it_matters = node_data.get("why_it_matters", "")

    _divider = html.Hr(style={"margin": "10px 0", "border": "none", "borderTop": "1px solid #e0e0e0"})
    children: list = []

    # 1. identity (badges, label, section, parente)
    # extreact node number from label "[N] ..." if present
    import re as _re
    num_match = _re.match(r'\[(\d+)\]\s*(.*)', label)
    node_num = num_match.group(1) if num_match else ""
    clean_label = num_match.group(2) if num_match else label

    badges = []
    # node number badge big and visible
    if node_num:
        badges.append(html.Span(
            f"#{node_num}",
            style={"background": "#333", "color": "#fff",
                   "padding": "2px 10px", "borderRadius": "10px",
                   "fontSize": "13px", "fontWeight": "bold"},
        ))
    type_color = "#1565C0" if node_type == "concept" else "#2E7D32"
    type_bg = "#E3F2FD" if node_type == "concept" else "#E8F5E9"
    badges.append(html.Span(
        node_type,
        style={"background": type_bg, "color": type_color,
               "padding": "2px 8px", "borderRadius": "10px",
               "fontSize": "11px", "fontWeight": "bold",
               "marginLeft": "6px" if node_num else "0"},
    ))
    if is_backbone:
        badges.append(html.Span(
            "backbone",
            style={"background": "#FFF3E0", "color": "#E65100",
                   "padding": "2px 8px", "borderRadius": "10px",
                   "fontSize": "11px", "marginLeft": "6px"},
        ))
    children.append(html.Div(badges, style={"marginBottom": "6px"}))

    # label (without number prefix)
    children.append(html.H3(clean_label, style={"margin": "0 0 4px 0", "fontSize": "16px", "lineHeight": "1.3"}))

    # section (lecture part)
    sec_title = node_data.get("slide_anchor_title", "")
    sec_id = node_data.get("slide_anchor_id", "")
    if sec_title or sec_id:
        if is_debug:
            sec_text = f"{sec_id} · {sec_title}" if sec_title else sec_id
        else:
            sec_text = sec_title or sec_id
        children.append(html.P([
            # section indicator
            html.Span(sec_text, style={"color": "#666"}),
        ], style={"fontSize": "12px", "margin": "0 0 4px 0"}))

    # 2. semantic (description, why_it_matters)
    children.append(_divider)

    if description:
        children.append(html.P(description,
                               style={"fontSize": "13px", "color": "#333", "lineHeight": "1.5", "margin": "0 0 8px 0"}))
    else:
        children.append(html.P("No description available.",
                               style={"fontSize": "12px", "color": "#aaa", "fontStyle": "italic", "marginBottom": "8px"}))

    if why_it_matters:
        children.append(html.Div([
            html.P("Why It Matters",
                   style={"fontSize": "11px", "color": "#E65100", "margin": "0 0 2px 0", "fontWeight": "bold"}),
            html.P(why_it_matters, style={"fontSize": "13px", "color": "#555", "lineHeight": "1.4", "margin": "0"}),
        ], style={"marginBottom": "4px", "borderLeft": "3px solid #FFA726", "paddingLeft": "8px"}))

    # 3. relations (in/out edges)
    children.append(_divider)
    children.append(_render_node_connections(nid, kg))

    # debug panel
    if is_debug:
        children.append(html.Hr(style={"margin": "10px 0", "border": "none",
                                       "borderTop": "2px dashed #ccc"}))
        children.append(html.P("Debug Info",
                               style={"fontSize": "11px", "color": "#888", "fontWeight": "bold",
                                      "marginBottom": "4px", "textTransform": "uppercase", "letterSpacing": "0.5px"}))
        debug_fields = [
            ("ID", nid),
            ("source_sentence", node_data.get("source_sentence", "")),
            ("slide_anchor_id", node_data.get("slide_anchor_id", "")),
            ("is_backbone", str(is_backbone)),
        ]
        for fname, fval in debug_fields:
            if fval:
                children.append(html.P([
                    html.Span(f"{fname}: ", style={"fontWeight": "bold", "color": "#999"}),
                    html.Span(fval, style={"color": "#666"}),
                ], style={"fontSize": "11px", "margin": "2px 0", "wordBreak": "break-word"}))

    return html.Div(children)


def _render_node_connections(nid: str, kg) -> html.Div:
    """Render connected relations for a node — split into outgoing / incoming"""
    if not kg:
        return html.P("No graph loaded.", style={"fontSize": "11px", "color": "#888"})

    connected = get_edges_connected_to_node(kg, nid)
    if not connected:
        return html.P("No edges connected.", style={"fontSize": "11px", "color": "#888"})

    outgoing = []
    incoming = []
    for e in connected:
        src_id, tgt_id = e["source"], e["target"]
        rel = e.get("relation", "")
        color = EDGE_COLORS.get(rel, "#888")
        edge_source = e.get("edge_source", "")
        is_inf = edge_source == "inferred"
        conf = e.get("confidence_score")
        conf_str = f"  {float(conf):.2f}" if conf is not None and is_inf else ""
        style_tag = " (inferred)" if is_inf else ""

        item_style = {"marginBottom": "4px", "borderLeft": f"3px solid {color}", "paddingLeft": "8px"}

        if src_id == nid:
            lbl = node_label(kg, tgt_id)
            outgoing.append(html.Div([
                html.P([
                    html.Span("→ ", style={"color": color, "fontWeight": "bold"}),
                    html.Strong(lbl),
                    html.Span(f" [{rel}{conf_str}]{style_tag}", style={"color": color, "fontSize": "11px"}),
                ], style={"margin": "0"}),
            ], style=item_style))
        else:
            lbl = node_label(kg, src_id)
            incoming.append(html.Div([
                html.P([
                    html.Span("← ", style={"color": color, "fontWeight": "bold"}),
                    html.Strong(lbl),
                    html.Span(f" [{rel}{conf_str}]{style_tag}", style={"color": color, "fontSize": "11px"}),
                ], style={"margin": "0"}),
            ], style=item_style))

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
                    html.Summary(f"+ {len(hidden)} more", style={"fontSize": "11px", "color": "#888", "cursor": "pointer"}),
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


def _render_edge_card(edge_data: dict, kg) -> html.Div:
    is_debug = False
    """Render Edge Card — General or Debug mode"""
    rel = edge_data.get("label", "")
    color = EDGE_COLORS.get(rel, "#888")
    src_id = edge_data.get("source", "")
    tgt_id = edge_data.get("target", "")
    src_lbl = node_label(kg, src_id) if kg else src_id
    tgt_lbl = node_label(kg, tgt_id) if kg else tgt_id
    evidence = edge_data.get("evidence") or edge_data.get("justification", "")
    explanation = edge_data.get("explanation") or edge_data.get("reason", "")
    evidence_section = edge_data.get("evidence_section", "")
    edge_source = edge_data.get("edge_source", "")
    conf = edge_data.get("confidence_score")
    is_inferred = edge_source == "inferred" or conf is not None

    children: list = []

    # --- Edge type + line style ---
    source_label = "implied" if is_inferred else "grounded"
    line_style = "dashed" if is_inferred else "solid"
    children.append(html.Div([
        html.Span(rel, style={"color": color, "fontWeight": "bold", "fontSize": "15px"}),
        html.Span(f" ── {source_label} ({line_style})", style={"fontSize": "11px", "color": "#888", "marginLeft": "8px"}),
    ]))

    # --- From → To (labels) ---
    children.append(html.P([
        html.Strong(src_lbl),
        html.Span(" → ", style={"color": "#888"}),
        html.Strong(tgt_lbl),
    ], style={"fontSize": "13px", "margin": "4px 0 12px 0"}))

    # --- Justification (WHERE) ---
    if evidence:
        children.append(html.Div([
            html.P("Evidence",
                   style={"fontSize": "11px", "color": "#888", "margin": "0 0 2px 0", "fontWeight": "bold"}),
            html.P(f'"{evidence}"',
                   style={"fontSize": "12px", "fontStyle": "italic", "color": "#555", "margin": "0"}),
        ], style={"marginBottom": "10px", "borderLeft": f"3px solid {color}", "paddingLeft": "8px"}))

    # --- Reason (WHY) ---
    if explanation:
        children.append(html.Div([
            html.P("Explanation",
                   style={"fontSize": "11px", "color": "#888", "margin": "0 0 2px 0", "fontWeight": "bold"}),
            html.P(reason, style={"fontSize": "13px", "color": "#333", "lineHeight": "1.4", "margin": "0"}),
        ], style={"marginBottom": "10px"}))
    else:
        children.append(html.P("No explanation available.",
                               style={"fontSize": "12px", "color": "#aaa", "fontStyle": "italic", "marginBottom": "10px"}))

    # --- Section + Ocnfidnence ---
    if evidence_section:
        if is_debug:
            sec_display = f"{evidence_section} · {section_title(kg, evidence_section)}"
        else:
            sec_display = section_title(kg, evidence_section)
        children.append(html.P([
            # section indicator
            html.Strong("Section: "), sec_display,
        ], style={"fontSize": "12px", "marginBottom": "4px"}))

    # general: show ocnfidnence only on inferred edges; Debug: always show
    if conf is not None and (is_inferred or is_debug):
        children.append(html.P([
            html.Strong("Confidence: "), f"{float(conf):.2f}",
        ], style={"fontSize": "12px", "marginBottom": "4px"}))

    # --- Debug Mode: extra fields ---
    if is_debug:
        children.append(html.Hr(style={"margin": "8px 0", "borderColor": "#eee"}))
        children.append(html.P("Debug Info", style={"fontSize": "11px", "color": "#888", "fontWeight": "bold", "marginBottom": "4px"}))
        debug_fields = [
            ("edge_source", edge_source),
            ("from (ID)", src_id),
            ("to (ID)", tgt_id),
        ]
        for fname, fval in debug_fields:
            if fval:
                children.append(html.P([
                    html.Span(f"{fname}: ", style={"fontWeight": "bold", "color": "#888"}),
                    html.Span(fval, style={"color": "#666"}),
                ], style={"fontSize": "11px", "margin": "2px 0"}))

    return html.Div(children)



@callback(
    Output("transcript-input", "value"),
    Output("transcript-name-input", "value"),
    Input("upload-transcript", "contents"),
    State("upload-transcript", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    """Decode uploaded .txt file → populate textarea and name input"""
    if not contents:
        raise PreventUpdate
    try:
        _, content_string = contents.split(",", 1)
        text = base64.b64decode(content_string).decode("utf-8")
    except Exception:
        raise PreventUpdate
    name = os.path.splitext(filename)[0] if filename else "transcript"
    return text, name


@callback(
    Output("download-json", "data"),
    Input("download-btn", "n_clicks"),
    State("kg-store", "data"),
    State("transcript-name-input", "value"),
    prevent_initial_call=True,
)
def download_graph_json(n_clicks: int, kg, name_input):
    """Download the current graph as graph.json from the browser"""
    if not kg:
        raise PreventUpdate
    name = (name_input or "").strip() or datetime.now().strftime("transcript_%Y%m%d_%H%M%S")
    safe_name = re.sub(r"[^\w\-]", "_", name)[:60]
    filename = f"{safe_name}_graph.json"
    return dcc.send_string(json.dumps(kg, indent=2, ensure_ascii=False), filename)


# halusination Checker callback

@callback(
    Output("halluc-result", "children"),
    Input("halluc-btn", "n_clicks"),
    State("kg-store", "data"),
    State("transcript-input", "value"),
    prevent_initial_call=True,
)
def run_hallucination_check(n_clicks: int, kg, transcript):
    """Run Gemini-based hallucination check on the current KG + transcript"""
    if not kg or not transcript or not transcript.strip():
        return html.Div(
            "graph and transcript are both required for hallucination check.",
            style={"color": "#E65100", "fontSize": "12px", "padding": "6px",
                   "backgroundColor": "#FFF3E0", "borderRadius": "4px"},
        )

    try:
        result = check_hallucinations(kg, transcript)
    except ValueError as e:
        return html.Div(
            f"error: {e}",
            style={"color": "#C62828", "fontSize": "12px", "padding": "6px",
                   "backgroundColor": "#FFEBEE", "borderRadius": "4px"},
        )
    except Exception as e:
        return html.Div(
            f"hallucination check failed: {e}",
            style={"color": "#C62828", "fontSize": "12px", "padding": "6px",
                   "backgroundColor": "#FFEBEE", "borderRadius": "4px"},
        )

    issues = result.get("issues", [])
    n_nodes = result.get("total_nodes_checked", 0)
    n_edges = result.get("total_edges_checked", 0)
    summary = result.get("summary", "")

    severity_colors = {"high": "#C62828", "medium": "#E65100", "low": "#F9A825"}
    severity_bg = {"high": "#FFEBEE", "medium": "#FFF3E0", "low": "#FFFDE7"}

    # header
    if not issues:
        header_color, header_bg = "#2E7D32", "#E8F5E9"
        header_icon = "ok:"
    else:
        header_color, header_bg = "#E65100", "#FFF3E0"
        header_icon = "warn:"

    children = [
        html.Div([
            html.Span(f"{header_icon} Fact-check: ", style={"fontWeight": "bold"}),
            html.Span(f"{len(issues)} issue(s) found"),
            html.Span(f" — {n_nodes} nodes, {n_edges} edges checked (criterion: factual correctness via Google Search; absence from transcript is NOT flagged)",
                       style={"color": "#888", "marginLeft": "8px"}),
        ], style={"color": header_color, "backgroundColor": header_bg,
                  "padding": "8px 10px", "borderRadius": "4px", "fontSize": "13px"}),
    ]

    if summary:
        children.append(
            html.P(summary, style={"fontSize": "12px", "color": "#555",
                                   "margin": "6px 0 4px 0", "fontStyle": "italic"})
        )

    # issue list
    for issue in issues:
        sev = issue.get("severity", "low")
        body_lines = [
            html.Span(
                f"{issue.get('issue', '')}: {issue.get('explanation', '')}",
                style={"fontSize": "11px", "color": "#555"},
            )
        ]
        correction = issue.get("correction")
        if correction:
            body_lines.append(html.Br())
            body_lines.append(html.Span(
                f"Correction: {correction}",
                style={"fontSize": "11px", "color": "#2E7D32"},
            ))
        sources = issue.get("sources") or []
        if sources:
            body_lines.append(html.Br())
            body_lines.append(html.Span("Sources: ", style={"fontSize": "10px", "color": "#888"}))
            for i, src in enumerate(sources[:5]):
                if i:
                    body_lines.append(html.Span(", ", style={"fontSize": "10px", "color": "#888"}))
                body_lines.append(html.A(
                    src if len(src) < 60 else src[:60] + "…",
                    href=src, target="_blank",
                    style={"fontSize": "10px", "color": "#1565C0"},
                ))

        children.append(
            html.Div([
                html.Span(
                    f"[{sev.upper()}] ",
                    style={"fontWeight": "bold", "color": severity_colors.get(sev, "#555")},
                ),
                html.Span(
                    f"{issue.get('type', '?')} ",
                    style={"fontWeight": "bold"},
                ),
                html.Span(
                    f"{issue.get('id', '?')} — {issue.get('label', '?')}",
                    style={"fontWeight": "500"},
                ),
                html.Br(),
                *body_lines,
            ], style={
                "padding": "6px 8px", "margin": "3px 0",
                "backgroundColor": severity_bg.get(sev, "#f5f5f5"),
                "borderRadius": "3px", "borderLeft": f"3px solid {severity_colors.get(sev, '#ccc')}",
                "fontSize": "12px",
            })
        )

    return html.Details([
        html.Summary(
            f"{header_icon} Fact-check — {len(issues)} issue(s)",
            style={"cursor": "pointer", "fontSize": "12px", "fontWeight": "bold",
                   "color": header_color},
        ),
        html.Div(children, style={"marginTop": "4px"}),
    ], open=True, style={"marginTop": "4px"})


if __name__ == "__main__":
    # pORT env var is set by Hugging Face Spaces (7860) and Railway/Render.
    # falls back to 8050 for local development.
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("HF_SPACE") is None  # disable debug on HF
    app.run(debug=debug, host="0.0.0.0", port=port)
