# networkx + matplotlib로 KG json을 정적 그림으로 그림 (CLI용, dash_app과는 별개)
import json
import sys

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: install networkx, matplotlib first.", file=sys.stderr)


EDGE_COLORS = {
    'defines':    '#9B59B6',
    'requires':   '#FF9500',
    'explains':   '#3498DB',
    'details':    '#2ECC71',
    'example_of': '#E74C3C',
    'contrasts':  '#F39C12',
    'drives':     '#1ABC9C',
    'unknown':    '#95A5A6',
}

EDGE_TYPES = ['defines', 'requires', 'explains', 'details',
              'example_of', 'contrasts', 'drives', 'unknown']


def load_graph(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading graph: {e}", file=sys.stderr)
        sys.exit(1)


def create_networkx_graph(data):
    G = nx.DiGraph()

    for n in data.get('nodes', []):
        nid = n.get('node_id', '')
        G.add_node(nid, label=n.get('label', nid), order=n.get('order', 0))

    for e in data.get('edges', []):
        s = e.get('source_node', '')
        t = e.get('target_node', '')
        if not s or not t:
            continue
        if s not in G:
            G.add_node(s, label=s)
        if t not in G:
            G.add_node(t, label=t)
        G.add_edge(s, t,
                   edge_type=e.get('edge_type', 'unknown'),
                   confidence=e.get('confidence', 'strict'),
                   justification=e.get('justification_sentence') or e.get('justification_span', ''))

    return G


def visualize_graph(graph_file, output_file=None, layout='spring'):
    if not VISUALIZATION_AVAILABLE:
        print("Error: visualization libs not available", file=sys.stderr)
        sys.exit(1)

    data = load_graph(graph_file)
    G = create_networkx_graph(data)
    if len(G.nodes()) == 0:
        print("Error: empty graph", file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(16, 12))

    if layout == 'hierarchical':
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except Exception:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color='#4A90E2', node_size=2000, alpha=0.9)

    for et in EDGE_TYPES:
        strict_edges = [(u, v) for u, v, d in G.edges(data=True)
                        if d.get('edge_type', 'unknown') == et and d.get('confidence', 'strict') == 'strict']
        if strict_edges:
            nx.draw_networkx_edges(G, pos, edgelist=strict_edges,
                                   edge_color=EDGE_COLORS.get(et, '#95A5A6'),
                                   width=2.5, alpha=0.7, arrows=True,
                                   arrowsize=25, arrowstyle='->', style='solid')

    for et in EDGE_TYPES:
        soft_edges = [(u, v) for u, v, d in G.edges(data=True)
                      if d.get('edge_type', 'unknown') == et and d.get('confidence', 'strict') == 'soft']
        if soft_edges:
            nx.draw_networkx_edges(G, pos, edgelist=soft_edges,
                                   edge_color=EDGE_COLORS.get(et, '#95A5A6'),
                                   width=1.5, alpha=0.4, arrows=True,
                                   arrowsize=20, arrowstyle='->', style='dashed')

    labels = {}
    for n, d in G.nodes(data=True):
        lab = d.get('label', n)
        if len(lab) > 25:
            lab = lab[:22] + '...'
        labels[n] = lab
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight='bold')

    legend = []
    for et, color in EDGE_COLORS.items():
        if any(d.get('edge_type') == et for _, _, d in G.edges(data=True)):
            sc = sum(1 for _, _, d in G.edges(data=True)
                     if d.get('edge_type') == et and d.get('confidence', 'strict') == 'strict')
            so = sum(1 for _, _, d in G.edges(data=True)
                     if d.get('edge_type') == et and d.get('confidence', 'strict') == 'soft')
            lab = et
            if sc > 0 and so > 0:
                lab += f' ({sc}S/{so}So)'
            elif sc > 0:
                lab += f' ({sc})'
            legend.append(mpatches.Patch(color=color, label=lab))

    if legend:
        plt.legend(handles=legend, loc='upper left', fontsize=9,
                   title='Edge Types (S=Strict, So=Soft)', title_fontsize=9)

    plt.title(f'Pedagogical Knowledge Graph\n{len(G.nodes())} nodes, {len(G.edges())} edges',
              fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"saved: {output_file}")
    else:
        plt.show()


def print_graph_stats(graph_file):
    data = load_graph(graph_file)
    G = create_networkx_graph(data)
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])

    counts = {}
    strict_total = 0
    soft_total = 0
    for e in edges:
        et = e.get('edge_type', 'unknown')
        conf = e.get('confidence', 'strict')
        counts.setdefault(et, {'strict': 0, 'soft': 0})
        if conf == 'soft':
            counts[et]['soft'] += 1
            soft_total += 1
        else:
            counts[et]['strict'] += 1
            strict_total += 1

    print("\n" + "=" * 70)
    print("GRAPH STATISTICS")
    print("=" * 70)
    print(f"\nTotal Nodes: {len(nodes)}")
    print(f"\nTotal Edges: {len(edges)}")
    print(f"  - Strict edges: {strict_total}")
    print(f"  - Soft edges: {soft_total}")
    print("\nEdges by type:")
    for et in sorted(counts.keys()):
        c = counts[et]
        print(f"  - {et}: {c['strict'] + c['soft']} (Strict: {c['strict']}, Soft: {c['soft']})")

    if len(G.nodes()) > 0:
        print("\nGraph Metrics:")
        print(f"  - Is weakly connected: {nx.is_weakly_connected(G)}")
        if len(G.nodes()) > 1:
            degs = dict(G.degree())
            print(f"  - Average degree: {sum(degs.values()) / len(G.nodes()):.2f}")
            if len(G.edges()) > 0:
                print(f"  - Density: {nx.density(G):.4f}")
                top = sorted(degs.items(), key=lambda x: x[1], reverse=True)[:5]
                print("  - Top 5 nodes by degree:")
                for nid, deg in top:
                    lab = G.nodes[nid].get('label', nid)
                    if len(lab) > 40:
                        lab = lab[:37] + '...'
                    print(f"      {nid}: {deg} connections - {lab}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_graph.py <graph.json> [output.png] [layout]", file=sys.stderr)
        print("Layouts: spring, hierarchical, circular, kamada_kawai", file=sys.stderr)
        sys.exit(1)

    graph_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    layout = sys.argv[3] if len(sys.argv) > 3 else 'spring'

    print_graph_stats(graph_file)
    visualize_graph(graph_file, output_file, layout)
