import json
import sys


def build_graph(nodes, edges):
    """노드/엣지 리스트를 KG dict로. 무효 엣지(끊긴 노드 참조)는 떨어뜨림."""
    nmap = {}
    for n in nodes:
        nid = n.get('id') or n.get('node_id')
        if nid and nid not in nmap:
            nmap[nid] = dict(n)

    out_edges = []
    seen_eids = set()
    for i, e in enumerate(edges, 1):
        src = e.get('from') or e.get('source_node')
        tgt = e.get('to') or e.get('target_node')
        if not src or not tgt or src not in nmap or tgt not in nmap:
            continue
        eid = e.get('edge_id') or f"edge_{i:03d}"
        if eid in seen_eids:
            continue
        seen_eids.add(eid)
        ee = dict(e)
        ee.setdefault('from', src)
        ee.setdefault('to', tgt)
        ee['edge_id'] = eid
        out_edges.append(ee)

    return {
        "nodes": list(nmap.values()),
        "edges": out_edges,
        "metadata": {},
    }


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_graph.py <nodes.json> <edges.json>", file=sys.stderr)
        sys.exit(1)
    try:
        nodes = json.load(open(sys.argv[1], encoding='utf-8')).get('nodes', [])
        edges = json.load(open(sys.argv[2], encoding='utf-8')).get('edges', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(build_graph(nodes, edges), indent=2))
