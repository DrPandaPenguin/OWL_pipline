from collections import defaultdict


def _bfs_components(adj, all_nodes):
    visited = set()
    comps = []
    for start in all_nodes:
        if start in visited:
            continue
        comp = set()
        queue = [start]
        while queue:
            n = queue.pop(0)
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in adj.get(n, set()):
                if nb not in visited:
                    queue.append(nb)
        comps.append(comp)
    return comps


def _ft(e):
    return (e.get("from") or e.get("source_node"),
            e.get("to") or e.get("target_node"))


def compute_all_metrics(nodes, edges, kus, config):
    node_ids = {n.get("id") or n.get("node_id") for n in nodes} - {None, ""}
    ku_ids = {k.get("id") for k in kus} - {None, ""}
    edge_types_cfg = set(config.get("edge_types", []))

    node_count = len(nodes)
    edge_count = len(edges)
    ku_count = len(kus)

    strict_edges = [e for e in edges if "justification_sentence" in e]
    soft_edges = [e for e in edges
                  if "justification_span" in e
                  or ("confidence_score" in e and "justification_sentence" not in e)]

    valid_edges = []
    for e in edges:
        fr, to = _ft(e)
        etype = e.get("edge_type")
        has_just = bool((e.get("justification_sentence") or "").strip()
                        or (e.get("justification_span") or "").strip())
        if (fr in node_ids and to in node_ids
                and (not edge_types_cfg or etype in edge_types_cfg)
                and has_just):
            valid_edges.append(e)
    valid_count = len(valid_edges)
    valid_ratio = valid_count / edge_count if edge_count > 0 else 0.0

    edge_type_dist = defaultdict(int)
    for e in edges:
        edge_type_dist[e.get("edge_type", "unknown")] += 1

    used_in_edges = set()
    for e in edges:
        fr, to = _ft(e)
        if fr:
            used_in_edges.add(fr)
        if to:
            used_in_edges.add(to)
    orphan_nodes = len(node_ids - used_in_edges)

    used_kus = set()
    for n in nodes:
        for kid in n.get("supporting_ku_ids") or []:
            used_kus.add(kid)
    orphan_kus = len(ku_ids - used_kus)

    adj = defaultdict(set)
    for e in edges:
        fr, to = _ft(e)
        if fr and to and fr in node_ids and to in node_ids:
            adj[fr].add(to)
            adj[to].add(fr)
    comps = _bfs_components(adj, node_ids)
    largest = max((len(c) for c in comps), default=0)
    max_edges = node_count * (node_count - 1) if node_count > 1 else 1
    density = edge_count / max_edges

    soft_confs = []
    for e in soft_edges:
        c = e.get("confidence_score")
        if c is not None:
            try:
                soft_confs.append(float(c))
            except (TypeError, ValueError):
                pass
    avg_soft = sum(soft_confs) / len(soft_confs) if soft_confs else 0.0

    kus_per_node = [len(n.get("supporting_ku_ids") or []) for n in nodes]
    avg_kpn = sum(kus_per_node) / len(kus_per_node) if kus_per_node else 0.0

    self_loops = sum(1 for e in edges if _ft(e)[0] == _ft(e)[1])

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "ku_count": ku_count,
        "strict_count": len(strict_edges),
        "soft_count": len(soft_edges),
        "valid_edge_count": valid_count,
        "valid_edge_ratio": round(valid_ratio, 4),
        "hallucination_count": edge_count - valid_count,
        "edge_type_distribution": dict(edge_type_dist),
        "orphan_node_count": orphan_nodes,
        "orphan_node_ratio": round(orphan_nodes / node_count, 4) if node_count else 0.0,
        "orphan_ku_count": orphan_kus,
        "orphan_ku_ratio": round(orphan_kus / ku_count, 4) if ku_count else 0.0,
        "num_connected_components": len(comps),
        "is_weakly_connected": len(comps) == 1 and node_count > 0,
        "largest_component_size": largest,
        "graph_density": round(density, 4),
        "avg_soft_confidence": round(avg_soft, 4),
        "avg_kus_per_node": round(avg_kpn, 2),
        "self_loop_count": self_loops,
    }
