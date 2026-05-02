import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import aegis_colab_chat as colab


def test_basic_chat_result_has_answer():
    result = colab.normalise_result({"answer": "No response generated", "trace_log": []}, "hello")

    assert result["route"] == "chat"
    assert result["intent"] == "chat"
    assert "AEGIS" in result["answer"] or "policy" in result["answer"].lower()


def test_node_rows_mark_visited_nodes():
    rows = colab.node_rows([{"node": "trace_start"}, {"node": "planner"}, {"node": "retrieval"}])
    visited = {row["node"] for row in rows if row["status"] == "visited"}

    assert {"trace_start", "planner", "retrieval"}.issubset(visited)
    assert len(rows) == len(colab.NODE_SPECS)


def test_workflow_edges_reference_known_nodes():
    names = {node.name for node in colab.NODE_SPECS}

    assert all(source in names and target in names for source, target, _ in colab.WORKFLOW_EDGES)
