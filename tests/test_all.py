"""
Test suite — covers compute, verify, SQL, encryption, cache, planner routing, context assembler.
Run: pytest tests/ -v
"""
import os, sys, time, pathlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest


# ── Compute ───────────────────────────────────────────────────────────────────
from app.tools.compute import (
    compute_per_diem, compute_hotel_entitlement, compute_reimbursement,
    compute_leave_encashment, compute_variable_pay, compute_pro_rata,
    compute_hra, compute_travel_allowance,
)

def test_per_diem():
    assert compute_per_diem(5, 900) == 4500.0
    assert compute_per_diem(0, 900) == 0.0

def test_hotel():
    assert compute_hotel_entitlement(3, 5500) == 16500.0

def test_reimbursement():
    assert compute_reimbursement(3, 900, 3, 5500) == 3*900 + 3*5500

def test_leave_encashment():
    assert compute_leave_encashment(60000, 15) == round((60000/26)*15, 2)

def test_variable_pay():
    assert compute_variable_pay(600000, 10) == 60000.0

def test_pro_rata():
    assert compute_pro_rata(12000, 15, 30) == 6000.0
    assert compute_pro_rata(12000, 0, 0)   == 0.0

def test_hra():
    assert compute_hra(40000, "metro")     == 20000.0
    assert compute_hra(40000, "non-metro") == 16000.0

def test_travel_allowance_rows():
    rows = [
        {"policy_code":"T-04","category":"meal","per_day_inr":900,"per_night_inr":None},
        {"policy_code":"T-04","category":"hotel","per_day_inr":None,"per_night_inr":5500},
    ]
    total, steps = compute_travel_allowance(rows, {"days":3,"nights":3})
    assert total == 3*900 + 3*5500
    assert len(steps) > 0

def test_travel_allowance_no_rows():
    total, steps = compute_travel_allowance([], {"days":3})
    assert total is None and steps == []


# ── Verify ────────────────────────────────────────────────────────────────────
from app.tools.verify import verify_numerical_consistency, verify_no_fabrication

def test_verify_numbers_pass():
    ctx = "Hotel rate is ₹5,500 per night."
    ok, _ = verify_numerical_consistency("For 3 nights: ₹5,500 × 3 = ₹16,500", ctx)
    assert ok

def test_verify_numbers_fail():
    ctx = "Hotel rate is ₹5,500 per night."
    ok, issues = verify_numerical_consistency("Rate is ₹9,999 per night.", ctx)
    assert not ok and issues

def test_fabrication_pass():
    ok, _ = verify_no_fabrication("The allowance is ₹5,500 per night per Policy T-04.")
    assert ok

def test_fabrication_fail():
    ok, issues = verify_no_fabrication("I think the rate is approximately ₹5,000.")
    assert not ok and issues


# ── Encryption ────────────────────────────────────────────────────────────────
def test_encrypt_decrypt():
    from app.utils.encryption import encrypt, decrypt
    msg = "Hotel allowance ₹12,000/night for VP grade."
    assert decrypt(encrypt(msg)) == msg

def test_encrypt_unique():
    from app.utils.encryption import encrypt
    assert encrypt("same") != encrypt("same")  # Fernet random IV


# ── Cache ─────────────────────────────────────────────────────────────────────
def test_cache_set_get(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CACHE_TTL_SECONDS", "60")
    from importlib import reload; import app.utils.pickle_cache as m; reload(m)
    c = m.PickleCache()
    c.set("q", b"answer")
    assert c.get("q") == b"answer"

def test_cache_miss(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    from importlib import reload; import app.utils.pickle_cache as m; reload(m)
    assert m.PickleCache().get("missing") is None

def test_cache_ttl(tmp_path, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("CACHE_TTL_SECONDS", "0")
    from importlib import reload; import app.utils.pickle_cache as m; reload(m)
    c = m.PickleCache(); c.set("q", b"v"); time.sleep(0.01)
    assert c.get("q") is None


# ── SQL ───────────────────────────────────────────────────────────────────────
def test_sql_domestic_meal(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path/"p.db"))
    from importlib import reload; import app.tools.sql as m; reload(m)
    rows = m.PolicyDatabase().query_policy(travel_type="domestic", category="meal")
    assert rows and all(r["travel_type"]=="domestic" for r in rows)

def test_sql_hotel_vp(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path/"p2.db"))
    from importlib import reload; import app.tools.sql as m; reload(m)
    rows = m.PolicyDatabase().query_policy(grade="VP", travel_type="domestic", category="hotel")
    assert any(r["per_night_inr"] == 12000 for r in rows)


# ── Planner routing ───────────────────────────────────────────────────────────
from app.nodes.planner import _keyword_route

def test_planner_sql():
    assert _keyword_route("What is the hotel allowance for L5?") == "sql"

def test_planner_compute():
    assert _keyword_route("calculate total for 5 days") == "compute"

def test_planner_retrieval():
    assert _keyword_route("What is the approval process for leave?") == "retrieval"


# ── Context assembler ─────────────────────────────────────────────────────────
from app.nodes.context_assembler import run as assemble

def test_context_includes_sql():
    s = assemble({"query":"hotel L5","sql_result":[{"policy_code":"T-04","per_night_inr":5500}],
                  "retrieval_docs":[],"compute_result":None,"compute_summary":"","history":[],"trace_log":[]})
    assert "5500" in s["context"] or "5,500" in s["context"]

def test_context_includes_compute():
    s = assemble({"query":"total","sql_result":[],"retrieval_docs":[],
                  "compute_result":22500.0,"compute_summary":"Total: 22500","history":[],"trace_log":[]})
    assert "22500" in s["context"] or "22,500" in s["context"]
