"""
PolicyDatabase — Parameterised SQL against the policy rates table.
Backend: SQLite (dev, default) or PostgreSQL (prod, DB_BACKEND=postgres).
"""
import logging, os, pathlib, sqlite3
from contextlib import contextmanager
from typing import List, Optional
logger     = logging.getLogger(__name__)
BACKEND    = os.getenv("DB_BACKEND", "sqlite")
SQLITE_PATH = os.getenv("SQLITE_PATH", "data/policy.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_rates (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_code      TEXT NOT NULL,
    category         TEXT NOT NULL,
    grade            TEXT NOT NULL,
    travel_type      TEXT,
    per_day_inr      REAL,
    per_night_inr    REAL,
    annual_limit_inr REAL,
    currency         TEXT DEFAULT 'INR',
    department       TEXT DEFAULT 'ALL',
    effective_date   TEXT,
    notes            TEXT
);"""

_SEED = [
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','meal','L1-L3','domestic',600,NULL,NULL,'INR','ALL','2025-04-01','Meal per day')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','meal','L4-L5','domestic',900,NULL,NULL,'INR','ALL','2025-04-01','Meal per day')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','meal','L6-L7','domestic',1200,NULL,NULL,'INR','ALL','2025-04-01','Meal per day')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','meal','VP','domestic',1800,NULL,NULL,'INR','ALL','2025-04-01','Meal per day')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','hotel','L1-L3','domestic',NULL,3500,NULL,'INR','ALL','2025-04-01','Hotel per night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','hotel','L4-L5','domestic',NULL,5500,NULL,'INR','ALL','2025-04-01','Hotel per night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','hotel','L6-L7','domestic',NULL,8000,NULL,'INR','ALL','2025-04-01','Hotel per night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','hotel','VP','domestic',NULL,12000,NULL,'INR','ALL','2025-04-01','Hotel per night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','per_diem','L1-L3','international',4150,9960,NULL,'INR','ALL','2025-04-01','USD50/day USD120/night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','per_diem','L4-L5','international',6225,14940,NULL,'INR','ALL','2025-04-01','USD75/day USD180/night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','per_diem','L6-L7','international',8300,20750,NULL,'INR','ALL','2025-04-01','USD100/day USD250/night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','per_diem','VP','international',12450,33200,NULL,'INR','ALL','2025-04-01','USD150/day USD400/night')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'T-04','transport','ALL','local',3000,NULL,NULL,'INR','ALL','2025-04-01','Max per day cab')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'IT-09','laptop','L1-L3',NULL,NULL,NULL,55000,'INR','ALL','2024-07-01','4yr cycle')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'IT-09','laptop','L4-L5',NULL,NULL,NULL,85000,'INR','ALL','2024-07-01','3yr cycle')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'IT-09','laptop','L6-L7',NULL,NULL,NULL,120000,'INR','ALL','2024-07-01','3yr cycle')",
    "INSERT OR IGNORE INTO policy_rates VALUES(NULL,'IT-09','laptop','VP',NULL,NULL,NULL,180000,'INR','ALL','2024-07-01','Premium+dock')",
]

class PolicyDatabase:
    def __init__(self):
        if BACKEND == "sqlite":
            db = pathlib.Path(SQLITE_PATH)
            db.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(db)
            with sqlite3.connect(self._path) as c:
                c.executescript(_SCHEMA)
                for s in _SEED: c.execute(s)
                c.commit()
            logger.info("SQLite DB ready: %s", self._path)
        else:
            import psycopg2.pool
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                1, 10,
                host=os.getenv("PG_HOST","localhost"), port=int(os.getenv("PG_PORT","5432")),
                dbname=os.getenv("PG_DB","policy_rag"), user=os.getenv("PG_USER","postgres"),
                password=os.getenv("PG_PASSWORD",""),
            )

    @contextmanager
    def _cursor(self):
        if BACKEND == "sqlite":
            c = sqlite3.connect(self._path); c.row_factory = sqlite3.Row
            try: yield c.cursor(); c.commit()
            finally: c.close()
        else:
            conn = self._pool.getconn()
            try: yield conn.cursor(); conn.commit()
            finally: self._pool.putconn(conn)

    def query_policy(self, grade=None, travel_type=None, category=None,
                     policy_code=None, department=None, limit=20) -> List[dict]:
        conds, params = [], []
        if grade:
            conds.append("grade LIKE ?"); params.append(f"%{grade}%")
        if travel_type:
            conds.append("travel_type=?"); params.append(travel_type.lower())
        if category:
            conds.append("category LIKE ?"); params.append(f"%{category}%")
        if policy_code:
            conds.append("policy_code=?"); params.append(policy_code.upper())
        if department:
            conds.append("(department=? OR department='ALL')"); params.append(department)
        where = ("WHERE " + " AND ".join(conds)) if conds else ""
        sql = f"SELECT * FROM policy_rates {where} ORDER BY grade LIMIT ?"
        params.append(limit)
        with self._cursor() as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
        logger.info("policy_rates: %d rows returned", len(rows))
        return rows
