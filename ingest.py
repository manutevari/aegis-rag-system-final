"""Legacy CLI wrapper for the canonical policy ingestion pipeline."""

from policy_ingestion import run_ingestion


if __name__ == "__main__":
    result = run_ingestion()
    raise SystemExit(0 if result.get("status") == "success" else 1)
