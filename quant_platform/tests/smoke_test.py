"""End-to-end smoke test on a throw-away SQLite DB with SYNTHETIC prices + your real FII file (if in data/raw).
Run:  python tests/smoke_test.py        (takes ~1-2 minutes; needs no PostgreSQL / internet)"""
import os, sys, subprocess, tempfile, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
db = pathlib.Path(tempfile.gettempdir()) / "quant_smoke.db"
db.unlink(missing_ok=True)
env = {**os.environ, "QUANT_DATABASE_URL": f"sqlite:///{db.as_posix()}"}
r = subprocess.run([sys.executable, "run_pipeline.py", "bootstrap", "--synthetic", "12", "--fast"], cwd=ROOT, env=env)
assert r.returncode == 0, "bootstrap failed"
for f in ["output/final_predictions_next5.csv", "output/backtest_comparison.csv",
          "output/stress_tests.csv", "output/eda/eda_report.html", "data/artifacts/models/alpha_bundle.joblib"]:
    assert (ROOT / f).exists(), f"missing {f}"
from fastapi.testclient import TestClient
os.environ["QUANT_DATABASE_URL"] = env["QUANT_DATABASE_URL"]
sys.path.insert(0, str(ROOT))
from api.main import app
c = TestClient(app)
for p in ["/health", "/predict", "/risk", "/market/overview", "/backtest/latest", "/monitoring", "/portfolio/latest", "/news"]:
    assert c.get(p).status_code == 200, p
print("\n✅ SMOKE TEST PASSED")
