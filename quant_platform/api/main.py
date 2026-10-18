"""FastAPI backend.  Run:  uvicorn api.main:app --reload --port 8000   (docs at http://127.0.0.1:8000/docs)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import config
from app import services

app = FastAPI(title="Quant AI Platform API", version="1.0")


class OptimizeRequest(BaseModel):
    symbols: List[str]
    method: str = "max_sharpe"            # max_sharpe | min_variance | risk_parity | equal
    max_weight: float = 0.20
    capital: float = config.CAPITAL


@app.get("/health")
def health(): return services.health()

@app.get("/symbols")
def symbols(): return services.symbols()

@app.get("/market/overview")
def market(days: int = 500): return services.market_overview(days)

@app.get("/predict")
def predict_all(): return services.latest_predictions()

@app.get("/predict/{symbol}")
def predict(symbol: str):
    d = services.prediction_detail(symbol.upper())
    if not d["current"]:
        raise HTTPException(404, f"no live prediction for {symbol}")
    return d

@app.get("/predictions/history/{symbol}")
def pred_history(symbol: str, days: int = 250): return services.prediction_history(symbol.upper(), days)

@app.get("/risk")
def risk_all(): return services.risk_latest()

@app.get("/risk/{symbol}")
def risk(symbol: str): return services.risk_detail(symbol.upper())

@app.get("/news")
def news(symbol: str = None, days: int = 30, limit: int = 200): return services.news(symbol.upper() if symbol else None, days, limit)

@app.post("/portfolio/optimize")
def optimize(req: OptimizeRequest):
    try:
        return services.optimize_custom([s.upper() for s in req.symbols], req.method, req.max_weight, req.capital)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/portfolio/latest")
def portfolio(): return services.portfolio_latest()

@app.get("/backtest/latest")
def backtest(): return services.backtest_latest()

@app.get("/monitoring")
def monitoring(): return services.monitoring_summary()
