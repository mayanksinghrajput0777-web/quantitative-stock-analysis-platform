"""Stock universe. Edit data/raw/universe.csv (columns: symbol,yf_ticker,name) to override.
NOTE: index constituents change. Tickers that return no data are skipped and reported in the quality log."""
import pandas as pd
import config

NIFTY50 = [
 ("ADANIENT","Adani Enterprises"),("ADANIPORTS","Adani Ports"),("APOLLOHOSP","Apollo Hospitals"),
 ("ASIANPAINT","Asian Paints"),("AXISBANK","Axis Bank"),("BAJAJ-AUTO","Bajaj Auto"),
 ("BAJFINANCE","Bajaj Finance"),("BAJAJFINSV","Bajaj Finserv"),("BEL","Bharat Electronics"),
 ("BHARTIARTL","Bharti Airtel"),("CIPLA","Cipla"),("COALINDIA","Coal India"),("DRREDDY","Dr Reddys"),
 ("EICHERMOT","Eicher Motors"),("GRASIM","Grasim"),("HCLTECH","HCL Technologies"),
 ("HDFCBANK","HDFC Bank"),("HDFCLIFE","HDFC Life"),("HEROMOTOCO","Hero MotoCorp"),
 ("HINDALCO","Hindalco"),("HINDUNILVR","Hindustan Unilever"),("ICICIBANK","ICICI Bank"),
 ("INDUSINDBK","IndusInd Bank"),("INFY","Infosys"),("ITC","ITC"),("JSWSTEEL","JSW Steel"),
 ("KOTAKBANK","Kotak Mahindra Bank"),("LT","Larsen & Toubro"),("M&M","Mahindra & Mahindra"),
 ("MARUTI","Maruti Suzuki"),("NESTLEIND","Nestle India"),("NTPC","NTPC"),("ONGC","ONGC"),
 ("POWERGRID","Power Grid"),("RELIANCE","Reliance Industries"),("SBILIFE","SBI Life"),
 ("SBIN","State Bank of India"),("SHRIRAMFIN","Shriram Finance"),("SUNPHARMA","Sun Pharma"),
 ("TATACONSUM","Tata Consumer"),("TATASTEEL","Tata Steel"),("TCS","Tata Consultancy Services"),
 ("TECHM","Tech Mahindra"),("TITAN","Titan Company"),("TRENT","Trent"),("ULTRACEMCO","UltraTech Cement"),
 ("WIPRO","Wipro"),("JIOFIN","Jio Financial Services"),("BRITANNIA","Britannia"),("DIVISLAB","Divis Laboratories"),
 ("TATAMOTORS","Tata Motors"),
]

def yf_ticker(sym: str) -> str:
    return sym + ".NS"

def get_universe() -> pd.DataFrame:
    f = config.RAW_DIR / "universe.csv"
    if f.exists():
        df = pd.read_csv(f)
        if "yf_ticker" not in df.columns:
            df["yf_ticker"] = df["symbol"].map(yf_ticker)
        if "name" not in df.columns:
            df["name"] = df["symbol"]
        return df[["symbol", "yf_ticker", "name"]]
    return pd.DataFrame([(s, yf_ticker(s), n) for s, n in NIFTY50], columns=["symbol", "yf_ticker", "name"])
