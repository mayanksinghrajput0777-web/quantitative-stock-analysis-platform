"""SIMPLE LAUNCHER - the only commands you need:

    python start.py check        test packages + PostgreSQL connection (creates the database if missing)
    python start.py setup        FIRST TIME: pull 5 years of data, build everything, produce your results (~10-25 min)
    python start.py daily        EVERY DAY (after 7 pm IST): pull only the new day, update predictions
    python start.py dashboard    open the web dashboard (API + Streamlit)

Options for setup:  --fast (quicker, shorter validation)   --skip-news   --skip-fundamentals
"""
import sys, subprocess, webbrowser, time


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    rest = sys.argv[2:]
    if cmd in ("help", "-h", "--help"):
        print(__doc__); return
    if cmd == "check":
        sys.argv = ["run_pipeline.py", "check"]
    elif cmd == "setup":
        sys.argv = ["run_pipeline.py", "bootstrap"] + rest
    elif cmd == "daily":
        sys.argv = ["run_pipeline.py", "daily"] + rest
    elif cmd == "dashboard":
        print("Starting API on http://127.0.0.1:8000 and dashboard on http://localhost:8501  (Ctrl+C to stop)")
        api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.main:app", "--port", "8000"])
        try:
            time.sleep(2)
            subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])
        finally:
            api.terminate()
        return
    else:
        print(f"Unknown command '{cmd}'\n{__doc__}"); return
    import run_pipeline
    run_pipeline.main()


if __name__ == "__main__":
    main()
