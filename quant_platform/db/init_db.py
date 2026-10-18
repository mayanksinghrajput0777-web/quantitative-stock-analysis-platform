import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config
from db.connection import init_db, get_engine, export_schema_sql

def main():
    eng = get_engine()
    tables = init_db(eng)
    export_schema_sql(config.ROOT / "db" / "schema.sql")
    print(f"✅ Database ready: {len(tables)} tables -> {', '.join(tables)}")

if __name__ == "__main__":
    main()
