"""启动服务：python -m app [--port 8000] [--host 127.0.0.1]"""
import argparse
import uvicorn

def main():
    ap = argparse.ArgumentParser(prog="app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    a = ap.parse_args()
    uvicorn.run("app.api.server:app", host=a.host, port=a.port,
                reload=a.reload, log_level="info")

if __name__ == "__main__":
    main()
