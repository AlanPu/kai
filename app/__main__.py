"""启动服务：python -m app [--host 0.0.0.0] [--port 8000] [--ssl]"""
import argparse
import sys
from pathlib import Path

import uvicorn

# make_cert.py 的默认输出位置
DEFAULT_CERT = Path("certs/cert.pem")
DEFAULT_KEY = Path("certs/key.pem")


def main():
    ap = argparse.ArgumentParser(prog="app")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")

    # HTTPS 不是可选项：浏览器只在 localhost 或 https 下给麦克风权限。
    # 想用手机/平板连电脑练习就必须走 HTTPS，所以这个开关得真的能用 ——
    # 之前文档里写了 --ssl-keyfile，但代码根本没实现，
    # 照着做只会启动失败。现在补上，并让它自动找 make_cert.py 的产物。
    ap.add_argument("--ssl", action="store_true",
                    help="启用 HTTPS（手机访问必需）")
    ap.add_argument("--ssl-certfile", default=None,
                    help=f"证书路径（默认 {DEFAULT_CERT}）")
    ap.add_argument("--ssl-keyfile", default=None,
                    help=f"私钥路径（默认 {DEFAULT_KEY}）")

    a = ap.parse_args()

    cert = Path(a.ssl_certfile) if a.ssl_certfile else DEFAULT_CERT
    key = Path(a.ssl_keyfile) if a.ssl_keyfile else DEFAULT_KEY

    # 显式给了证书路径，就等于要 HTTPS —— 不必再写 --ssl
    if a.ssl_certfile or a.ssl_keyfile:
        a.ssl = True

    if a.ssl and not (cert.is_file() and key.is_file()):
        print(f"❌ 找不到证书：{cert} / {key}", file=sys.stderr)
        print("   先生成：.venv/bin/python scripts/make_cert.py",
              file=sys.stderr)
        sys.exit(1)

    uvicorn.run(
        "app.api.server:app",
        host=a.host,
        port=a.port,
        reload=a.reload,
        log_level="info",
        ssl_certfile=str(cert) if a.ssl else None,
        ssl_keyfile=str(key) if a.ssl else None,
    )


if __name__ == "__main__":
    main()
