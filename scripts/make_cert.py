#!/usr/bin/env python3
"""
生成自签 HTTPS 证书 —— 让手机也能用麦克风

为什么需要：
    浏览器只在「安全上下文」下允许 getUserMedia：
      ✅ http://localhost / 127.0.0.1
      ✅ https://任意地址
      ❌ http://192.168.x.x     ← 手机访问会被直接拒绝

    所以手机测试必须走 HTTPS。自签证书就够了 —— 浏览器会警告，
    点「继续前往」即可。这只是本机局域网使用，不需要正式证书。

用法:
    .venv/bin/python make_cert.py
    .venv/bin/python server.py --host 0.0.0.0 --ssl
"""

import ipaddress
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone

CERT_DIR = "certs"
CERT = os.path.join(CERT_DIR, "cert.pem")
KEY = os.path.join(CERT_DIR, "key.pem")

C = {"dim": "\033[2m", "ok": "\033[32m", "err": "\033[31m",
     "warn": "\033[33m", "r": "\033[0m"}


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def main():
    os.makedirs(CERT_DIR, exist_ok=True)

    if os.path.exists(CERT) and os.path.exists(KEY) and "--force" not in sys.argv:
        print(f"{C['warn']}证书已存在：{CERT}{C['r']}")
        print(f"  要重新生成请加 --force")
        return 0

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        print(f"{C['err']}需要 cryptography 库{C['r']}")
        print("  安装：.venv/bin/python -m pip install cryptography")
        return 1

    ip = lan_ip()
    print(f"  生成自签证书…")
    if ip:
        print(f"  将包含局域网 IP: {ip}")

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, ip or "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "English Practice Local"),
    ])

    alt = [x509.DNSName("localhost")]
    try:
        alt.append(x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")))
        if ip:
            alt.append(x509.IPAddress(ipaddress.IPv4Address(ip)))
    except Exception:
        pass

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1))
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None),
                       critical=True)
        .sign(key, hashes.SHA256())
    )

    with open(KEY, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()))
    with open(CERT, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    os.chmod(KEY, 0o600)

    print(f"\n{C['ok']}✅ 证书已生成{C['r']}")
    print(f"  {CERT}")
    print(f"  {KEY}")
    print(f"\n  启动服务：")
    print(f"    .venv/bin/python server.py --host 0.0.0.0 --ssl")
    if ip:
        print(f"\n  手机访问：{C['ok']}https://{ip}:8000{C['r']}")
        print(f"  {C['dim']}首次会提示证书不受信任 → 点「高级」→「继续前往」{C['r']}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
