"""生成自签证书 (key.pem + cert.pem), 供 uvicorn HTTPS 使用.

依赖: cryptography (缺则 pip install cryptography).
用法 (start.bat 缺证书时自动调用, 也可手动跑):
    python tools/gen_cert.py [--out-dir certs] [--ip 192.168.1.10] [--days 825]

SAN 固定含 DNS:localhost + IP:127.0.0.1, 另加 --ip 传入的局域网 IP (浏览器同源访问不报域 mismatch).
"""
import argparse
import datetime
import ipaddress
import os
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成自签证书")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "certs"))
    ap.add_argument("--cn", default="localhost")
    ap.add_argument("--ip", action="append", default=[],
                    help="追加进 SAN 的局域网 IP, 可多次传入")
    ap.add_argument("--days", type=int, default=825)
    args = ap.parse_args(argv)

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        print("[gen_cert] 缺少 cryptography, 先执行: pip install cryptography",
              file=sys.stderr)
        return 1

    sans = [x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    for ip in args.ip:
        ip = ip.strip()
        if not ip:
            continue
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            print(f"[gen_cert] 跳过非法 IP: {ip}", file=sys.stderr)

    print(f"[gen_cert] 正在生成自签证书 (CN={args.cn}, {args.days}天)...", flush=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, args.cn)])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=args.days))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))

    os.makedirs(args.out_dir, exist_ok=True)
    key_path = os.path.join(args.out_dir, "key.pem")
    cert_path = os.path.join(args.out_dir, "cert.pem")
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"[gen_cert] 已生成: {key_path} / {cert_path} "
          f"(CN={args.cn}, {args.days}天, SAN={len(sans)}项)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
