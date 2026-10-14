"""取本机局域网 IP (供 setup.bat / start.bat 的 :get_ip 调用).

原理: UDP connect 只让系统选择默认出口网卡, 不真正发包;
比解析 ipconfig 更准 (多网卡/虚拟网卡时 ipconfig 首个 IPv4 常是错的).

约定: 成功时 stdout 只打印一行 IP (bat 用 for /f 取首行), 不要加多余输出.
"""
import socket


def get_local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 不会真正发包，只是让系统选择默认出口网卡
        s.connect(("8.8.8.8", 80))
        ip: str = s.getsockname()[0]
    except OSError:
        # 兜底方案
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    print(get_local_ip())
