"""取本机局域网 IP (供 setup.bat / start.bat 的 :get_ip 调用).

原理: UDP connect 只让系统选择默认出口网卡, 不真正发包;
比解析 ipconfig 更准 (多网卡/虚拟网卡时 ipconnect 首个 IPv4 常是错的).
再叠一层过滤+优选: 开了 Clash TUN 等代理时默认出口会走到虚拟网卡
(如 198.18.0.1), 手机根本连不上; 这类地址必须排除, 优选 RFC1918 内网段.

排除: 127/8 回环, 169.254/16 链路本地, 198.18/15 基准测试保留段
(Clash TUN 虚拟网卡常用 198.18.0.1), 224+/0 组播保留.
优选: 192.168/16 > 10/8 = 172.16/12 > 其它单播.

约定: 成功时 stdout 只打印一行 IP (bat 用 for /f 取首行), 不要加多余输出.
"""
import socket

# 排除前缀 (字符串匹配, 快且无依赖)
_EXCLUDED_PREFIXES = ("127.", "169.254.", "198.18.", "198.19.", "0.")

# 虚拟网卡名关键字 (VMware/Hyper-V/WSL 等也占 192.168.x, 但手机连不上;
# 有网卡名时直接跳过, 见 psutil 分支; 纯 stdlib 分支无名可用, 靠段优选)
_VIRTUAL_NAME_KEYWORDS = ("vmware", "vmnet", "virtualbox", "vbox", "hyper-v",
                          "vethernet", "wsl", "loopback", "sandbox")


def _virtual_name(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in _VIRTUAL_NAME_KEYWORDS)


def _usable(ip: str) -> bool:
    if not ip or ip.startswith(_EXCLUDED_PREFIXES):
        return False
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return False
    if any(n < 0 or n > 255 for n in nums):
        return False
    if nums[0] >= 224:  # 组播/保留
        return False
    return True


def _rank(ip: str) -> int:
    if ip.startswith("192.168."):
        return 3
    if ip.startswith("10."):
        return 2
    if ip.startswith("172."):
        try:
            second = int(ip.split(".")[1])
        except ValueError:
            return 1
        if 16 <= second <= 31:
            return 2
    return 1


def get_local_ip() -> str:
    # 首选 psutil (有网卡名, 能点名排除 VMware/WSL 等虚拟网卡);
    # 无 psutil 时退化为 stdlib (默认出口 + getaddrinfo, 只做段过滤优选).
    try:
        import psutil
        cands = []
        for nic, addrs in psutil.net_if_addrs().items():
            if _virtual_name(nic):
                continue
            for a in addrs or []:
                fam = getattr(a.family, "value", a.family)
                if fam == socket.AF_INET or fam == 2:
                    if a.address:
                        cands.append(a.address)
        usable = [c for c in dict.fromkeys(cands) if _usable(c)]
        if usable:
            return max(usable, key=_rank)
    except Exception:
        pass
    cands = []
    default_ip = ""
    try:
        # 不会真正发包，只是让系统选择默认出口网卡
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            default_ip = s.getsockname()[0]
            cands.append(default_ip)
        finally:
            s.close()
    except OSError:
        pass
    try:
        # 本机全部 IPv4 (含真实 LAN 卡, 不只看默认出口)
        for fam, _, _, _, sockaddr in socket.getaddrinfo(socket.gethostname(), None):
            if fam == socket.AF_INET and sockaddr:
                cands.append(sockaddr[0])
    except OSError:
        pass
    uniq = list(dict.fromkeys(cands))  # 去重保序
    usable = [c for c in uniq if _usable(c)]
    if not usable:
        return "127.0.0.1"  # 连可用内网都没有时宁可回环, 也不打印连不上的虚拟 IP
    # 默认出口 IP(可用时)加权: 同段多块网卡并存时优先选真正能上网的那块
    return max(usable, key=lambda c: _rank(c) + (10 if c == default_ip else 0))


if __name__ == "__main__":
    print(get_local_ip())
