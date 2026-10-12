"""签到 DB 层 (sqlite3 标准库, 无额外依赖).

表结构按需求文档 T14 数据字典:
    签到记录 = 学号 + 姓名 + 识别置信度 + 签到时间 + 摄像头编号
"""
import os
import sqlite3
from datetime import datetime
from typing import List, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    name TEXT NOT NULL,
    sim REAL NOT NULL,
    ts TEXT NOT NULL,
    day TEXT NOT NULL,
    cam_id TEXT NOT NULL DEFAULT 'cam0',
    UNIQUE(person_id, day)
);
CREATE INDEX IF NOT EXISTS idx_att_day ON attendance(day);
"""


def get_conn(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db(db_path: str) -> None:
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def try_sign_in(db_path: str, person_id: str, name: str, sim: float,
                cam_id: str = "cam0", min_sim: float = 0.4,
                cooldown_s: int = 60) -> Tuple[bool, str]:
    """尝试写入一条签到记录.

    Returns:
        (True, "signed") 写入成功
        (False, reason) 未写入, reason in:
          stranger / low-sim / duplicate-today / cooldown
    """
    if not person_id or person_id == "stranger":
        return False, "stranger"
    if sim < min_sim:
        return False, "low-sim"
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    conn = get_conn(db_path)
    try:
        conn.executescript(SCHEMA)
        cur = conn.execute(
            "SELECT ts FROM attendance WHERE person_id=? AND day=? ORDER BY ts DESC LIMIT 1",
            (person_id, day))
        row = cur.fetchone()
        if row is not None:
            try:
                last = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
                if (now - last).total_seconds() < cooldown_s:
                    return False, "cooldown"
            except Exception:
                pass
            return False, "duplicate-today"
        conn.execute(
            "INSERT INTO attendance(person_id,name,sim,ts,day,cam_id) VALUES(?,?,?,?,?,?)",
            (person_id, name, float(sim), ts, day, cam_id))
        conn.commit()
        return True, "signed"
    except sqlite3.IntegrityError:
        return False, "duplicate-today"
    finally:
        conn.close()


def fetch_today(db_path: str) -> List[tuple]:
    day = datetime.now().strftime("%Y-%m-%d")
    conn = get_conn(db_path)
    try:
        return list(conn.execute(
            "SELECT person_id,name,sim,ts,cam_id FROM attendance WHERE day=? ORDER BY ts",
            (day,)).fetchall())
    finally:
        conn.close()
