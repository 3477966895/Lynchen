import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    KMeans = None
    StandardScaler = None

try:
    from web3 import Web3
except Exception:  # pragma: no cover
    Web3 = None

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = BASE_DIR / "tracecoin.db"

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN_CHAIN_ID = os.getenv("ETHERSCAN_CHAIN_ID", "1")
ETHERSCAN_URL = "https://api.etherscan.io/v2/api"
DEFAULT_TX_LIMIT = int(os.getenv("TX_LIMIT", "50"))

WEB3_PROVIDER_URI = os.getenv("WEB3_PROVIDER_URI", "http://127.0.0.1:7545")
CONTRACT_ADDRESS = os.getenv("CONTRACT_ADDRESS", "")
SERVER_PRIVATE_KEY = os.getenv("SERVER_PRIVATE_KEY", "")

BLACKLIST = {"0x000000000000000000000000000000000000dead"}

LABEL_HIGH_FREQ = "高频交互簇"
LABEL_HIGH_FEE = "高费率交互簇"
LABEL_HIGH_VOLUME = "大额流转簇"
LABEL_NORMAL = "一般交互簇"

CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "address", "name": "_target", "type": "address"},
            {"internalType": "string", "name": "_hash", "type": "string"},
        ],
        "name": "storeEvidence",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"internalType": "address", "name": "_target", "type": "address"}],
        "name": "getEvidences",
        "outputs": [
            {
                "components": [
                    {"internalType": "address", "name": "target", "type": "address"},
                    {"internalType": "string", "name": "pdfHash", "type": "string"},
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                ],
                "internalType": "struct TraceCoin.Evidence[]",
                "name": "",
                "type": "tuple[]",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

app = Flask(__name__)
CORS(app)

UI_PAGES = {"index.html", "analyze.html", "batch.html", "insight.html", "evidence.html"}

PDF_FONT_CANDIDATES = [
    # Optional project-local font override:
    # put a TTF font at ./assets/fonts/cjk.ttf for stable offline PDF rendering.
    BASE_DIR / "assets" / "fonts" / "cjk.ttf",
    # Windows common CJK fonts.
    Path(r"C:\Windows\Fonts\simhei.ttf"),
]

# 演示缓存：用于导出、批量复制、聚类与最新交易检测
cache: dict[str, Any] = {
    "address": None,
    "transactions": [],
    "counterparties": {},
    "cluster_labels": None,
    "latest_tx_hash": None,
}


class ValidationError(Exception):
    pass


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db_conn()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                address TEXT NOT NULL,
                report_hash TEXT NOT NULL,
                tx_count INTEGER NOT NULL,
                risk_score INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                chain_status TEXT NOT NULL,
                chain_tx_hash TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_address_id ON evidence_records(address, id DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_evidence_addr_hash ON evidence_records(address, report_hash)")
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_evidence_chain_tx
            ON evidence_records(chain_tx_hash)
            WHERE chain_tx_hash IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                address TEXT,
                payload TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def write_audit(action: str, address: str | None, payload: dict[str, Any]) -> None:
    conn = db_conn()
    try:
        conn.execute(
            "INSERT INTO audit_logs(action,address,payload,created_at) VALUES (?,?,?,?)",
            (action, address, json.dumps(payload, ensure_ascii=False), utc_now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def normalize_address(address: str) -> str:
    address = (address or "").strip()
    if not re.fullmatch(r"0x[a-fA-F0-9]{40}", address):
        raise ValidationError("address must be a valid EVM address")
    return address.lower()


def canonical_hash(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def parse_eth(value_wei: Any) -> float:
    try:
        return int(str(value_wei)) / 1e18
    except Exception:
        return 0.0


def parse_amount(value: Any, decimals: Any = 18) -> float:
    try:
        base = int(str(value))
        dec = int(str(decimals))
        if dec < 0:
            dec = 18
        return base / (10 ** dec)
    except Exception:
        return 0.0


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return default


def vec_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    if np is not None:
        return float(np.mean(values))
    return float(sum(values) / len(values))


def vec_sum(values: list[float]) -> float:
    if not values:
        return 0.0
    if np is not None:
        return float(np.sum(values))
    return float(sum(values))


def vec_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    if np is not None:
        return float(np.std(values))
    mean = vec_mean(values)
    return float((sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5)


def parse_limit(raw: Any, default_limit: int = DEFAULT_TX_LIMIT) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default_limit
    return max(10, min(value, 100))


def etherscan_request(action: str, address: str, limit: int) -> list[dict[str, Any]]:
    if not ETHERSCAN_API_KEY:
        raise RuntimeError("ETHERSCAN_API_KEY is not configured")
    params = {
        "chainid": ETHERSCAN_CHAIN_ID,
        "module": "account",
        "action": action,
        "address": address,
        "page": 1,
        "offset": limit,
        "sort": "desc",
        "apikey": ETHERSCAN_API_KEY,
    }
    resp = requests.get(ETHERSCAN_URL, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    result = payload.get("result", [])
    if isinstance(result, str):
        return []
    if not isinstance(result, list):
        return []
    return result


def etherscan_fetch(address: str, limit: int) -> list[dict[str, Any]]:
    """
    获取更完整的交易集：
    - 普通交易(txlist)
    - 内部交易(txlistinternal)
    - ERC20 交易(tokentx)
    """
    normal_rows = etherscan_request("txlist", address, limit)
    internal_rows = etherscan_request("txlistinternal", address, limit)
    token_rows = etherscan_request("tokentx", address, limit)

    merged: list[dict[str, Any]] = []

    for row in normal_rows:
        merged.append(
            {
                "hash": row.get("hash", ""),
                "from": row.get("from", ""),
                "to": row.get("to", ""),
                "value": row.get("value", "0"),
                "gasPrice": row.get("gasPrice", "0"),
                "gasUsed": row.get("gasUsed", "0"),
                "timeStamp": row.get("timeStamp", "0"),
                "asset_symbol": "ETH",
                "asset_type": "native",
                "trace_id": row.get("transactionIndex", ""),
            }
        )

    for row in internal_rows:
        merged.append(
            {
                "hash": row.get("hash", ""),
                "from": row.get("from", ""),
                "to": row.get("to", ""),
                "value": row.get("value", "0"),
                "gasPrice": "0",
                "gasUsed": "0",
                "timeStamp": row.get("timeStamp", "0"),
                "asset_symbol": "ETH",
                "asset_type": "internal",
                "trace_id": row.get("traceId", ""),
            }
        )

    for row in token_rows:
        amount = parse_amount(row.get("value", "0"), row.get("tokenDecimal", "18"))
        merged.append(
            {
                "hash": row.get("hash", ""),
                "from": row.get("from", ""),
                "to": row.get("to", ""),
                "value": str(amount),
                "gasPrice": row.get("gasPrice", "0"),
                "gasUsed": row.get("gasUsed", "0"),
                "timeStamp": row.get("timeStamp", "0"),
                "asset_symbol": row.get("tokenSymbol", "TOKEN"),
                "asset_type": "erc20",
                "trace_id": row.get("logIndex", ""),
            }
        )

    # 去重：同类交易使用 hash + 类型 + 索引键
    seen = set()
    deduped = []
    for row in merged:
        key = f"{row.get('asset_type')}:{row.get('hash')}:{row.get('trace_id')}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    deduped.sort(key=lambda x: parse_int(x.get("timeStamp", "0")), reverse=True)
    return deduped


def latest_tx_hash(address: str) -> str | None:
    rows = etherscan_fetch(address, 1)
    if not rows:
        return None
    tx_hash = str(rows[0].get("hash", ""))
    return tx_hash or None


def compact_txs(txs: list[dict[str, Any]], target_address: str | None = None) -> list[dict[str, Any]]:
    output = []
    target = target_address.lower() if target_address else None
    for tx in txs:
        src = str(tx.get("from", "")).lower()
        dst = str(tx.get("to", "")).lower()
        asset_type = str(tx.get("asset_type", "native"))
        asset_symbol = str(tx.get("asset_symbol", "ETH") or "ETH").upper()
        raw_value = tx.get("value", "0")
        if asset_type == "erc20":
            try:
                amount = float(raw_value) if isinstance(raw_value, (int, float)) else float(str(raw_value or "0"))
            except Exception:
                amount = 0.0
            value_wei = "0"
        else:
            amount = parse_eth(raw_value)
            value_wei = raw_value
        ts = parse_int(tx.get("timeStamp", "0"), 0)
        direction = "other"
        counterparty = ""
        if target:
            if src == target:
                direction = "out"
                counterparty = dst
            elif dst == target:
                direction = "in"
                counterparty = src
        output.append(
            {
                "hash": tx.get("hash", ""),
                "from": src,
                "to": dst,
                "direction": direction,
                "counterparty": counterparty,
                "value_wei": value_wei,
                "value_eth": round(amount, 6),
                "gas_price_wei": str(tx.get("gasPrice", "0")),
                "gas_used": str(tx.get("gasUsed", "0")),
                "fee_eth": round(parse_int(tx.get("gasPrice", 0)) * parse_int(tx.get("gasUsed", 0)) / 1e18, 10),
                "asset_symbol": asset_symbol,
                "asset_type": asset_type,
                "time_stamp": str(ts),
                "time_text": datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if ts else "",
            }
        )
    return output


def build_graph(address: str, txs: list[dict[str, Any]], cluster_map: dict[str, str] | None = None) -> dict[str, Any]:
    core_color = "#0b5fff"
    nodes = [{"id": address, "name": address, "symbolSize": 56, "category": "core_address", "itemStyle": {"color": core_color}}]
    links = []
    seen = {address}
    color_map = {
        LABEL_HIGH_FREQ: "#ef4444",
        LABEL_HIGH_FEE: "#f59e0b",
        LABEL_HIGH_VOLUME: "#8b5cf6",
        LABEL_NORMAL: "#64748b",
    }

    for tx in txs:
        src = str(tx.get("from", "")).lower()
        dst = str(tx.get("to", "")).lower()
        if not src or not dst:
            continue

        if src not in seen:
            category = cluster_map.get(src, LABEL_NORMAL) if cluster_map else LABEL_NORMAL
            nodes.append(
                {
                    "id": src,
                    "name": src,
                    "symbolSize": 28,
                    "category": category,
                    "itemStyle": {"color": color_map.get(category, "#64748b")},
                }
            )
            seen.add(src)

        if dst not in seen:
            category = cluster_map.get(dst, LABEL_NORMAL) if cluster_map else LABEL_NORMAL
            nodes.append(
                {
                    "id": dst,
                    "name": dst,
                    "symbolSize": 28,
                    "category": category,
                    "itemStyle": {"color": color_map.get(category, "#64748b")},
                }
            )
            seen.add(dst)

        links.append({"source": src, "target": dst, "amount": tx.get("value_eth", 0), "tx_count": 1})

    categories = [
        {"name": "core_address", "itemStyle": {"color": core_color}},
        {"name": LABEL_HIGH_FREQ, "itemStyle": {"color": color_map[LABEL_HIGH_FREQ]}},
        {"name": LABEL_HIGH_FEE, "itemStyle": {"color": color_map[LABEL_HIGH_FEE]}},
        {"name": LABEL_HIGH_VOLUME, "itemStyle": {"color": color_map[LABEL_HIGH_VOLUME]}},
        {"name": LABEL_NORMAL, "itemStyle": {"color": color_map[LABEL_NORMAL]}},
    ]
    return {"nodes": nodes, "links": links, "categories": categories}

def detect_risk(address: str, txs: list[dict[str, Any]]) -> dict[str, Any]:
    incoming, outgoing = 0, 0
    counterparties = set()
    max_eth = 0.0
    total_amount = 0.0

    for tx in txs:
        f = str(tx.get("from", "")).lower()
        t = str(tx.get("to", "")).lower()
        asset_symbol = str(tx.get("asset_symbol", "ETH")).upper()
        amount = float(tx.get("value_eth", 0.0))
        if asset_symbol == "ETH":
            total_amount += amount

        if t == address:
            incoming += 1
            counterparties.add(f)
        elif f == address:
            outgoing += 1
            counterparties.add(t)

        if asset_symbol == "ETH":
            max_eth = max(max_eth, amount)

    score = 0
    alerts = []
    if len(txs) >= 40:
        score += 30
        alerts.append("high-frequency-transactions")
    if len(counterparties) >= 18:
        score += 25
        alerts.append("too-many-counterparties")
    if max_eth >= 200:
        score += 20
        alerts.append("large-single-transfer")
    if outgoing > incoming * 3 and outgoing >= 15:
        score += 15
        alerts.append("one-way-outflow")
    if address in BLACKLIST:
        score += 40
        alerts.append("blacklist-hit")
    if total_amount >= 1000:
        score += 10
        alerts.append("large-total-volume")

    score = min(100, score)
    level = "LOW" if score < 35 else ("MEDIUM" if score < 70 else "HIGH")

    return {
        "in_count": incoming,
        "out_count": outgoing,
        "counterparty_count": len(counterparties),
        "max_tx_eth": round(max_eth, 6),
        "total_amount_eth": round(total_amount, 6),
        "score": score,
        "level": level,
        "alerts": alerts,
    }

def cluster_addresses(address: str, txs: list[dict[str, Any]]) -> dict[str, Any]:
    peers: dict[str, dict[str, Any]] = {}
    for tx in txs:
        src = str(tx.get("from", "")).lower()
        dst = str(tx.get("to", "")).lower()
        if not src or not dst:
            continue
        if src != address and dst != address:
            continue

        peer = src if dst == address else dst
        if peer == address:
            continue

        if peer not in peers:
            peers[peer] = {
                "in_cnt": 0,
                "out_cnt": 0,
                "volume_eth": 0.0,
                "fees": [],
            }

        if dst == address:
            peers[peer]["in_cnt"] += 1
        else:
            peers[peer]["out_cnt"] += 1

        if str(tx.get("asset_symbol", "ETH")).upper() == "ETH":
            peers[peer]["volume_eth"] += float(tx.get("value_eth", 0))
        peers[peer]["fees"].append(float(tx.get("fee_eth", 0)))

    if len(peers) < 2:
        return {
            "method": "kmeans-feature-clustering",
            "target": address,
            "cluster_count": 0,
            "address_count": len(peers),
            "clusters": [],
            "peer_cluster_map": {},
            "note": "not-enough-counterparties-for-clustering",
        }

    peer_list = list(peers.keys())
    feature_rows = []
    for peer in peer_list:
        row = peers[peer]
        tx_count = row["in_cnt"] + row["out_cnt"]
        avg_fee = vec_mean(row["fees"]) if row["fees"] else 0.0
        total_fee = vec_sum(row["fees"]) if row["fees"] else 0.0
        fee_std = vec_std(row["fees"]) if row["fees"] else 0.0
        in_ratio = row["in_cnt"] / tx_count if tx_count else 0.0
        out_ratio = row["out_cnt"] / tx_count if tx_count else 0.0
        feature_rows.append([avg_fee, total_fee, tx_count, fee_std, in_ratio, out_ratio])

    labels: list[int] = []
    method = "kmeans-feature-clustering"

    if KMeans is not None and StandardScaler is not None and np is not None:
        mat = np.array(feature_rows, dtype=float)
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat)
        n_clusters = min(3, len(peer_list))
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = [int(x) for x in model.fit_predict(mat_scaled)]
    else:
        method = "rule-based-fallback-clustering"
        for row in feature_rows:
            avg_fee, _total_fee, tx_count, _fee_std, _in_ratio, _out_ratio = row
            if tx_count >= 15:
                labels.append(0)
            elif avg_fee > 0.001:
                labels.append(1)
            elif tx_count >= 8:
                labels.append(2)
            else:
                labels.append(3)

    grouped: dict[int, list[str]] = defaultdict(list)
    for peer, label in zip(peer_list, labels):
        grouped[int(label)].append(peer)

    peer_cluster_map: dict[str, str] = {}
    peer_cluster_id_map: dict[str, str] = {}
    clusters = []
    for idx, (label, members) in enumerate(sorted(grouped.items(), key=lambda x: x[0]), start=1):
        interaction_count = sum(peers[m]["in_cnt"] + peers[m]["out_cnt"] for m in members)
        interaction_volume = round(sum(peers[m]["volume_eth"] for m in members), 6)
        cluster_fees = [fee for m in members for fee in peers[m]["fees"]]
        avg_fee = vec_mean(cluster_fees) if cluster_fees else 0.0
        avg_tx_per_addr = interaction_count / len(members) if members else 0.0
        avg_volume_per_addr = interaction_volume / len(members) if members else 0.0

        if avg_tx_per_addr >= 8:
            cluster_name = LABEL_HIGH_FREQ
        elif avg_volume_per_addr > 20:
            cluster_name = LABEL_HIGH_VOLUME
        elif avg_fee > 0.001:
            cluster_name = LABEL_HIGH_FEE
        else:
            cluster_name = LABEL_NORMAL

        for member in members:
            peer_cluster_map[member] = cluster_name
            peer_cluster_id_map[member] = f"C{idx}"

        clusters.append(
            {
                "cluster_id": f"C{idx}",
                "raw_label": int(label),
                "label": cluster_name,
                "size": len(members),
                "interaction_count": interaction_count,
                "interaction_volume_eth": interaction_volume,
                "avg_fee_eth": round(avg_fee, 10),
                "avg_tx_per_address": round(avg_tx_per_addr, 4),
                "avg_volume_per_address_eth": round(avg_volume_per_addr, 6),
                "members": members,
            }
        )

    return {
        "method": method,
        "feature_columns": ["avg_fee_eth", "total_fee_eth", "tx_count", "fee_std_eth", "in_ratio", "out_ratio"],
        "target": address,
        "cluster_count": len(clusters),
        "address_count": len(peers),
        "clusters": clusters,
        "peer_cluster_map": peer_cluster_map,
        "peer_cluster_id_map": peer_cluster_id_map,
        "note": "" if method == "kmeans-feature-clustering" else "numpy-or-sklearn-missing-fallback-rule-clustering",
    }


def one_hop_trace(
    address: str,
    txs: list[dict[str, Any]],
    direction: str = "both",
    top_n: int = 20,
    min_tx_count: int = 1,
    sort_by: str = "activity",
) -> dict[str, Any]:
    # 对每个一跳对手聚合：保留流入/流出、净流、资产分布与时间区间，提升可解释性
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "in_amount_eth": 0.0,
            "out_amount_eth": 0.0,
            "in_count": 0,
            "out_count": 0,
            "first_time": "",
            "last_time": "",
            "assets": defaultdict(lambda: {"amount": 0.0, "count": 0, "in_amount": 0.0, "out_amount": 0.0}),
        }
    )

    for tx in txs:
        tx_dir = str(tx.get("direction", ""))
        if tx_dir not in {"in", "out"}:
            continue
        peer = str(tx.get("counterparty", "")).lower()
        if not peer:
            continue

        amount = abs(float(tx.get("value_eth", 0.0)))
        asset = str(tx.get("asset_symbol", "ETH")).upper()
        t = str(tx.get("time_text", ""))
        g = grouped[peer]

        if tx_dir == "in":
            g["in_count"] += 1
            if asset == "ETH":
                g["in_amount_eth"] += amount
            g["assets"][asset]["in_amount"] += amount
        else:
            g["out_count"] += 1
            if asset == "ETH":
                g["out_amount_eth"] += amount
            g["assets"][asset]["out_amount"] += amount

        g["assets"][asset]["amount"] += amount
        g["assets"][asset]["count"] += 1
        if t:
            if (not g["first_time"]) or t < g["first_time"]:
                g["first_time"] = t
            if (not g["last_time"]) or t > g["last_time"]:
                g["last_time"] = t

    rows: list[dict[str, Any]] = []
    for peer, g in grouped.items():
        total_count = int(g["in_count"] + g["out_count"])
        if total_count < max(1, int(min_tx_count)):
            continue

        in_eth = round(float(g["in_amount_eth"]), 6)
        out_eth = round(float(g["out_amount_eth"]), 6)
        gross_eth = round(in_eth + out_eth, 6)
        net_eth = round(in_eth - out_eth, 6)

        assets = []
        for asset, stats in g["assets"].items():
            assets.append(
                {
                    "asset": asset,
                    "amount": round(float(stats["amount"]), 6),
                    "in_amount": round(float(stats["in_amount"]), 6),
                    "out_amount": round(float(stats["out_amount"]), 6),
                    "count": int(stats["count"]),
                }
            )
        assets.sort(key=lambda x: x["count"], reverse=True)
        dominant_asset = assets[0]["asset"] if assets else "ETH"

        flags = []
        if g["out_count"] >= 8 and g["in_count"] == 0:
            flags.append("单向外流")
        if g["in_count"] >= 8 and g["out_count"] == 0:
            flags.append("单向流入")
        if total_count >= 10:
            flags.append("高频交互")
        if gross_eth >= 100:
            flags.append("高ETH流量")

        if direction == "forward":
            amount_focus = out_eth
            tx_count_focus = int(g["out_count"])
            from_addr = address
            to_addr = peer
        elif direction == "backward":
            amount_focus = in_eth
            tx_count_focus = int(g["in_count"])
            from_addr = peer
            to_addr = address
        else:  # both
            amount_focus = gross_eth
            tx_count_focus = total_count
            from_addr = address
            to_addr = peer

        if tx_count_focus <= 0:
            continue

        rows.append(
            {
                "peer": peer,
                "from": from_addr,
                "to": to_addr,
                "amount_eth": round(amount_focus, 6),
                "tx_count": tx_count_focus,
                "total_count": total_count,
                "in_count": int(g["in_count"]),
                "out_count": int(g["out_count"]),
                "in_amount_eth": in_eth,
                "out_amount_eth": out_eth,
                "gross_amount_eth": gross_eth,
                "net_amount_eth": net_eth,
                "dominant_asset": dominant_asset,
                "assets": assets[:8],
                "first_time": g["first_time"],
                "last_time": g["last_time"],
                "flags": flags,
            }
        )

    if sort_by == "eth_volume":
        rows.sort(key=lambda x: (x["gross_amount_eth"], x["total_count"]), reverse=True)
    elif sort_by == "net_flow":
        rows.sort(key=lambda x: abs(x["net_amount_eth"]), reverse=True)
    else:
        rows.sort(key=lambda x: (x["total_count"], x["gross_amount_eth"]), reverse=True)

    top_n = max(1, min(100, int(top_n)))
    rows = rows[:top_n]

    total_in_eth = round(sum(x["in_amount_eth"] for x in rows), 6)
    total_out_eth = round(sum(x["out_amount_eth"] for x in rows), 6)
    summary = {
        "peer_count": len(rows),
        "total_in_eth": total_in_eth,
        "total_out_eth": total_out_eth,
        "total_net_eth": round(total_in_eth - total_out_eth, 6),
        "total_tx_count": int(sum(x["total_count"] for x in rows)),
    }

    return {"direction": direction, "sort_by": sort_by, "top_n": top_n, "summary": summary, "rows": rows}


def build_trace_graph(
    root_address: str,
    direction: str,
    hops: int,
    per_address_limit: int,
    max_children: int,
    min_amount_eth: float,
) -> dict[str, Any]:
    # 多跳追踪：按层扩展一跳交易，构建可视化资金链路
    if direction not in {"forward", "backward"}:
        raise ValidationError("trace graph direction must be forward/backward")
    hops = max(1, min(4, int(hops)))
    per_address_limit = max(20, min(100, int(per_address_limit)))
    max_children = max(2, min(20, int(max_children)))
    min_amount_eth = max(0.0, float(min_amount_eth))

    tx_cache: dict[str, list[dict[str, Any]]] = {}

    def get_txs(addr: str) -> list[dict[str, Any]]:
        key = addr.lower()
        if key not in tx_cache:
            tx_cache[key] = compact_txs(etherscan_fetch(key, per_address_limit), key)
        return tx_cache[key]

    nodes_map: dict[str, dict[str, Any]] = {
        root_address: {"id": root_address, "name": root_address, "level": 0, "role": "root"}
    }
    links_map: dict[tuple[str, str], dict[str, Any]] = {}

    frontier = [root_address]
    expanded: set[str] = set()
    max_total_nodes = 120

    for depth in range(1, hops + 1):
        if not frontier:
            break
        next_frontier: list[str] = []
        for current in frontier:
            current = current.lower()
            if current in expanded:
                continue
            expanded.add(current)

            grouped: dict[str, dict[str, Any]] = defaultdict(
                lambda: {"amount_eth": 0.0, "count": 0, "first_time": "", "last_time": ""}
            )
            for tx in get_txs(current):
                tx_dir = str(tx.get("direction", ""))
                peer = str(tx.get("counterparty", "")).lower()
                if not peer:
                    continue
                amount = abs(float(tx.get("value_eth", 0.0)))
                if amount <= 0:
                    continue
                t = str(tx.get("time_text", ""))

                if direction == "forward":
                    if tx_dir != "out":
                        continue
                else:  # backward
                    if tx_dir != "in":
                        continue

                g = grouped[peer]
                g["amount_eth"] += amount
                g["count"] += 1
                if t:
                    if (not g["first_time"]) or t < g["first_time"]:
                        g["first_time"] = t
                    if (not g["last_time"]) or t > g["last_time"]:
                        g["last_time"] = t

            candidates = []
            for peer, g in grouped.items():
                amount = round(float(g["amount_eth"]), 6)
                if amount < min_amount_eth:
                    continue
                candidates.append((peer, amount, int(g["count"]), g["first_time"], g["last_time"]))

            candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
            candidates = candidates[:max_children]

            for peer, amount, count, first_time, last_time in candidates:
                if direction == "forward":
                    source, target = current, peer
                else:
                    source, target = peer, current

                key = (source, target)
                if key not in links_map:
                    links_map[key] = {
                        "source": source,
                        "target": target,
                        "amount_eth": 0.0,
                        "tx_count": 0,
                        "first_time": first_time,
                        "last_time": last_time,
                        "hop": depth,
                    }
                links_map[key]["amount_eth"] = round(float(links_map[key]["amount_eth"]) + amount, 6)
                links_map[key]["tx_count"] = int(links_map[key]["tx_count"]) + count
                if first_time and ((not links_map[key]["first_time"]) or first_time < links_map[key]["first_time"]):
                    links_map[key]["first_time"] = first_time
                if last_time and ((not links_map[key]["last_time"]) or last_time > links_map[key]["last_time"]):
                    links_map[key]["last_time"] = last_time

                if peer not in nodes_map:
                    nodes_map[peer] = {
                        "id": peer,
                        "name": peer,
                        "level": depth,
                        "role": "peer",
                    }
                else:
                    nodes_map[peer]["level"] = min(int(nodes_map[peer].get("level", depth)), depth)

                if len(nodes_map) < max_total_nodes:
                    next_frontier.append(peer)

        # 控制下一层规模，避免API爆炸
        uniq = []
        seen = set()
        for a in next_frontier:
            if a in seen:
                continue
            seen.add(a)
            uniq.append(a)
        frontier = uniq[: max_children * 3]

    if len(nodes_map) <= 1:
        return {
            "direction": direction,
            "hops": hops,
            "nodes": [],
            "links": [],
            "summary": {"node_count": 0, "edge_count": 0, "total_amount_eth": 0.0, "total_tx_count": 0},
        }

    # 生成可视化节点/边
    nodes = []
    for node_id, n in nodes_map.items():
        level = int(n.get("level", 0))
        role = str(n.get("role", "peer"))
        symbol_size = 56 if role == "root" else max(24, 46 - level * 6)
        color = "#0b5fff" if role == "root" else ("#f59e0b" if direction == "forward" else "#10b981")
        nodes.append(
            {
                "id": node_id,
                "name": node_id,
                "category": f"L{level}",
                "symbolSize": symbol_size,
                "itemStyle": {"color": color},
                "value": level,
            }
        )

    links = []
    total_amount = 0.0
    total_tx_count = 0
    for link in links_map.values():
        total_amount += float(link["amount_eth"])
        total_tx_count += int(link["tx_count"])
        links.append(
            {
                "source": link["source"],
                "target": link["target"],
                "value": round(float(link["amount_eth"]), 6),
                "amount_eth": round(float(link["amount_eth"]), 6),
                "tx_count": int(link["tx_count"]),
                "hop": int(link["hop"]),
                "first_time": link["first_time"],
                "last_time": link["last_time"],
                "lineStyle": {"width": max(1, min(8, int(float(link["amount_eth"]) ** 0.5) + 1))},
            }
        )

    categories = [{"name": f"L{i}"} for i in range(0, hops + 1)]
    return {
        "direction": direction,
        "hops": hops,
        "nodes": nodes,
        "links": links,
        "categories": categories,
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(links),
            "total_amount_eth": round(total_amount, 6),
            "total_tx_count": int(total_tx_count),
        },
    }


def build_address_profile(address: str, txs: list[dict[str, Any]]) -> dict[str, Any]:
    if not txs:
        raise ValidationError("no-transaction-data")

    counterparties = {tx.get("counterparty") for tx in txs if tx.get("counterparty")}
    in_amount = sum(
        float(tx.get("value_eth", 0.0))
        for tx in txs
        if tx.get("direction") == "in" and str(tx.get("asset_symbol", "ETH")).upper() == "ETH"
    )
    out_amount = sum(
        float(tx.get("value_eth", 0.0))
        for tx in txs
        if tx.get("direction") == "out" and str(tx.get("asset_symbol", "ETH")).upper() == "ETH"
    )
    total_amount = in_amount + out_amount
    total_fee = sum(float(tx.get("fee_eth", 0.0)) for tx in txs)
    active_days = len({(tx.get("time_text") or "")[:10] for tx in txs if tx.get("time_text")})
    risk = detect_risk(address, txs)

    asset_counter: dict[str, int] = defaultdict(int)
    for tx in txs:
        asset = str(tx.get("asset_symbol", "ETH")).upper()
        asset_counter[asset] += 1
    asset_mix = [{"asset": k, "tx_count": v} for k, v in sorted(asset_counter.items(), key=lambda kv: kv[1], reverse=True)]

    tags = []
    if risk["counterparty_count"] > 30:
        tags.append("active-network")
    if out_amount > in_amount * 2 and risk["out_count"] >= 10:
        tags.append("outflow-heavy")
    if total_amount > 500:
        tags.append("high-volume")
    if not tags:
        tags.append("normal-account")

    return {
        "address": address,
        "total_transactions": len(txs),
        "in_amount_eth": round(in_amount, 6),
        "out_amount_eth": round(out_amount, 6),
        "total_amount_eth": round(total_amount, 6),
        "avg_amount_eth": round(total_amount / len(txs), 6),
        "total_fee_eth": round(total_fee, 8),
        "unique_counterparties": len(counterparties),
        "active_days": active_days,
        "risk_score": risk["score"],
        "risk_level": risk["level"],
        "risk_alerts": risk["alerts"],
        "tags": tags,
        "asset_mix": asset_mix,
    }


def build_time_series(txs: list[dict[str, Any]]) -> dict[str, Any]:
    # series: 默认统计 ETH，且区分流入/流出/净流，避免“总额全为正”导致信息丢失
    bucket: dict[str, dict[str, float]] = defaultdict(
        lambda: {"in_amount": 0.0, "out_amount": 0.0, "gross_amount": 0.0, "count": 0.0}
    )
    asset_bucket: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"in_amount": 0.0, "out_amount": 0.0, "gross_amount": 0.0, "count": 0.0})
    )
    for tx in txs:
        t = tx.get("time_text", "")
        date_key = t[:10] if t else ""
        if not date_key:
            continue
        asset = str(tx.get("asset_symbol", "ETH")).upper()
        direction = str(tx.get("direction", "")).lower()
        amount = abs(float(tx.get("value_eth", 0.0)))
        if amount <= 0:
            continue

        target_bucket = asset_bucket[asset][date_key]
        if direction == "in":
            target_bucket["in_amount"] += amount
        elif direction == "out":
            target_bucket["out_amount"] += amount
        else:
            # 仅统计与目标地址相关的in/out，other不纳入时序
            continue
        target_bucket["gross_amount"] += amount
        target_bucket["count"] += 1

        if asset == "ETH":
            if direction == "in":
                bucket[date_key]["in_amount"] += amount
            elif direction == "out":
                bucket[date_key]["out_amount"] += amount
            bucket[date_key]["gross_amount"] += amount
            bucket[date_key]["count"] += 1

    sorted_days = sorted(bucket.items())
    series = [{"date": d, "amount": round(v["gross_amount"], 6), "count": int(v["count"])} for d, v in sorted_days]
    series_in_eth = [{"date": d, "amount": round(v["in_amount"], 6)} for d, v in sorted_days]
    series_out_eth = [{"date": d, "amount": round(v["out_amount"], 6)} for d, v in sorted_days]
    series_net_eth = [{"date": d, "amount": round(v["in_amount"] - v["out_amount"], 6)} for d, v in sorted_days]
    series_tx_count = [{"date": d, "count": int(v["count"])} for d, v in sorted_days]

    amounts = [x["amount"] for x in series]
    if not amounts:
        return {
            "series": [],
            "series_in_eth": [],
            "series_out_eth": [],
            "series_net_eth": [],
            "series_tx_count": [],
            "anomalies": [],
            "anomalies_net_eth": [],
            "series_by_asset": {},
            "series_unit": "ETH-only",
        }

    mean = vec_mean(amounts)
    std = vec_std(amounts)
    upper = mean + 3 * std
    lower = mean - 3 * std
    anomalies = [item for item in series if item["amount"] > upper or item["amount"] < lower]

    net_amounts = [x["amount"] for x in series_net_eth]
    net_mean = vec_mean(net_amounts)
    net_std = vec_std(net_amounts)
    net_upper = net_mean + 3 * net_std
    net_lower = net_mean - 3 * net_std
    anomalies_net_eth = [item for item in series_net_eth if item["amount"] > net_upper or item["amount"] < net_lower]

    series_by_asset: dict[str, list[dict[str, Any]]] = {}
    for asset, per_day in asset_bucket.items():
        series_by_asset[asset] = [
            {
                "date": d,
                # amount保留为兼容字段（历史前端/报表仍可读取）
                "amount": round(v["gross_amount"], 6),
                "in_amount": round(v["in_amount"], 6),
                "out_amount": round(v["out_amount"], 6),
                "net_amount": round(v["in_amount"] - v["out_amount"], 6),
                "gross_amount": round(v["gross_amount"], 6),
                "count": int(v["count"]),
            }
            for d, v in sorted(per_day.items())
        ]
    return {
        # 兼容旧前端：series.amount = ETH每日总流量（|in|+|out|）
        "series": series,
        "series_in_eth": series_in_eth,
        "series_out_eth": series_out_eth,
        "series_net_eth": series_net_eth,
        "series_tx_count": series_tx_count,
        "anomalies": anomalies,
        "anomalies_net_eth": anomalies_net_eth,
        "series_by_asset": series_by_asset,
        "series_unit": "ETH-only",
        "note": "series为ETH总流量（in+out），series_net_eth为ETH净流量（in-out）；ERC20分币种结果见series_by_asset",
    }


def get_contract(readonly: bool = True):
    if Web3 is None:
        raise RuntimeError("web3 is not installed")
    if not CONTRACT_ADDRESS:
        raise RuntimeError("CONTRACT_ADDRESS is not configured")
    web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
    if not web3.is_connected():
        raise RuntimeError("unable to connect WEB3_PROVIDER_URI")
    contract = web3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=CONTRACT_ABI)
    if readonly:
        return web3, contract, None
    if not SERVER_PRIVATE_KEY:
        raise RuntimeError("SERVER_PRIVATE_KEY is not configured")
    account = web3.eth.account.from_key(SERVER_PRIVATE_KEY)
    return web3, contract, account


def read_chain_evidences(address: str) -> list[dict[str, Any]]:
    web3, contract, _ = get_contract(readonly=True)
    rows = contract.functions.getEvidences(Web3.to_checksum_address(address)).call()
    out = []
    for r in rows:
        out.append(
            {
                "target": r[0].lower(),
                "hash": r[1],
                "timestamp": int(r[2]),
                "datetime": datetime.fromtimestamp(int(r[2]), tz=timezone.utc).isoformat(timespec="seconds"),
            }
        )
    return out


def store_evidence_onchain(address: str, report_hash: str) -> str:
    web3, contract, account = get_contract(readonly=False)
    target = Web3.to_checksum_address(address)
    sender = account.address
    nonce = web3.eth.get_transaction_count(sender)

    tx_data = contract.functions.storeEvidence(target, report_hash).build_transaction(
        {
            "from": sender,
            "nonce": nonce,
            "chainId": web3.eth.chain_id,
            "gasPrice": web3.eth.gas_price,
        }
    )

    if "gas" not in tx_data:
        try:
            tx_data["gas"] = web3.eth.estimate_gas(tx_data) + 20000
        except Exception:
            tx_data["gas"] = 250000

    signed = web3.eth.account.sign_transaction(tx_data, private_key=SERVER_PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    if int(receipt.status) != 1:
        raise RuntimeError("on-chain transaction reverted")
    return tx_hash.hex()


def evidence_from_tx_hash(tx_hash: str) -> dict[str, Any]:
    if Web3 is None:
        raise RuntimeError("web3 is not installed")
    if not CONTRACT_ADDRESS:
        raise RuntimeError("CONTRACT_ADDRESS is not configured")

    tx_hash = str(tx_hash).strip()
    if not re.fullmatch(r"0x[a-fA-F0-9]{64}", tx_hash):
        raise ValidationError("tx_hash must be 0x + 64 hex chars")

    web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
    if not web3.is_connected():
        raise RuntimeError("unable to connect WEB3_PROVIDER_URI")

    contract_addr = Web3.to_checksum_address(CONTRACT_ADDRESS)
    contract = web3.eth.contract(address=contract_addr, abi=CONTRACT_ABI)
    try:
        tx = web3.eth.get_transaction(Web3.to_hex(Web3.to_bytes(hexstr=tx_hash)))
    except Exception:
        raise ValidationError("tx hash not found on current chain")

    if not tx.get("to") or Web3.to_checksum_address(tx["to"]) != contract_addr:
        raise ValidationError("tx is not sent to TraceCoin contract")
    if not tx.get("input") or tx["input"] == "0x":
        raise ValidationError("tx input has no contract call data")

    fn, args = contract.decode_function_input(tx["input"])
    if fn.fn_name != "storeEvidence":
        raise ValidationError("tx is not a storeEvidence call")

    return {
        "tx_hash": Web3.to_hex(tx["hash"]),
        "block_number": int(tx["blockNumber"]) if tx.get("blockNumber") is not None else None,
        "target": str(args.get("_target", "")).lower(),
        "report_hash": str(args.get("_hash", "")).lower(),
        "from": str(tx.get("from", "")).lower(),
        "to": str(tx.get("to", "")).lower(),
    }


def evidence_from_block_number(block_number: int) -> dict[str, Any]:
    if Web3 is None:
        raise RuntimeError("web3 is not installed")
    if not CONTRACT_ADDRESS:
        raise RuntimeError("CONTRACT_ADDRESS is not configured")

    web3 = Web3(Web3.HTTPProvider(WEB3_PROVIDER_URI))
    if not web3.is_connected():
        raise RuntimeError("unable to connect WEB3_PROVIDER_URI")

    contract_addr = Web3.to_checksum_address(CONTRACT_ADDRESS)
    contract = web3.eth.contract(address=contract_addr, abi=CONTRACT_ABI)
    block = web3.eth.get_block(block_number, full_transactions=True)

    rows = []
    for tx in block["transactions"]:
        to_addr = tx.get("to")
        if not to_addr:
            continue
        if Web3.to_checksum_address(to_addr) != contract_addr:
            continue
        if not tx.get("input") or tx["input"] == "0x":
            continue
        try:
            fn, args = contract.decode_function_input(tx["input"])
        except Exception:
            continue
        if fn.fn_name != "storeEvidence":
            continue
        rows.append(
            {
                "tx_hash": Web3.to_hex(tx["hash"]),
                "target": str(args.get("_target", "")).lower(),
                "report_hash": str(args.get("_hash", "")).lower(),
                "from": str(tx.get("from", "")).lower(),
                "to": str(to_addr).lower(),
            }
        )

    return {"block_number": int(block_number), "count": len(rows), "records": rows}


def save_evidence_record(
    address: str,
    report_hash: str,
    tx_count: int = 0,
    risk_score: int = 0,
    risk_level: str = "LOW",
    chain_tx_hash: str | None = None,
    note: str = "",
    chain_status: str | None = None,
    created_at: str | None = None,
) -> tuple[int, bool]:
    chain_tx_hash = (str(chain_tx_hash).strip() or None) if chain_tx_hash is not None else None
    chain_status = chain_status or ("SUCCESS" if chain_tx_hash else "SKIPPED")
    created_at = created_at or utc_now_iso()
    risk_level = (risk_level or "LOW").upper()

    conn = db_conn()
    try:
        existing = conn.execute(
            """
            SELECT id, chain_tx_hash FROM evidence_records
            WHERE address = ? AND report_hash = ?
            ORDER BY id DESC LIMIT 1
            """,
            (address, report_hash),
        ).fetchone()
        if existing:
            existing_id = int(existing["id"])
            existing_chain_tx = existing["chain_tx_hash"]
            if chain_tx_hash and not existing_chain_tx:
                conn.execute(
                    """
                    UPDATE evidence_records
                    SET chain_status = ?, chain_tx_hash = ?, note = ?, tx_count = ?, risk_score = ?, risk_level = ?
                    WHERE id = ?
                    """,
                    (
                        "SUCCESS",
                        chain_tx_hash,
                        note or "pdf-report",
                        tx_count,
                        risk_score,
                        risk_level,
                        existing_id,
                    ),
                )
                conn.commit()
            return existing_id, False

        cur = conn.execute(
            """
            INSERT INTO evidence_records
            (address, report_hash, tx_count, risk_score, risk_level, chain_status, chain_tx_hash, note, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                address,
                report_hash,
                tx_count,
                risk_score,
                risk_level,
                chain_status,
                chain_tx_hash,
                note,
                created_at,
            ),
        )
        conn.commit()
        return int(cur.lastrowid), True
    finally:
        conn.close()


def sync_chain_history_to_db(address: str, chain_rows: list[dict[str, Any]]) -> int:
    imported = 0
    for row in chain_rows:
        report_hash = str(row.get("hash", "")).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", report_hash):
            continue
        _, created = save_evidence_record(
            address=address,
            report_hash=report_hash,
            tx_count=0,
            risk_score=0,
            risk_level="UNKNOWN",
            chain_tx_hash=None,
            note="synced-from-chain",
            chain_status="CHAIN_ONLY",
            created_at=str(row.get("datetime") or utc_now_iso()),
        )
        if created:
            imported += 1
    return imported


@app.errorhandler(ValidationError)
def on_validation_error(err: ValidationError):
    return jsonify({"error": str(err)}), 400


@app.errorhandler(RuntimeError)
def on_runtime_error(err: RuntimeError):
    return jsonify({"error": str(err)}), 400


@app.errorhandler(requests.RequestException)
def on_requests_error(err: requests.RequestException):
    return jsonify({"error": f"upstream-request-failed: {str(err)}"}), 502

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/page/<name>")
def page(name: str):
    filename = f"{name}.html"
    if filename not in UI_PAGES:
        abort(404)
    return send_from_directory(".", filename)


@app.route("/assets/font/cjk", methods=["GET"])
def cjk_font():
    env_path = str(os.getenv("TRACECOIN_PDF_FONT_PATH", "")).strip()
    candidates = list(PDF_FONT_CANDIDATES)
    if env_path:
        candidates.insert(0, Path(env_path))

    for path in candidates:
        try:
            if path and path.exists() and path.is_file():
                return send_file(path, mimetype="font/ttf", conditional=True, max_age=86400)
        except Exception:
            continue

    return jsonify({"error": "cjk font file not found"}), 404


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify(
        {
            "ok": True,
            "time": utc_now_iso(),
            "has_etherscan_key": bool(ETHERSCAN_API_KEY),
            "has_contract": bool(CONTRACT_ADDRESS),
        }
    )


@app.route("/api/query", methods=["POST"])
def query_transactions():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    limit = parse_limit(body.get("limit", DEFAULT_TX_LIMIT))

    txs = compact_txs(etherscan_fetch(address, limit), address)
    if not txs:
        return jsonify({"error": "no-transactions-found"}), 404

    counterparty_fees: dict[str, list[float]] = defaultdict(list)
    for tx in txs:
        cp = tx.get("counterparty")
        if cp:
            counterparty_fees[cp].append(float(tx.get("fee_eth", 0.0)))

    cache["address"] = address
    cache["transactions"] = txs
    cache["counterparties"] = {
        cp: {
            "avg_fee": vec_mean(fees),
            "total_fee": vec_sum(fees),
            "tx_count": len(fees),
            "fee_std": vec_std(fees),
        }
        for cp, fees in counterparty_fees.items()
    }
    cache["latest_tx_hash"] = latest_tx_hash(address)

    clustering = cluster_addresses(address, txs)
    cache["cluster_labels"] = clustering.get("peer_cluster_map", {})
    write_audit("query", address, {"tx_count": len(txs), "counterparty_count": len(counterparty_fees)})

    return jsonify(
        {
            "address": address,
            "transactions": txs,
            "counterparty_count": len(counterparty_fees),
            "cluster_available": clustering["cluster_count"] > 0,
        }
    )


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    limit = parse_limit(body.get("limit", DEFAULT_TX_LIMIT))

    txs = compact_txs(etherscan_fetch(address, limit), address)
    risk = detect_risk(address, txs)
    clustering = cluster_addresses(address, txs)
    graph = build_graph(address, txs, clustering["peer_cluster_map"])

    report_hash = canonical_hash({"address": address, "risk": risk, "clustering": clustering, "txs": txs})
    write_audit("analyze", address, {"limit": limit, "risk": risk, "cluster_count": clustering["cluster_count"]})

    return jsonify(
        {
            "address": address,
            "tx_count": len(txs),
            "risk": risk,
            "clustering": clustering,
            "graph": graph,
            "txs": txs,
            "report_hash": report_hash,
        }
    )


@app.route("/api/batch", methods=["POST"])
def batch_query():
    body = request.get_json(silent=True) or {}
    addresses = body.get("addresses", [])
    if not isinstance(addresses, list):
        raise ValidationError("addresses must be an array")

    addresses = [normalize_address(a) for a in addresses if str(a).strip()]
    if not addresses:
        raise ValidationError("address list cannot be empty")
    if len(addresses) > 5:
        raise ValidationError("max 5 addresses in batch")

    all_txs = []
    cp_sets = []
    for address in addresses:
        txs = compact_txs(etherscan_fetch(address, min(DEFAULT_TX_LIMIT, 50)), address)
        cp_sets.append({tx["counterparty"] for tx in txs if tx.get("counterparty")})
        for tx in txs:
            tx["source_address"] = address
        all_txs.extend(txs)

    common_counterparties = sorted(list(set.intersection(*cp_sets))) if cp_sets else []
    write_audit("batch_query", None, {"address_count": len(addresses), "tx_count": len(all_txs)})
    return jsonify({"transactions": all_txs, "common_counterparties": common_counterparties})


@app.route("/api/trace", methods=["POST"])
def trace_api():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    direction = str(body.get("direction", "both")).strip()
    if direction not in {"forward", "backward", "both"}:
        raise ValidationError("direction must be forward/backward/both")

    sort_by = str(body.get("sort_by", "activity")).strip()
    if sort_by not in {"activity", "eth_volume", "net_flow"}:
        raise ValidationError("sort_by must be activity/eth_volume/net_flow")

    top_n = parse_int(body.get("top_n", 20), 20)
    min_tx_count = parse_int(body.get("min_tx_count", 1), 1)

    limit = parse_limit(body.get("limit", min(DEFAULT_TX_LIMIT, 100)))
    txs = compact_txs(etherscan_fetch(address, limit), address)
    trace_result = one_hop_trace(address, txs, direction, top_n=top_n, min_tx_count=min_tx_count, sort_by=sort_by)

    # 向后兼容旧前端字段
    trace_result["paths"] = [
        {
            "from": row["from"],
            "to": row["to"],
            "amount_eth": row["amount_eth"],
            "tx_count": row["tx_count"],
        }
        for row in trace_result["rows"]
    ]
    return jsonify(trace_result)


@app.route("/api/profile", methods=["POST"])
def profile_api():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    txs = compact_txs(etherscan_fetch(address, min(DEFAULT_TX_LIMIT, 100)), address)
    profile = build_address_profile(address, txs)
    write_audit("profile", address, {"risk_score": profile["risk_score"]})
    return jsonify(profile)


@app.route("/api/timeseries", methods=["POST"])
def timeseries_api():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    txs = compact_txs(etherscan_fetch(address, min(DEFAULT_TX_LIMIT, 100)), address)
    return jsonify(build_time_series(txs))


@app.route("/api/graph", methods=["POST"])
def graph_api():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    txs = compact_txs(etherscan_fetch(address, min(DEFAULT_TX_LIMIT, 100)), address)
    clustering = cluster_addresses(address, txs)
    return jsonify(build_graph(address, txs, clustering["peer_cluster_map"]))


@app.route("/api/cluster", methods=["POST", "GET"])
def cluster_api():
    if request.method == "GET":
        if not cache["address"] or not cache["transactions"]:
            return jsonify({"error": "query-address-first"}), 400
        return jsonify(cluster_addresses(cache["address"], cache["transactions"]))

    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    limit = parse_limit(body.get("limit", DEFAULT_TX_LIMIT))
    txs = compact_txs(etherscan_fetch(address, limit), address)
    clustering = cluster_addresses(address, txs)
    write_audit("cluster", address, {"limit": limit, "cluster_count": clustering["cluster_count"]})
    return jsonify(clustering)


@app.route("/api/latest", methods=["GET"])
def latest_api():
    raw = request.args.get("address", "")
    if not raw:
        return jsonify({"latest_hash": None, "changed": False})

    address = normalize_address(raw)
    if cache["address"] != address:
        return jsonify({"latest_hash": None, "changed": False})

    current_hash = latest_tx_hash(address)
    changed = bool(current_hash and cache["latest_tx_hash"] != current_hash)
    cache["latest_tx_hash"] = current_hash
    return jsonify({"latest_hash": current_hash, "changed": changed})


@app.route("/api/evidence/store_onchain", methods=["POST"])
def evidence_store_onchain_api():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    report_hash = str(body.get("report_hash", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", report_hash):
        raise ValidationError("report_hash must be 64-char sha256 hex")

    tx_hash = store_evidence_onchain(address, report_hash)
    auto_register = parse_bool(body.get("auto_register", True), True)
    record_id = None
    record_created = False
    if auto_register:
        record_id, record_created = save_evidence_record(
            address=address,
            report_hash=report_hash,
            tx_count=parse_int(body.get("tx_count", 0), 0),
            risk_score=parse_int(body.get("risk_score", 0), 0),
            risk_level=str(body.get("risk_level", "LOW")),
            chain_tx_hash=tx_hash,
            note=str(body.get("note", "pdf-report")).strip(),
            chain_status="SUCCESS",
        )
    write_audit("store_onchain", address, {"report_hash": report_hash, "tx_hash": tx_hash})
    return jsonify(
        {
            "address": address,
            "report_hash": report_hash,
            "tx_hash": tx_hash,
            "record_id": record_id,
            "record_created": record_created,
            "auto_register": auto_register,
        }
    )


@app.route("/api/evidence/from_tx", methods=["POST"])
def evidence_from_tx_api():
    body = request.get_json(silent=True) or {}
    tx_hash = str(body.get("tx_hash", "")).strip()
    result = evidence_from_tx_hash(tx_hash)
    write_audit("evidence_from_tx", result.get("target"), {"tx_hash": tx_hash})
    return jsonify(result)


@app.route("/api/evidence/from_block", methods=["POST"])
def evidence_from_block_api():
    body = request.get_json(silent=True) or {}
    block_number = parse_int(body.get("block_number", -1), -1)
    if block_number < 0:
        raise ValidationError("block_number must be a non-negative integer")
    result = evidence_from_block_number(block_number)
    write_audit("evidence_from_block", None, {"block_number": block_number, "count": result.get("count", 0)})
    return jsonify(result)


@app.route("/api/evidence/chain/<address>", methods=["GET"])
def evidence_chain_by_address(address: str):
    address = normalize_address(address)
    rows = read_chain_evidences(address)
    return jsonify({"address": address, "onchain_records": rows, "count": len(rows)})


@app.route("/api/evidence/register", methods=["POST"])
def register_evidence():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    report_hash = str(body.get("report_hash", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", report_hash):
        raise ValidationError("report_hash must be 64-char sha256 hex")

    tx_count = parse_int(body.get("tx_count", 0), 0)
    risk_score = parse_int(body.get("risk_score", 0), 0)
    risk_level = str(body.get("risk_level", "LOW")).upper()
    chain_tx_hash = body.get("chain_tx_hash") or None

    evidence_id, created = save_evidence_record(
        address=address,
        report_hash=report_hash,
        tx_count=tx_count,
        risk_score=risk_score,
        risk_level=risk_level,
        chain_tx_hash=chain_tx_hash,
        note=str(body.get("note", "")).strip(),
        chain_status="SUCCESS" if chain_tx_hash else "SKIPPED",
    )

    write_audit("register_evidence", address, {"id": evidence_id, "created": created})
    return jsonify(
        {
            "id": evidence_id,
            "address": address,
            "report_hash": report_hash,
            "chain_tx_hash": chain_tx_hash,
            "created": created,
        }
    )


@app.route("/api/evidence/verify", methods=["POST"])
def verify_evidence():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    report_hash = str(body.get("report_hash", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", report_hash):
        raise ValidationError("report_hash must be 64-char sha256 hex")

    conn = db_conn()
    try:
        row = conn.execute(
            """
            SELECT id, created_at, risk_level, chain_status, chain_tx_hash
            FROM evidence_records
            WHERE address = ? AND report_hash = ?
            ORDER BY id DESC LIMIT 1
            """,
            (address, report_hash),
        ).fetchone()
    finally:
        conn.close()

    chain_match = False
    chain_record = None
    chain_error = None
    try:
        for r in read_chain_evidences(address):
            if r["hash"] == report_hash:
                chain_match = True
                chain_record = r
                break
    except Exception as e:
        chain_error = str(e)

    result = {
        "address": address,
        "report_hash": report_hash,
        "db_match": bool(row),
        "chain_match": chain_match,
        "is_valid": bool(row) or chain_match,
        "record": dict(row) if row else None,
        "chain_record": chain_record,
        "chain_error": chain_error,
    }
    write_audit("verify_evidence", address, {"is_valid": result["is_valid"]})
    return jsonify(result)


@app.route("/api/evidence/history/<address>", methods=["GET"])
def history(address: str):
    address = normalize_address(address)
    include_chain = parse_bool(request.args.get("include_chain", "1"), True)
    synced_count = 0
    chain_error = None
    if include_chain:
        try:
            chain_rows = read_chain_evidences(address)
            synced_count = sync_chain_history_to_db(address, chain_rows)
        except Exception as e:
            chain_error = str(e)

    conn = db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, address, report_hash, tx_count, risk_score, risk_level, chain_status, chain_tx_hash, note, created_at
            FROM evidence_records WHERE address = ? ORDER BY id DESC LIMIT 100
            """,
            (address,),
        ).fetchall()
    finally:
        conn.close()
    return jsonify(
        {
            "address": address,
            "history": [dict(r) for r in rows],
            "synced_from_chain": synced_count,
            "chain_error": chain_error,
        }
    )


init_db()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
