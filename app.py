import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np

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


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def etherscan_fetch(address: str, limit: int) -> list[dict[str, Any]]:
    if not ETHERSCAN_API_KEY:
        raise RuntimeError("ETHERSCAN_API_KEY is not configured")
    params = {
        "chainid": ETHERSCAN_CHAIN_ID,
        "module": "account",
        "action": "txlist",
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
    return result


def compact_txs(txs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for tx in txs:
        value_wei = tx.get("value", "0")
        output.append(
            {
                "hash": tx.get("hash", ""),
                "from": str(tx.get("from", "")).lower(),
                "to": str(tx.get("to", "")).lower(),
                "value_wei": value_wei,
                "value_eth": round(parse_eth(value_wei), 6),
                "gas_price_wei": str(tx.get("gasPrice", "0")),
                "gas_used": str(tx.get("gasUsed", "0")),
                "fee_eth": round(parse_int(tx.get("gasPrice", 0)) * parse_int(tx.get("gasUsed", 0)) / 1e18, 10),
                "time_stamp": tx.get("timeStamp", "0"),
            }
        )
    return output


def build_graph(address: str, txs: list[dict[str, Any]], cluster_map: dict[str, str] | None = None) -> dict[str, Any]:
    nodes = [{"id": address, "name": address, "symbolSize": 56, "itemStyle": {"color": "#0b5fff"}}]
    links = []
    seen = {address}
    color_map = {
        "高频交互簇": "#ef4444",
        "高费率交互簇": "#f59e0b",
        "大额流转簇": "#8b5cf6",
        "一般交互簇": "#64748b",
    }

    for tx in txs:
        src = str(tx.get("from", "")).lower()
        dst = str(tx.get("to", "")).lower()
        if not src or not dst:
            continue
        if src not in seen:
            style = {}
            if cluster_map and src in cluster_map:
                style = {"color": color_map.get(cluster_map[src], "#64748b")}
            nodes.append({"id": src, "name": src, "symbolSize": 28, "itemStyle": style})
            seen.add(src)
        if dst not in seen:
            style = {}
            if cluster_map and dst in cluster_map:
                style = {"color": color_map.get(cluster_map[dst], "#64748b")}
            nodes.append({"id": dst, "name": dst, "symbolSize": 28, "itemStyle": style})
            seen.add(dst)
        links.append({"source": src, "target": dst})
    return {"nodes": nodes, "links": links}


def detect_risk(address: str, txs: list[dict[str, Any]]) -> dict[str, Any]:
    incoming, outgoing = 0, 0
    counterparties = set()
    max_eth = 0.0

    for tx in txs:
        f = str(tx.get("from", "")).lower()
        t = str(tx.get("to", "")).lower()
        if t == address:
            incoming += 1
            counterparties.add(f)
        elif f == address:
            outgoing += 1
            counterparties.add(t)
        max_eth = max(max_eth, parse_eth(tx.get("value", "0")))

    score = 0
    alerts = []
    if len(txs) >= 40:
        score += 35
        alerts.append("短窗口交易频次高")
    if len(counterparties) >= 18:
        score += 25
        alerts.append("交易对手分散度异常")
    if max_eth >= 200:
        score += 30
        alerts.append("存在大额转账")
    if outgoing > incoming * 3 and outgoing >= 15:
        score += 20
        alerts.append("资金单向外流明显")

    score = min(100, score)
    level = "LOW" if score < 35 else "MEDIUM" if score < 70 else "HIGH"

    return {
        "in_count": incoming,
        "out_count": outgoing,
        "counterparty_count": len(counterparties),
        "max_tx_eth": round(max_eth, 6),
        "score": score,
        "level": level,
        "alerts": alerts,
    }


def cluster_addresses(address: str, txs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    参考 TPM 方案：基于对手地址特征做标准化后执行 KMeans 聚类。
    特征维度：
    - avg_fee_eth
    - total_fee_eth
    - tx_count
    - fee_std_eth
    - in_ratio
    - out_ratio
    """
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
        peers[peer]["volume_eth"] += parse_eth(tx.get("value", "0"))
        peers[peer]["fees"].append(float(tx.get("fee_eth", 0)))

    if len(peers) < 2:
        return {
            "method": "kmeans-feature-clustering",
            "target": address,
            "cluster_count": 0,
            "address_count": len(peers),
            "clusters": [],
            "peer_cluster_map": {},
            "note": "对手地址数量不足，无法执行聚类。",
        }

    if KMeans is None or StandardScaler is None:
        return {
            "method": "kmeans-feature-clustering",
            "target": address,
            "cluster_count": 0,
            "address_count": len(peers),
            "clusters": [],
            "peer_cluster_map": {},
            "note": "未安装 scikit-learn，无法执行 KMeans 聚类。",
        }

    peer_list = list(peers.keys())
    feature_rows = []
    for p in peer_list:
        s = peers[p]
        tx_count = s["in_cnt"] + s["out_cnt"]
        avg_fee = float(np.mean(s["fees"])) if s["fees"] else 0.0
        total_fee = float(np.sum(s["fees"])) if s["fees"] else 0.0
        fee_std = float(np.std(s["fees"])) if len(s["fees"]) > 1 else 0.0
        in_ratio = s["in_cnt"] / tx_count if tx_count else 0.0
        out_ratio = s["out_cnt"] / tx_count if tx_count else 0.0
        feature_rows.append([avg_fee, total_fee, tx_count, fee_std, in_ratio, out_ratio])

    mat = np.array(feature_rows, dtype=float)
    scaler = StandardScaler()
    mat_scaled = scaler.fit_transform(mat)

    n_clusters = min(3, len(peer_list))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(mat_scaled)

    grouped: dict[int, list[str]] = {}
    for peer, label in zip(peer_list, labels):
        grouped.setdefault(int(label), []).append(peer)

    peer_cluster_map: dict[str, str] = {}
    clusters = []
    for idx, (label, members) in enumerate(sorted(grouped.items(), key=lambda x: x[0]), start=1):
        interaction_count = sum(peers[m]["in_cnt"] + peers[m]["out_cnt"] for m in members)
        interaction_volume = round(sum(peers[m]["volume_eth"] for m in members), 6)
        cluster_fees = [fee for m in members for fee in peers[m]["fees"]]
        avg_fee = float(np.mean(cluster_fees)) if cluster_fees else 0.0
        total_fee = float(np.sum(cluster_fees)) if cluster_fees else 0.0

        # 将聚类标签映射为可解释业务名称
        if interaction_count >= 15:
            cluster_name = "高频交互簇"
        elif avg_fee > 0.001:
            cluster_name = "高费率交互簇"
        elif interaction_volume > 50:
            cluster_name = "大额流转簇"
        else:
            cluster_name = "一般交互簇"

        for m in members:
            peer_cluster_map[m] = cluster_name

        clusters.append(
            {
                "cluster_id": f"C{idx}",
                "raw_label": int(label),
                "label": cluster_name,
                "size": len(members),
                "interaction_count": interaction_count,
                "interaction_volume_eth": interaction_volume,
                "avg_fee_eth": round(avg_fee, 10),
                "total_fee_eth": round(total_fee, 10),
                "members": members,
            }
        )

    return {
        "method": "kmeans-feature-clustering",
        "feature_columns": ["avg_fee_eth", "total_fee_eth", "tx_count", "fee_std_eth", "in_ratio", "out_ratio"],
        "target": address,
        "cluster_count": len(clusters),
        "address_count": len(peers),
        "clusters": clusters,
        "peer_cluster_map": peer_cluster_map,
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


@app.errorhandler(ValidationError)
def on_validation_error(err: ValidationError):
    return jsonify({"error": str(err)}), 400


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


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


@app.route("/api/analyze", methods=["POST"])
def analyze():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    limit = max(10, min(int(body.get("limit", DEFAULT_TX_LIMIT)), 100))

    txs_raw = etherscan_fetch(address, limit)
    txs = compact_txs(txs_raw)
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


@app.route("/api/cluster", methods=["POST"])
def cluster_api():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    limit = max(10, min(int(body.get("limit", DEFAULT_TX_LIMIT)), 100))
    txs = compact_txs(etherscan_fetch(address, limit))
    clustering = cluster_addresses(address, txs)
    write_audit("cluster", address, {"limit": limit, "cluster_count": clustering["cluster_count"]})
    return jsonify(clustering)


@app.route("/api/evidence/register", methods=["POST"])
def register_evidence():
    body = request.get_json(silent=True) or {}
    address = normalize_address(body.get("address", ""))
    report_hash = str(body.get("report_hash", "")).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", report_hash):
        raise ValidationError("report_hash must be 64-char sha256 hex")

    tx_count = int(body.get("tx_count", 0))
    risk_score = int(body.get("risk_score", 0))
    risk_level = str(body.get("risk_level", "LOW")).upper()
    chain_tx_hash = body.get("chain_tx_hash") or None

    conn = db_conn()
    try:
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
                "SUCCESS" if chain_tx_hash else "SKIPPED",
                chain_tx_hash,
                str(body.get("note", "")).strip(),
                utc_now_iso(),
            ),
        )
        conn.commit()
        evidence_id = cur.lastrowid
    finally:
        conn.close()

    write_audit("register_evidence", address, {"id": evidence_id})
    return jsonify(
        {
            "id": evidence_id,
            "address": address,
            "report_hash": report_hash,
            "chain_tx_hash": chain_tx_hash,
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
    conn = db_conn()
    try:
        rows = conn.execute(
            """
            SELECT id, report_hash, tx_count, risk_score, risk_level, chain_status, chain_tx_hash, note, created_at
            FROM evidence_records WHERE address = ? ORDER BY id DESC LIMIT 50
            """,
            (address,),
        ).fetchall()
    finally:
        conn.close()
    return jsonify({"address": address, "history": [dict(r) for r in rows]})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
