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

try:
    from web3 import Web3
except Exception:
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


def build_graph(address: str, txs: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [{"id": address, "name": address, "symbolSize": 56, "itemStyle": {"color": "#0b5fff"}}]
    links = []
    seen = {address}
    for tx in txs:
        src = str(tx.get("from", "")).lower()
        dst = str(tx.get("to", "")).lower()
        if not src or not dst:
            continue
        if src not in seen:
            nodes.append({"id": src, "name": src, "symbolSize": 28})
            seen.add(src)
        if dst not in seen:
            nodes.append({"id": dst, "name": dst, "symbolSize": 28})
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
        try:
            max_eth = max(max_eth, int(tx.get("value", "0")) / 1e18)
        except Exception:
            pass

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
def _on_validation(err):
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

    txs = etherscan_fetch(address, limit)
    graph = build_graph(address, txs)
    risk = detect_risk(address, txs)

    compact = []
    for tx in txs:
        v = tx.get("value", "0")
        compact.append(
            {
                "hash": tx.get("hash", ""),
                "from": str(tx.get("from", "")).lower(),
                "to": str(tx.get("to", "")).lower(),
                "value_wei": v,
                "value_eth": round(int(v) / 1e18, 6) if str(v).isdigit() else 0,
                "time_stamp": tx.get("timeStamp", "0"),
            }
        )

    report_hash = canonical_hash({"address": address, "risk": risk, "txs": compact})
    write_audit("analyze", address, {"limit": limit, "risk": risk})
    return jsonify(
        {
            "address": address,
            "tx_count": len(compact),
            "risk": risk,
            "graph": graph,
            "txs": compact,
            "report_hash": report_hash,
        }
    )


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
