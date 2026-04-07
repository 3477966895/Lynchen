import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
from community import community_louvain
import base58

app = Flask(__name__)
CORS(app)

# 全局缓存（演示用）
cache = {
    'address': None,
    'transactions': [],
    'counterparties': {},
    'cluster_labels': None,
    'latest_tx_hash': None
}

# 波场API基础URL
TRONGRID_API = "https://api.trongrid.io"
USDT_CONTRACT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
TRANSFER_SIG = "a9059cbb"
BLACKLIST = ["TBlacklistExample1234567890", "TAnotherBadAddress111"]

# ----------------------------------------------------------------------
# 辅助函数
def normalize_address(address):
    """将输入地址统一转换为Hex格式（以41开头，42字符）"""
    address = address.strip()
    if address.startswith('41') and len(address) == 42:
        return address
    try:
        decoded = base58.b58decode(address)
        hex_addr = decoded.hex()
        if not hex_addr.startswith('41'):
            hex_addr = '41' + hex_addr
        return hex_addr
    except Exception as e:
        print(f"[ERROR] Base58解码失败: {e}")
        return None

def decode_transfer_data(data):
    """简化版解码transfer方法的data字段，返回(to, value)"""
    if data.startswith(TRANSFER_SIG):
        # 实际解析需要更严谨，这里返回示例值（可扩展）
        to_addr = "TReceiverExample"
        value = 100
        return to_addr, value
    return None, None

def fetch_transactions(address, limit=100):
    """获取地址的TRX和USDT交易记录（自动识别Base58或Hex格式）"""
    hex_address = normalize_address(address)
    if not hex_address:
        print("[ERROR] 地址格式无效")
        return []

    url = f"{TRONGRID_API}/v1/accounts/{hex_address}/transactions"
    params = {'limit': limit, 'only_confirmed': True}
    print(f"[DEBUG] 请求URL: {url}")
    transactions = []

    try:
        response = requests.get(url, params=params, timeout=10)
        print(f"[DEBUG] 状态码: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERROR] API错误: {response.text}")
            return []

        data = response.json()
        if 'data' not in data or not data['data']:
            print("[DEBUG] API返回空数据")
            return []

        for tx in data['data']:
            tx_id = tx.get('txID', '')
            timestamp = tx.get('block_timestamp', 0)
            if timestamp:
                timestamp = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')

            # 手续费
            fee = 0
            if 'ret' in tx and len(tx['ret']) > 0:
                fee_info = tx['ret'][0]
                fee = fee_info.get('fee', fee_info.get('cost', 0)) / 1e6

            contracts = tx.get('raw_data', {}).get('contract', [])
            if not contracts:
                continue
            contract = contracts[0]
            contract_type = contract.get('type')
            param = contract.get('parameter', {}).get('value', {})

            # ---------- TRX转账 ----------
            if contract_type == 'TransferContract':
                from_addr = param.get('owner_address')
                to_addr = param.get('to_address')
                amount = param.get('amount', 0) / 1e6
                token = 'TRX'
                method = None

                if from_addr == hex_address:
                    direction = 'out'
                    counterparty = to_addr
                elif to_addr == hex_address:
                    direction = 'in'
                    counterparty = from_addr
                else:
                    continue

                transactions.append({
                    'tx_id': tx_id,
                    'timestamp': timestamp,
                    'direction': direction,
                    'counterparty': counterparty,
                    'amount': amount,
                    'fee': fee,
                    'token': token,
                    'method': method
                })

            # ---------- USDT (TRC20) 转账 ----------
            elif contract_type == 'TriggerSmartContract':
                contract_address = param.get('contract_address')
                if contract_address == USDT_CONTRACT:
                    data_hex = param.get('data', '')
                    if data_hex.startswith(TRANSFER_SIG):
                        to_addr, value = decode_transfer_data(data_hex)
                        from_addr = param.get('owner_address')
                        amount = value / 1e6
                        token = 'USDT'
                        method = 'transfer'

                        if from_addr == hex_address:
                            direction = 'out'
                            counterparty = to_addr
                        elif to_addr == hex_address:
                            direction = 'in'
                            counterparty = from_addr
                        else:
                            continue

                        transactions.append({
                            'tx_id': tx_id,
                            'timestamp': timestamp,
                            'direction': direction,
                            'counterparty': counterparty,
                            'amount': amount,
                            'fee': fee,
                            'token': token,
                            'method': method
                        })

        return transactions
    except Exception as e:
        print(f"[ERROR] fetch_transactions异常: {e}")
        return []

def get_latest_tx_hash(address):
    """获取地址最新的一笔交易哈希（用于实时更新）"""
    txs = fetch_transactions(address, limit=1)
    if txs:
        return txs[0]['tx_id']
    return None

# ----------------------------------------------------------------------
# 路由

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/test', methods=['GET'])
def test():
    return jsonify({'status': 'ok', 'message': '服务正常运行'})

@app.route('/api/query', methods=['POST'])
def query_transactions():
    data = request.json
    address = data.get('address', '').strip()
    print(f"[DEBUG] 接收到地址: {address}")
    if not address:
        return jsonify({'error': '地址不能为空'}), 400

    transactions = fetch_transactions(address)
    print(f"[DEBUG] 获取到交易数量: {len(transactions)}")
    if not transactions:
        return jsonify({'error': '未查询到交易记录，请检查地址是否正确或有无TRX/USDT转账'}), 404

    cache['address'] = address
    cache['transactions'] = transactions
    cache['latest_tx_hash'] = get_latest_tx_hash(address)

    counterparty_fees = defaultdict(list)
    for tx in transactions:
        cp = tx['counterparty']
        if cp:
            counterparty_fees[cp].append(tx['fee'])
    features = {}
    for cp, fees in counterparty_fees.items():
        features[cp] = {
            'avg_fee': np.mean(fees),
            'total_fee': np.sum(fees),
            'tx_count': len(fees),
            'fee_std': np.std(fees) if len(fees) > 1 else 0
        }
    cache['counterparties'] = features

    if len(features) >= 2:
        addresses = list(features.keys())
        mat = np.array([[f['avg_fee'], f['total_fee'], f['tx_count'], f['fee_std']] for f in features.values()])
        scaler = StandardScaler()
        mat_scaled = scaler.fit_transform(mat)
        n_clusters = min(3, len(addresses))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(mat_scaled)
        cache['cluster_labels'] = {addr: int(label) for addr, label in zip(addresses, labels)}
    else:
        cache['cluster_labels'] = None

    return jsonify({
        'address': address,
        'transactions': transactions,
        'counterparty_count': len(features),
        'cluster_available': cache['cluster_labels'] is not None
    })

@app.route('/api/export', methods=['GET'])
def export_transactions():
    if not cache['transactions']:
        return jsonify({'error': '无数据'}), 400
    export_data = []
    for tx in cache['transactions']:
        export_data.append({
            '交易哈希': tx['tx_id'],
            '时间': tx['timestamp'],
            '方向': '支出' if tx['direction'] == 'out' else '收入',
            '对手地址': tx['counterparty'],
            '金额': tx['amount'],
            '币种': tx['token'],
            '手续费(TRX)': tx['fee'],
            '合约方法': tx['method'] or ''
        })
    return jsonify(export_data)

@app.route('/api/batch', methods=['POST'])
def batch_query():
    data = request.json
    addresses = data.get('addresses', [])
    if len(addresses) > 5:
        return jsonify({'error': '最多支持5个地址'}), 400
    all_txs = []
    for addr in addresses:
        txs = fetch_transactions(addr)
        all_txs.extend(txs)
    cp_counts = defaultdict(int)
    for tx in all_txs:
        cp_counts[tx['counterparty']] += 1
    common = [cp for cp, cnt in cp_counts.items() if cnt >= len(addresses)]
    return jsonify({'transactions': all_txs, 'common_counterparties': common})

@app.route('/api/trace', methods=['POST'])
def trace_flow():
    data = request.json
    address = data.get('address')
    direction = data.get('direction')
    if not address:
        return jsonify({'error': '地址不能为空'}), 400
    txs = fetch_transactions(address)
    neighbors = set()
    for tx in txs:
        if direction == 'forward' and tx['direction'] == 'out':
            neighbors.add(tx['counterparty'])
        elif direction == 'backward' and tx['direction'] == 'in':
            neighbors.add(tx['counterparty'])
    paths = []
    for nb in neighbors:
        nb_txs = fetch_transactions(nb)
        related = [t for t in nb_txs if t['counterparty'] == address]
        if related:
            total_amount = sum(t['amount'] for t in related)
            paths.append({
                'from': address if direction == 'forward' else nb,
                'to': nb if direction == 'forward' else address,
                'amount': total_amount,
                'token': related[0]['token']
            })
    return jsonify({'paths': paths, 'direction': direction})

@app.route('/api/profile', methods=['POST'])
def address_profile():
    data = request.json
    address = data.get('address')
    txs = fetch_transactions(address)
    if not txs:
        return jsonify({'error': '无交易数据'}), 400

    total_amount = sum(tx['amount'] for tx in txs)
    total_fee = sum(tx['fee'] for tx in txs)
    counterparties = set(tx['counterparty'] for tx in txs)
    active_days = len(set(tx['timestamp'][:10] for tx in txs))

    risk_score = 0
    if address in BLACKLIST:
        risk_score += 50
    if len(counterparties) > 50:
        risk_score += 20
    if total_amount > 1000000:
        risk_score += 30
    risk_score = min(risk_score, 100)
    risk_level = '高危' if risk_score > 70 else ('中危' if risk_score > 40 else '低危')

    profile = {
        'address': address,
        'total_transactions': len(txs),
        'total_amount': total_amount,
        'total_fee': total_fee,
        'unique_counterparties': len(counterparties),
        'active_days': active_days,
        'risk_score': risk_score,
        'risk_level': risk_level
    }
    return jsonify(profile)

@app.route('/api/timeseries', methods=['POST'])
def time_series():
    data = request.json
    address = data.get('address')
    txs = fetch_transactions(address)
    if not txs:
        return jsonify({'error': '无交易数据'}), 400

    df = pd.DataFrame(txs)
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_amount = df.groupby('date')['amount'].sum().reset_index()
    daily_amount.columns = ['date', 'amount']
    daily_count = df.groupby('date').size().reset_index(name='count')
    merged = pd.merge(daily_amount, daily_count, on='date')
    mean = merged['amount'].mean()
    std = merged['amount'].std()
    merged['is_anomaly'] = (merged['amount'] > mean + 3*std) | (merged['amount'] < mean - 3*std)
    series = merged.to_dict(orient='records')
    anomalies = [row for row in series if row['is_anomaly']]
    return jsonify({'series': series, 'anomalies': anomalies})

@app.route('/api/graph', methods=['POST'])
def build_graph():
    data = request.json
    address = data.get('address')
    txs = fetch_transactions(address)
    if not txs:
        return jsonify({'error': '无交易数据'}), 400

    G = nx.Graph()
    G.add_node(address)
    for tx in txs:
        cp = tx['counterparty']
        G.add_node(cp)
        G.add_edge(address, cp, weight=tx['amount'])

    partition = community_louvain.best_partition(G)
    centrality = nx.degree_centrality(G)

    nodes = []
    for node in G.nodes():
        size = 20 + (centrality.get(node, 0) * 50)
        category = partition.get(node, -1) + 1
        if node == address:
            category = 0
        nodes.append({
            'name': node,
            'label': node[:10] + '...' if len(node) > 16 else node,
            'symbolSize': size,
            'category': category,
            'value': centrality.get(node, 0)
        })

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({
            'source': u,
            'target': v,
            'amount': d['weight'],
            'lineStyle': {'width': min(5, 1 + d['weight'] / 5000)}
        })

    categories = [{'name': '主地址'}]
    max_cat = max(partition.values()) if partition else -1
    for i in range(max_cat + 1):
        categories.append({'name': f'社区 {i+1}'})

    return jsonify({'nodes': nodes, 'links': edges, 'categories': categories})

@app.route('/api/cluster', methods=['GET'])
def get_cluster():
    if not cache['counterparties']:
        return jsonify({'error': '请先查询地址'}), 400
    if cache['cluster_labels'] is None:
        return jsonify({'error': '对手地址不足，无法聚类'}), 400
    clusters = defaultdict(list)
    for addr, label in cache['cluster_labels'].items():
        clusters[label].append(addr)
    result = {}
    for label, addrs in clusters.items():
        avg_fees = [cache['counterparties'][a]['avg_fee'] for a in addrs]
        total_fees = [cache['counterparties'][a]['total_fee'] for a in addrs]
        result[int(label)] = {
            'addresses': addrs,
            'count': len(addrs),
            'avg_fee': np.mean(avg_fees) if avg_fees else 0,
            'total_fee': np.sum(total_fees) if total_fees else 0
        }
    return jsonify({'clusters': result, 'total': len(cache['counterparties'])})

@app.route('/api/latest', methods=['GET'])
def latest_transaction():
    address = request.args.get('address')
    if not address or address != cache['address']:
        return jsonify({'latest_hash': None, 'changed': False})
    current_hash = get_latest_tx_hash(address)
    changed = (cache['latest_tx_hash'] != current_hash)
    cache['latest_tx_hash'] = current_hash
    return jsonify({'latest_hash': current_hash, 'changed': changed})

if __name__ == '__main__':
    print("已注册的路由:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.methods} {rule}")
    app.run(debug=True, host='0.0.0.0', port=5000)