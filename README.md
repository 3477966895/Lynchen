# TraceCoin 毕业设计项目

## 一、项目内容
1. 地址交易溯源分析（图谱可视化）
2. 地址聚类分析（参考 TPM 的 KMeans 特征聚类）
3. 规则化风险预警
4. PDF 报告生成与链上存证
5. 报告验真（链上 + 库内双通道）
6. 存证历史审计查询

## 二、目录说明
- `app.py`：Flask 后端
- `index.html`：前端页面
- `TraceCoin.sol.txt`：智能合约源码文本
- `environment.yml` / `requirements.txt`：Python 环境配置
- `.env`：运行配置

## 三、创建 Python 环境
### Conda 方式（推荐）
```powershell
conda env create -f environment.yml
conda activate tracecoin
```

### Pip 方式
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 四、运行系统
```powershell
python app.py
```

浏览器打开：`http://127.0.0.1:5000`

## 五、聚类接口
- `POST /api/cluster`
- 请求体：`{"address":"0x...","limit":50}`
- 返回：聚类方法、簇数量、各簇成员、交互次数、交互金额、平均手续费

## 六、聚类特征（KMeans）
对每个对手地址提取以下特征后做标准化，再执行 KMeans：
- `avg_fee_eth`：平均手续费
- `total_fee_eth`：总手续费
- `tx_count`：交互次数
- `fee_std_eth`：手续费标准差
- `in_ratio`：流入占比
- `out_ratio`：流出占比

## 七、链上部署说明（Ganache + Remix + MetaMask）
1. 启动 Ganache，本地 RPC 设为 `http://127.0.0.1:7545`
2. 在 Remix 新建 `TraceCoin.sol`，粘贴 `TraceCoin.sol.txt` 内容
3. 使用 `0.8.20` 编译并部署
4. 把部署后的合约地址同步到：
   - `.env` 的 `CONTRACT_ADDRESS`
   - `index.html` 的 `CONTRACT_ADDRESS`
5. 用 MetaMask 连接 Ganache 网络并导入测试账户进行签名
