# TraceCoin 毕业设计项目

## 一、项目内容
1. 地址交易溯源分析（图谱可视化）
2. 规则化风险预警
3. PDF 报告生成与链上存证
4. 报告验真（链上 + 库内双通道）
5. 存证历史审计查询

## 二、目录说明
- `app.py`：Flask 后端
- `index.html`：前端页面
- `TraceCoin.sol.txt`：智能合约源码文本
- `environment.yml` / `requirements.txt`：Python 环境配置
- `.env`：运行配置（已按你之前参数写入）

## 三、创建 Python 环境
### Conda 方式（推荐）
```powershell
conda env create -f environment.yml
conda activate tracecoin
```

### Pip 方式
```powershell
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## 四、运行系统
```powershell
python app.py
```

浏览器打开：`http://127.0.0.1:5000`

## 五、链上部署说明（Ganache + Remix + MetaMask）
1. 启动 Ganache，本地 RPC 设为 `http://127.0.0.1:7545`
2. 在 Remix 新建 `TraceCoin.sol`，粘贴 `TraceCoin.sol.txt` 内容
3. 使用 `0.8.20` 编译并部署
4. 把部署后的合约地址同步到：
   - `.env` 的 `CONTRACT_ADDRESS`
   - `index.html` 的 `CONTRACT_ADDRESS`
5. 用 MetaMask 连接 Ganache 网络并导入测试账户进行签名

## 六、论文说明
本科论文文本已恢复为：`毕业论文_TraceCoin.md`
