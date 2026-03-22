# Eco Cycle System（宏观经济周期分析与可视化）

基于多指标融合的综合经济周期指数、阶段划分与短期预测；提供 **Flask REST API**、**SQLite/MySQL 持久化**、**Vue3 + ECharts 前端**，可选 **LSTM** 预测。

## 功能概览

1. **数据预处理**：CSV（`date` + 指标列）、缺失插值、分位数截断、可选 `mom`/`yoy`
2. **融合**：`zscore` / `minmax` + `equal` / `pca` / `entropy`
3. **周期阶段**：基于综合指数与移动平均的启发式四阶段
4. **预测**：`Holt-Winters`（默认）或 **LSTM**（需安装 TensorFlow，见 `requirements-ml.txt`）
5. **数据库**：默认 SQLite（`data/eco_cycle.db`），可切换 MySQL

## 环境

```bash
pip install -r requirements.txt
# 可选：LSTM
pip install -r requirements-ml.txt
```

## 后端 API

```bash
python api_server.py
```

- `GET /api/health`
- `GET /api/sample`
- `POST /api/analyze` — 上传 CSV 分析（表单字段见下）
- `POST /api/datasets` — 上传 CSV **保存到数据库**
- `GET /api/datasets` — 数据集列表
- `GET /api/datasets/<id>` — 预览数据
- `POST /api/datasets/<id>/analyze` — 对已存数据集分析并写入 `analysis_runs`
- `GET /api/runs/<id>` — 查询某次分析结果

### `POST /api/analyze` 表单字段

| 字段 | 说明 |
|------|------|
| `file` | CSV 文件 |
| `use_sample` | `true` 使用内置样例 |
| `forecast_model` | `hw`（默认）或 `lstm` |
| `inverse_columns` | 逗号分隔需要反向的列名 |
| `transform_type` | `none` / `mom` / `yoy` |
| `standardize` | `zscore` / `minmax` |
| `fusion_method` | `equal` / `pca` / `entropy` |
| `horizon_months` | 预测月数 3–12 |

返回 JSON 含：`weights`、`composite_history`、`states_history`、`forecast`、`future_states`、`forecast_model`、`forecast_meta`。

## MySQL（可选）

```bash
set DATABASE_URL=mysql+pymysql://用户:密码@127.0.0.1:3306/数据库名
python api_server.py
```

需先建空库；表结构由 SQLAlchemy 自动创建。

## 前端（Vue3 + Vite + ECharts）

```bash
cd frontend
npm install
npm run dev
```

浏览器访问 `http://localhost:5173`（`vite` 代理 `/api` 到 `http://127.0.0.1:5000`）。请先启动 `api_server.py`。

## Streamlit（可选原型）

```bash
streamlit run app.py
```

## CSV 格式

首列必须为 `date`（如 `2010-01-01`），其余列为数值型指标。

```csv
date,cpi_yoy,pmi,m2_yoy
2010-01-01,1.5,55.8,25.98
```
