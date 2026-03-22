<template>
  <div class="page">
    <div class="bg-pattern" aria-hidden="true" />

    <div class="wrap">
      <header class="hero">
        <div class="hero-badge">Macro · Cycle · Forecast</div>
        <h1>中国宏观经济周期分析</h1>
        <p class="hero-desc">
          上传月度 <code>CSV</code>（首列为 <code>date</code>），进行多指标融合、周期阶段识别与短期预测
        </p>
      </header>

      <section class="card card-upload">
        <div class="card-head">
          <span class="step">01</span>
          <h2>上传数据</h2>
        </div>
        <div class="upload-row">
          <label class="file-label">
            <input type="file" accept=".csv" class="file-input" @change="onFile" />
            <span class="file-btn">选择 CSV 文件</span>
            <span v-if="fileName" class="file-name">{{ fileName }}</span>
            <span v-else class="file-hint">未选择文件</span>
          </label>
          <div class="btn-group">
            <button type="button" class="btn btn-primary" :disabled="!file || loading" @click="analyze">
              <span v-if="loading" class="spinner" />
              {{ loading ? "分析中…" : "运行分析" }}
            </button>
            <button type="button" class="btn btn-secondary" :disabled="!file || loading" @click="saveDataset">
              保存到数据库
            </button>
          </div>
        </div>
        <p v-if="error" class="alert alert-error">{{ error }}</p>
      </section>

      <div class="grid-2">
        <section class="card">
          <div class="card-head">
            <span class="step">02</span>
            <h2>模型参数</h2>
          </div>
          <div class="form-grid">
            <div class="field">
              <label>预测模型</label>
              <select v-model="form.forecast_model" class="input">
                <option value="hw">Holt-Winters</option>
                <option value="lstm">LSTM（需 TensorFlow）</option>
              </select>
            </div>
            <div class="field">
              <label>融合方式</label>
              <select v-model="form.fusion_method" class="input">
                <option value="pca">PCA</option>
                <option value="entropy">熵权</option>
                <option value="equal">等权</option>
              </select>
            </div>
            <div class="field">
              <label>标准化</label>
              <select v-model="form.standardize" class="input">
                <option value="zscore">Z-score</option>
                <option value="minmax">Min-Max</option>
              </select>
            </div>
            <div class="field">
              <label>变换</label>
              <select v-model="form.transform_type" class="input">
                <option value="none">无（同比序列）</option>
                <option value="yoy">同比</option>
                <option value="mom">环比</option>
              </select>
            </div>
            <div class="field">
              <label>预测月数</label>
              <input v-model.number="form.horizon_months" class="input" type="number" min="3" max="12" />
            </div>
            <div class="field field-span">
              <label>逆向指标（逗号分隔）</label>
              <input v-model="form.inverse_columns" class="input" placeholder="例如 unemployment" />
            </div>
          </div>
        </section>

        <section class="card card-weights">
          <div class="card-head">
            <span class="step">03</span>
            <h2>指标权重</h2>
          </div>
          <ul v-if="result?.weights" class="weight-list">
            <li v-for="(w, k) in result.weights" :key="k">
              <span class="w-name">{{ labelZh(k) }}</span>
              <div class="w-bar-wrap">
                <div class="w-bar" :style="{ width: `${Math.min(100, w * 400)}%` }" />
              </div>
              <span class="w-val">{{ w.toFixed(4) }}</span>
            </li>
          </ul>
          <p v-else class="empty-hint">运行分析后显示各指标权重</p>
        </section>
      </div>

      <section v-if="result" class="card card-chart">
        <div class="card-head">
          <span class="step">04</span>
          <h2>综合指数与预测</h2>
        </div>
        <p v-if="result.forecast_meta?.note" class="meta-note">{{ result.forecast_meta.note }}</p>
        <div ref="chartEl" class="chart"></div>
      </section>

      <section v-if="result?.future_states?.length" class="card">
        <div class="card-head">
          <span class="step">05</span>
          <h2>未来阶段预测</h2>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>日期</th>
                <th>预测指数</th>
                <th>阶段</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in result.future_states" :key="r.date">
                <td>{{ r.date }}</td>
                <td class="num">{{ r.forecast_composite?.toFixed?.(4) ?? r.forecast_composite }}</td>
                <td><span class="tag">{{ r.state_cn || r.state }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <footer class="footer">
        <span>Eco Cycle System · 毕业设计原型</span>
      </footer>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onUnmounted, watch, nextTick } from "vue";
import axios from "axios";
import * as echarts from "echarts";

const file = ref(null);
const loading = ref(false);
const error = ref("");
const result = ref(null);
const chartEl = ref(null);
let chart = null;

const fileName = computed(() => file.value?.name ?? "");

/** 英文列名 -> 中文展示（与 Wind/CSV 常用列一致） */
const INDICATOR_LABELS = {
  cpi_yoy: "CPI（当月同比）",
  pmi: "制造业PMI",
  m2_yoy: "M2（同比）",
  ind_growth_yoy: "工业增加值（同比）",
  fai_acc_yoy: "固定资产投资（累计同比）",
  social_finance_yoy: "社会融资规模存量（同比）",
  gdp: "GDP（代理）",
  industrial_production: "工业增加值",
  cpi: "CPI",
  m2: "M2",
  credit: "信贷",
  unemployment: "失业率",
  fx: "汇率",
};

function labelZh(key) {
  return INDICATOR_LABELS[key] ?? key;
}

const form = ref({
  forecast_model: "hw",
  fusion_method: "pca",
  standardize: "zscore",
  transform_type: "none",
  horizon_months: 6,
  inverse_columns: "",
});

function onFile(e) {
  const f = e.target.files?.[0];
  file.value = f || null;
  error.value = "";
}

async function analyze() {
  if (!file.value) return;
  loading.value = true;
  error.value = "";
  result.value = null;
  const fd = new FormData();
  fd.append("file", file.value);
  fd.append("use_sample", "false");
  fd.append("forecast_model", form.value.forecast_model);
  fd.append("fusion_method", form.value.fusion_method);
  fd.append("standardize", form.value.standardize);
  fd.append("transform_type", form.value.transform_type);
  fd.append("horizon_months", String(form.value.horizon_months));
  if (form.value.inverse_columns) fd.append("inverse_columns", form.value.inverse_columns);
  try {
    const { data } = await axios.post("/api/analyze", fd, {
      timeout: 120000,
    });
    result.value = data;
  } catch (e) {
    error.value = e.response?.data?.error || e.message || String(e);
  } finally {
    loading.value = false;
  }
}

async function saveDataset() {
  if (!file.value) return;
  loading.value = true;
  error.value = "";
  const fd = new FormData();
  fd.append("file", file.value);
  fd.append("name", file.value.name || "upload");
  try {
    const { data } = await axios.post("/api/datasets", fd);
    alert(`已保存到数据库，dataset_id=${data.id}，共 ${data.rows} 行`);
  } catch (e) {
    error.value = e.response?.data?.error || e.message;
  } finally {
    loading.value = false;
  }
}

function renderChart() {
  if (!chartEl.value || !result.value) return;
  const hist = result.value.composite_history || [];
  const fc = result.value.forecast || [];
  const histData = hist.map((x) => [x.date, x.composite]);
  const fcData = fc.map((x) => [x.date, x.forecast_composite]);

  chart?.dispose();
  chart = echarts.init(chartEl.value, null, { renderer: "canvas" });
  chart.setOption({
    color: ["#0d9488", "#f59e0b"],
    textStyle: { fontFamily: "'Noto Sans SC', 'DM Sans', sans-serif" },
    grid: { left: 48, right: 24, top: 48, bottom: 56 },
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(15, 23, 42, 0.92)",
      borderColor: "transparent",
      textStyle: { color: "#f1f5f9" },
    },
    legend: {
      data: ["历史综合指数", "预测"],
      top: 0,
      textStyle: { color: "#64748b" },
    },
    xAxis: {
      type: "time",
      axisLine: { lineStyle: { color: "#cbd5e1" } },
      axisLabel: { color: "#64748b" },
      splitLine: { show: false },
    },
    yAxis: {
      type: "value",
      scale: true,
      axisLine: { show: false },
      axisLabel: { color: "#64748b" },
      splitLine: { lineStyle: { color: "#e2e8f0", type: "dashed" } },
    },
    series: [
      {
        name: "历史综合指数",
        type: "line",
        data: histData,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 2.5 },
        areaStyle: {
          color: {
            type: "linear",
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: "rgba(13, 148, 136, 0.28)" },
              { offset: 1, color: "rgba(13, 148, 136, 0)" },
            ],
          },
        },
      },
      {
        name: "预测",
        type: "line",
        data: fcData,
        smooth: true,
        lineStyle: { type: "dashed", width: 2 },
        showSymbol: true,
        symbolSize: 6,
      },
    ],
  });
}

watch(result, async (val) => {
  if (!val) {
    chart?.dispose();
    chart = null;
    return;
  }
  await nextTick();
  renderChart();
});

onUnmounted(() => {
  chart?.dispose();
});
</script>

<style scoped>
.page {
  min-height: 100vh;
  position: relative;
  background: linear-gradient(165deg, #f0fdfa 0%, #f8fafc 45%, #eef2ff 100%);
  font-family: "Noto Sans SC", "DM Sans", system-ui, sans-serif;
  color: #0f172a;
}

.bg-pattern {
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.4;
  background-image: radial-gradient(#94a3b8 0.5px, transparent 0.5px);
  background-size: 20px 20px;
  mask-image: linear-gradient(to bottom, black, transparent 85%);
}

.wrap {
  position: relative;
  max-width: 1080px;
  margin: 0 auto;
  padding: 2rem 1.25rem 3rem;
}

.hero {
  text-align: center;
  margin-bottom: 2rem;
  padding: 1.5rem 0;
}

.hero-badge {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #0d9488;
  background: rgba(13, 148, 136, 0.12);
  padding: 0.35rem 0.75rem;
  border-radius: 999px;
  margin-bottom: 0.75rem;
}

.hero h1 {
  margin: 0;
  font-size: clamp(1.5rem, 4vw, 1.85rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #0f172a 0%, #0d9488 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.hero-desc {
  margin: 0.75rem auto 0;
  max-width: 36rem;
  font-size: 0.95rem;
  line-height: 1.6;
  color: #64748b;
}

.hero-desc code {
  font-size: 0.88em;
  padding: 0.08rem 0.35rem;
  background: #e2e8f0;
  border-radius: 4px;
  color: #0f172a;
}

.card {
  background: #fff;
  border-radius: 16px;
  padding: 1.35rem 1.5rem;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 8px 24px rgba(15, 23, 42, 0.06);
  border: 1px solid rgba(148, 163, 184, 0.15);
  margin-bottom: 1.25rem;
}

.card-head {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  margin-bottom: 1.1rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #f1f5f9;
}

.card-head h2 {
  margin: 0;
  font-size: 1.05rem;
  font-weight: 600;
  color: #0f172a;
}

.step {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.75rem;
  height: 1.75rem;
  font-size: 0.75rem;
  font-weight: 700;
  color: #fff;
  background: linear-gradient(135deg, #0d9488, #14b8a6);
  border-radius: 8px;
}

.card-upload .upload-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 1rem;
}

.file-label {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.65rem;
  cursor: pointer;
}

.file-input {
  position: absolute;
  width: 0;
  height: 0;
  opacity: 0;
}

.file-btn {
  display: inline-block;
  padding: 0.55rem 1.1rem;
  font-size: 0.9rem;
  font-weight: 500;
  color: #0f766e;
  background: #ecfdf5;
  border: 1px solid #99f6e4;
  border-radius: 10px;
  transition: background 0.2s, box-shadow 0.2s;
}

.file-label:hover .file-btn {
  background: #d1fae5;
  box-shadow: 0 2px 8px rgba(13, 148, 136, 0.15);
}

.file-name {
  font-size: 0.88rem;
  color: #334155;
  max-width: 220px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.file-hint {
  font-size: 0.85rem;
  color: #94a3b8;
}

.btn-group {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}

.btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.55rem 1.15rem;
  font-size: 0.9rem;
  font-weight: 600;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: transform 0.15s, box-shadow 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn:not(:disabled):hover {
  transform: translateY(-1px);
}

.btn-primary {
  color: #fff;
  background: linear-gradient(135deg, #0d9488, #0f766e);
  box-shadow: 0 4px 14px rgba(13, 148, 136, 0.35);
}

.btn-secondary {
  color: #475569;
  background: #f1f5f9;
  border: 1px solid #e2e8f0;
}

.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(255, 255, 255, 0.35);
  border-top-color: #fff;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.alert-error {
  margin: 0.85rem 0 0;
  padding: 0.65rem 0.85rem;
  font-size: 0.88rem;
  color: #b91c1c;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 10px;
}

.grid-2 {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.25rem;
  margin-bottom: 1.25rem;
}

@media (max-width: 880px) {
  .grid-2 {
    grid-template-columns: 1fr;
  }
}

.form-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.85rem 1rem;
}

.field-span {
  grid-column: 1 / -1;
}

.field label {
  display: block;
  margin-bottom: 0.35rem;
  font-size: 0.78rem;
  font-weight: 500;
  color: #64748b;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.input {
  width: 100%;
  padding: 0.5rem 0.65rem;
  font-size: 0.9rem;
  font-family: inherit;
  color: #0f172a;
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  transition: border-color 0.2s, box-shadow 0.2s;
}

.input:focus {
  outline: none;
  border-color: #2dd4bf;
  box-shadow: 0 0 0 3px rgba(45, 212, 191, 0.2);
}

.card-weights {
  min-height: 200px;
}

.weight-list {
  list-style: none;
  padding: 0;
  margin: 0;
}

.weight-list li {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 2fr auto;
  align-items: center;
  gap: 0.65rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid #f1f5f9;
}

.weight-list li:last-child {
  border-bottom: none;
}

.w-name {
  font-size: 0.82rem;
  color: #475569;
  overflow: hidden;
  text-overflow: ellipsis;
}

.w-bar-wrap {
  height: 6px;
  background: #f1f5f9;
  border-radius: 999px;
  overflow: hidden;
}

.w-bar {
  height: 100%;
  background: linear-gradient(90deg, #0d9488, #5eead4);
  border-radius: 999px;
  max-width: 100%;
  transition: width 0.4s ease;
}

.w-val {
  font-size: 0.8rem;
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  color: #0ea5e9;
  min-width: 4.5rem;
  text-align: right;
}

.empty-hint {
  margin: 2rem 0;
  text-align: center;
  font-size: 0.9rem;
  color: #94a3b8;
}

.card-chart .meta-note {
  margin: 0 0 0.75rem;
  font-size: 0.82rem;
  color: #64748b;
}

.chart {
  width: 100%;
  height: 400px;
}

.table-wrap {
  overflow-x: auto;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}

thead {
  background: linear-gradient(180deg, #f8fafc, #f1f5f9);
}

th {
  padding: 0.65rem 0.85rem;
  text-align: left;
  font-weight: 600;
  color: #475569;
  border-bottom: 1px solid #e2e8f0;
}

td {
  padding: 0.65rem 0.85rem;
  border-bottom: 1px solid #f1f5f9;
}

tbody tr:nth-child(even) {
  background: #fafbfc;
}

tbody tr:hover {
  background: #f0fdfa;
}

.num {
  font-variant-numeric: tabular-nums;
  color: #0ea5e9;
  font-weight: 500;
}

.tag {
  display: inline-block;
  padding: 0.2rem 0.55rem;
  font-size: 0.8rem;
  font-weight: 500;
  color: #0d9488;
  background: #ecfdf5;
  border-radius: 6px;
}

.footer {
  margin-top: 2rem;
  padding-top: 1.25rem;
  text-align: center;
  font-size: 0.78rem;
  color: #94a3b8;
}
</style>
