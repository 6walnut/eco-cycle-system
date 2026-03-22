<template>
  <div class="wrap">
    <header>
      <h1>中国宏观经济周期 — 融合指数与预测</h1>
      <p class="sub">上传月度 CSV（首列 <code>date</code>），PCA 融合 + 阶段划分 + Holt-Winters / LSTM 预测</p>
    </header>

    <section class="panel">
      <h2>1. 上传数据</h2>
      <input type="file" accept=".csv" @change="onFile" />
      <button :disabled="!file || loading" @click="analyze">运行分析</button>
      <button :disabled="!file || loading" @click="saveDataset">保存到数据库</button>
      <span v-if="loading" class="muted">处理中…</span>
      <p v-if="error" class="err">{{ error }}</p>
    </section>

    <section class="panel grid">
      <div>
        <h2>2. 参数</h2>
        <label>预测模型</label>
        <select v-model="form.forecast_model">
          <option value="hw">Holt-Winters</option>
          <option value="lstm">LSTM（需安装 TensorFlow）</option>
        </select>
        <label>融合方式</label>
        <select v-model="form.fusion_method">
          <option value="pca">PCA</option>
          <option value="entropy">熵权</option>
          <option value="equal">等权</option>
        </select>
        <label>标准化</label>
        <select v-model="form.standardize">
          <option value="zscore">Z-score</option>
          <option value="minmax">Min-Max</option>
        </select>
        <label>变换</label>
        <select v-model="form.transform_type">
          <option value="none">无（同比数据用）</option>
          <option value="yoy">同比</option>
          <option value="mom">环比</option>
        </select>
        <label>预测月数</label>
        <input type="number" v-model.number="form.horizon_months" min="3" max="12" />
        <label>逆向指标（逗号分隔）</label>
        <input v-model="form.inverse_columns" placeholder="例如 unemployment" />
      </div>
      <div>
        <h2>3. 指标权重（PCA 等）</h2>
        <ul v-if="result?.weights" class="weights">
          <li v-for="(w, k) in result.weights" :key="k">
            <span>{{ k }}</span><strong>{{ w.toFixed(4) }}</strong>
          </li>
        </ul>
        <p v-else class="muted">运行后显示</p>
      </div>
    </section>

    <section class="panel" v-if="result">
      <h2>4. 综合指数与预测</h2>
      <p v-if="result.forecast_meta" class="muted">
        {{ result.forecast_meta?.note || JSON.stringify(result.forecast_meta) }}
      </p>
      <div ref="chartEl" class="chart"></div>
    </section>

    <section class="panel" v-if="result?.future_states?.length">
      <h2>5. 未来阶段</h2>
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
            <td>{{ r.forecast_composite?.toFixed?.(4) ?? r.forecast_composite }}</td>
            <td>{{ r.state_cn || r.state }}</td>
          </tr>
        </tbody>
      </table>
    </section>
  </div>
</template>

<script setup>
import { ref, onUnmounted, watch, nextTick } from "vue";
import axios from "axios";
import * as echarts from "echarts";

const file = ref(null);
const loading = ref(false);
const error = ref("");
const result = ref(null);
const chartEl = ref(null);
let chart = null;

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
    // Do not set Content-Type manually — browser must add multipart boundary
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

  if (!chart) chart = echarts.init(chartEl.value);
  chart.setOption({
    tooltip: { trigger: "axis" },
    legend: { data: ["历史综合指数", "预测"] },
    xAxis: { type: "time" },
    yAxis: { type: "value", scale: true },
    series: [
      { name: "历史综合指数", type: "line", data: histData, smooth: true, showSymbol: false },
      {
        name: "预测",
        type: "line",
        data: fcData,
        smooth: true,
        lineStyle: { type: "dashed" },
        showSymbol: true,
      },
    ],
  });
}

watch(result, async () => {
  await nextTick();
  renderChart();
});

onUnmounted(() => {
  chart?.dispose();
});
</script>

<style scoped>
.wrap {
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.5rem;
  font-family: system-ui, sans-serif;
}
header h1 {
  margin: 0 0 0.25rem;
  font-size: 1.35rem;
}
.sub {
  color: #555;
  font-size: 0.9rem;
}
.panel {
  margin-top: 1.25rem;
  padding: 1rem;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
}
.panel h2 {
  margin: 0 0 0.75rem;
  font-size: 1rem;
}
.grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
@media (max-width: 800px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
label {
  display: block;
  margin-top: 0.5rem;
  font-size: 0.85rem;
}
select,
input[type="number"],
input[type="text"] {
  width: 100%;
  max-width: 320px;
  padding: 0.35rem;
}
button {
  margin-left: 0.5rem;
  padding: 0.4rem 0.8rem;
  cursor: pointer;
}
.err {
  color: #c00;
}
.muted {
  color: #888;
  font-size: 0.9rem;
}
.chart {
  width: 100%;
  height: 380px;
}
.weights {
  list-style: none;
  padding: 0;
  margin: 0;
}
.weights li {
  display: flex;
  justify-content: space-between;
  padding: 0.25rem 0;
  border-bottom: 1px solid #eee;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th,
td {
  border: 1px solid #ddd;
  padding: 0.4rem;
  text-align: left;
}
</style>
