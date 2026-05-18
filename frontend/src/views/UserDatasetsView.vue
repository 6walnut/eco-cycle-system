<template>
  <div class="page">
    <header class="top">
      <h1>我的数据集</h1>
      <div class="actions">
        <button class="btn" @click="goHome">返回主页</button>
        <button class="btn primary" @click="refresh">刷新</button>
      </div>
    </header>

    <section class="card">
      <p class="hint">可基于历史数据集重新运行模型并修改参数。</p>
      <div class="table-wrap" v-if="datasets.length">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>名称</th>
              <th>行数</th>
              <th>创建时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="d in datasets" :key="d.id">
              <td>{{ d.id }}</td>
              <td>{{ d.name }}</td>
              <td>{{ d.rows }}</td>
              <td>{{ d.created_at }}</td>
              <td>
                <button class="btn mini" @click="rerun(d.id)">重跑模型</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="hint">暂无数据集</p>
    </section>

    <section class="card">
      <h2>重跑参数</h2>
      <div class="grid">
        <label>融合方式
          <select v-model="params.fusion_method">
            <option value="pca">PCA</option>
            <option value="dfm">DFM</option>
            <option value="entropy">Entropy</option>
            <option value="equal">Equal</option>
          </select>
        </label>
        <label>预测模型
          <select v-model="params.forecast_model">
            <option value="hw">Holt-Winters</option>
            <option value="lstm">LSTM</option>
          </select>
        </label>
        <label>预测月数
          <input v-model.number="params.horizon_months" type="number" min="3" max="12" />
        </label>
      </div>
      <p v-if="lastRunId" class="hint">已重跑成功，run_id={{ lastRunId }}</p>
      <p v-if="error" class="error">{{ error }}</p>
    </section>
  </div>
</template>

<script setup>
import axios from "axios";
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const datasets = ref([]);
const error = ref("");
const lastRunId = ref(null);
const params = ref({
  fusion_method: "pca",
  forecast_model: "hw",
  horizon_months: 6,
});

function auth() {
  const token = localStorage.getItem("eco_token") || "";
  return { headers: { Authorization: `Bearer ${token}` } };
}

async function refresh() {
  error.value = "";
  try {
    const { data } = await axios.get("/api/me/datasets", auth());
    datasets.value = data || [];
  } catch (e) {
    error.value = e.response?.data?.error || e.message || String(e);
  }
}

async function rerun(dsId) {
  error.value = "";
  try {
    const fd = new FormData();
    fd.append("fusion_method", params.value.fusion_method);
    fd.append("forecast_model", params.value.forecast_model);
    fd.append("horizon_months", String(params.value.horizon_months));
    const { data } = await axios.post(`/api/datasets/${dsId}/analyze`, fd, auth());
    lastRunId.value = data.run_id;
  } catch (e) {
    error.value = e.response?.data?.error || e.message || String(e);
  }
}

function goHome() {
  router.push("/home");
}

onMounted(refresh);
</script>

<style scoped>
.page { min-height: 100vh; padding: 1.2rem; background: #f8fafc; }
.top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.actions { display: flex; gap: 0.5rem; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.55rem; border-bottom: 1px solid #f1f5f9; text-align: left; }
.btn { border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.45rem 0.75rem; background: #fff; cursor: pointer; }
.btn.primary { background: #0d9488; color: #fff; border: none; }
.btn.mini { padding: 0.3rem 0.6rem; font-size: 0.82rem; }
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
label { display: flex; flex-direction: column; gap: 0.35rem; font-size: 0.86rem; color: #475569; }
input, select { border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.45rem 0.55rem; }
.hint { color: #64748b; }
.error { color: #b91c1c; }
</style>
