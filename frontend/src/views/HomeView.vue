<template>
  <div class="home-page">
    <header class="topbar">
      <h1>欢迎使用宏观周期分析系统</h1>
      <div class="actions">
        <button class="btn primary" @click="goSystem">进入系统分析页</button>
        <button class="btn" @click="goDatasets">我的数据集</button>
        <button class="btn" @click="goRuns">历史分析</button>
        <button v-if="me?.is_admin" class="btn admin" @click="goAdmin">管理员面板</button>
        <button class="btn" @click="logout">退出登录</button>
      </div>
    </header>

    <section class="card metrics">
      <div class="metric">
        <div class="metric-title">当前用户</div>
        <div class="metric-value">{{ me?.username || "-" }} <span v-if="me?.is_admin">（管理员）</span></div>
      </div>
      <div class="metric">
        <div class="metric-title">我的数据集</div>
        <div class="metric-value">{{ datasets.length }}</div>
      </div>
      <div class="metric">
        <div class="metric-title">历史分析</div>
        <div class="metric-value">{{ runs.length }}</div>
      </div>
    </section>

    <section class="card" v-if="runs.length">
      <h2>最近分析记录</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>run_id</th>
              <th>dataset_id</th>
              <th>预测模型</th>
              <th>创建时间</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="r in runs.slice(0, 10)" :key="r.id">
              <td>{{ r.id }}</td>
              <td>{{ r.dataset_id }}</td>
              <td>{{ r.forecast_model }}</td>
              <td>{{ formatTimeToMinute(r.created_at) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<script setup>
import axios from "axios";
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const me = ref(null);
const runs = ref([]);
const datasets = ref([]);

function authConfig() {
  const token = localStorage.getItem("eco_token") || "";
  return token ? { headers: { Authorization: `Bearer ${token}` } } : {};
}

async function load() {
  const [m, r, d] = await Promise.all([
    axios.get("/api/me", authConfig()),
    axios.get("/api/me/runs", authConfig()),
    axios.get("/api/me/datasets", authConfig()),
  ]);
  me.value = m.data;
  localStorage.setItem("eco_user", JSON.stringify(m.data || null));
  runs.value = r.data || [];
  datasets.value = d.data || [];
}

function goSystem() {
  router.push("/system");
}
function goDatasets() {
  router.push("/me/datasets");
}
function goRuns() {
  router.push("/me/runs");
}
function goAdmin() {
  router.push("/admin");
}

function formatTimeToMinute(v) {
  if (!v) return "-";
  const text = String(v).replace("T", " ");
  if (text.length >= 16) return text.slice(0, 16);
  return text;
}

function logout() {
  localStorage.removeItem("eco_token");
  localStorage.removeItem("eco_user");
  router.push("/login");
}

onMounted(async () => {
  try {
    await load();
  } catch (_e) {
    logout();
  }
});
</script>

<style scoped>
.home-page {
  min-height: 100vh;
  padding: 1.2rem;
  background: linear-gradient(165deg, #f0fdfa 0%, #f8fafc 45%, #eef2ff 100%);
}
.topbar {
  display: grid;
  gap: 0.8rem;
  margin-bottom: 1rem;
}
h1 {
  margin: 0;
  font-size: 1.35rem;
}
.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
}
.btn {
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 0.5rem 0.95rem;
  background: #fff;
  cursor: pointer;
}
.btn.primary {
  border: none;
  color: #fff;
  background: linear-gradient(135deg, #0d9488, #0f766e);
}
.btn.admin {
  border: 1px solid #fdba74;
  background: #fffbeb;
  color: #9a3412;
}
.card {
  background: #fff;
  border-radius: 14px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
  padding: 1rem;
  margin-bottom: 1rem;
}
.metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.8rem;
}
.metric {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 0.75rem;
}
.metric-title {
  color: #64748b;
  font-size: 0.82rem;
}
.metric-value {
  margin-top: 0.2rem;
  font-size: 1.2rem;
  font-weight: 700;
  color: #0f172a;
}
.meta {
  color: #475569;
  margin: 0.2rem 0;
}
@media (max-width: 920px) {
  .metrics {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
.table-wrap {
  overflow-x: auto;
}
table {
  width: 100%;
  border-collapse: collapse;
}
th,
td {
  padding: 0.55rem;
  border-bottom: 1px solid #f1f5f9;
  text-align: left;
}
</style>
