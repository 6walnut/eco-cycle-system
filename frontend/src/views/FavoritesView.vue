<template>
  <div class="page">
    <header class="top">
      <h1>我的收藏</h1>
      <div class="actions">
        <button class="btn" @click="goHome">返回主页</button>
        <button class="btn primary" @click="refresh">刷新</button>
      </div>
    </header>

    <section class="card">
      <div class="table-wrap" v-if="favorites.length">
        <table>
          <thead>
            <tr>
              <th>run_id</th>
              <th>dataset_id</th>
              <th>模型</th>
              <th>时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="f in favorites" :key="f.run_id">
              <td>{{ f.run_id }}</td>
              <td>{{ f.dataset_id }}</td>
              <td>{{ f.forecast_model }}</td>
              <td>{{ f.created_at }}</td>
              <td>
                <button class="btn mini" @click="viewRun(f.run_id)">查看详情</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="hint">暂无收藏</p>
    </section>

    <section class="card" v-if="detail">
      <h2>收藏详情（run_id={{ detail.id }}）</h2>
      <p class="hint">forecast_model: {{ detail.forecast_model }}</p>
      <pre class="json">{{ JSON.stringify(detail.result?.weights || {}, null, 2) }}</pre>
    </section>
  </div>
</template>

<script setup>
import axios from "axios";
import { onMounted, ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const favorites = ref([]);
const detail = ref(null);

function auth() {
  const token = localStorage.getItem("eco_token") || "";
  return { headers: { Authorization: `Bearer ${token}` } };
}

async function refresh() {
  const { data } = await axios.get("/api/me/favorites", auth());
  favorites.value = data || [];
}

async function viewRun(id) {
  const { data } = await axios.get(`/api/runs/${id}`, auth());
  detail.value = data;
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
.hint { color: #64748b; }
.json { background: #0f172a; color: #e2e8f0; padding: 0.7rem; border-radius: 10px; overflow-x: auto; font-size: 0.78rem; }
</style>
