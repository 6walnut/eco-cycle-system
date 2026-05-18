<template>
  <div class="page">
    <header class="top">
      <h1>历史分析</h1>
      <div class="actions">
        <button class="btn" @click="goHome">返回主页</button>
        <button class="btn primary" @click="refresh">刷新</button>
      </div>
    </header>

    <section class="card">
      <div class="table-wrap" v-if="runs.length">
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
            <tr v-for="r in runs" :key="r.id">
              <td>{{ r.id }}</td>
              <td>{{ r.dataset_id }}</td>
              <td>{{ r.forecast_model }}</td>
              <td>{{ formatTimeToMinute(r.created_at) }}</td>
              <td class="ops">
                <button class="btn mini" @click="viewRun(r.id)">查看详情</button>
                <button class="btn mini" @click="exportRunPdf(r.id)">导出PDF</button>
                <button class="btn mini danger" @click="removeRun(r.id)">删除</button>
                <button class="btn mini" @click="toggleSelect(r.id)">
                  {{ selected.includes(r.id) ? "取消对比" : "加入对比" }}
                </button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-else class="hint">暂无分析任务</p>
      <div class="actions" style="margin-top: 10px">
        <button class="btn primary" :disabled="selected.length !== 2" @click="compare">对比两次分析</button>
      </div>
    </section>

    <section class="card" v-if="compareResult">
      <h2>对比结果（权重差异 B-A）</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>指标</th><th>差异</th></tr>
          </thead>
          <tbody>
            <tr v-for="r in compareRows" :key="r.key">
              <td>{{ r.key }}</td>
              <td>{{ r.delta.toFixed(6) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<script setup>
import axios from "axios";
import { computed, onMounted, ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const runs = ref([]);
const selected = ref([]);
const compareResult = ref(null);

function auth() {
  const token = localStorage.getItem("eco_token") || "";
  return { headers: { Authorization: `Bearer ${token}` } };
}

async function refresh() {
  const { data } = await axios.get("/api/me/runs", auth());
  runs.value = data || [];
}

function toggleSelect(id) {
  if (selected.value.includes(id)) {
    selected.value = selected.value.filter((x) => x !== id);
    return;
  }
  if (selected.value.length >= 2) selected.value = [selected.value[1], id];
  else selected.value = [...selected.value, id];
}

async function viewRun(id) {
  router.push(`/system?run_id=${id}`);
}

async function compare() {
  if (selected.value.length !== 2) return;
  const [a, b] = selected.value;
  const { data } = await axios.get(`/api/runs/compare?run_id_a=${a}&run_id_b=${b}`, auth());
  compareResult.value = data;
}

async function exportRunPdf(id) {
  try {
    const resp = await axios.get(`/api/runs/${id}/export/pdf`, {
      ...auth(),
      responseType: "blob",
    });
    const blob = new Blob([resp.data], { type: "application/pdf" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis_${id}.pdf`;
    a.click();
    window.URL.revokeObjectURL(url);
  } catch (e) {
    alert(e.response?.data?.error || "导出PDF失败，请稍后重试。");
  }
}

async function removeRun(id) {
  const ok = window.confirm("确认删除这条历史分析记录吗？删除后不可恢复。");
  if (!ok) return;
  try {
    await axios.delete(`/api/runs/${id}`, auth());
    selected.value = selected.value.filter((x) => x !== id);
    compareResult.value = null;
    await refresh();
  } catch (e) {
    alert(e.response?.data?.error || "删除失败，请稍后重试。");
  }
}

const compareRows = computed(() => {
  const delta = compareResult.value?.weight_delta || {};
  return Object.entries(delta).map(([key, d]) => ({ key, delta: Number(d || 0) }));
});

function goHome() {
  router.push("/home");
}

function formatTimeToMinute(v) {
  if (!v) return "-";
  const text = String(v).replace("T", " ");
  if (text.length >= 16) return text.slice(0, 16);
  return text;
}

onMounted(refresh);
</script>

<style scoped>
.page { min-height: 100vh; padding: 1.2rem; background: #f8fafc; }
.top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.actions { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.55rem; border-bottom: 1px solid #f1f5f9; text-align: left; }
.btn { border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.45rem 0.75rem; background: #fff; cursor: pointer; }
.btn.primary { background: #0d9488; color: #fff; border: none; }
.btn.mini { padding: 0.3rem 0.6rem; font-size: 0.82rem; }
.btn.danger { color: #b91c1c; border-color: #fecaca; background: #fff1f2; }
.ops { display: flex; gap: 0.4rem; }
.hint { color: #64748b; }
</style>
