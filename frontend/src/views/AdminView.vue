<template>
  <div class="page">
    <header class="top">
      <h1>管理员面板</h1>
      <div class="actions">
        <button class="btn" @click="goLogin">返回登录界面</button>
        <button class="btn primary" @click="refreshAll">刷新全部</button>
      </div>
    </header>

    <section class="card">
      <h2>用户管理</h2>
      <div class="row">
        <input v-model="searchName" placeholder="按用户名查找" />
      </div>
      <div class="row">
        <input v-model="newUser.username" placeholder="用户名" />
        <input v-model="newUser.password" placeholder="密码" type="password" />
        <label><input v-model="newUser.is_admin" type="checkbox" /> 管理员</label>
        <button class="btn mini" :disabled="addingUser" @click="addUser">{{ addingUser ? "添加中..." : "添加用户" }}</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>ID</th><th>用户名</th><th>管理员</th><th>分析次数</th><th>操作</th></tr></thead>
          <tbody>
            <tr v-for="u in filteredUsers" :key="u.id">
              <td>{{ u.id }}</td>
              <td>{{ u.username }}</td>
              <td>{{ u.is_admin ? "是" : "否" }}</td>
              <td>{{ statMap[u.id] || 0 }}</td>
              <td><button class="btn mini" @click="delUser(u.id)">删除</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>模型管理</h2>
      <div class="row">
        <select v-model="modelForm.model_type">
          <option value="fusion">融合模型</option>
          <option value="forecast">预测模型</option>
        </select>
        <select v-model="modelForm.name">
          <option v-for="opt in currentModelNameOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
        <label><input v-model="modelForm.enabled" type="checkbox" /> 启用</label>
        <input v-model="modelForm.paramsText" placeholder='参数JSON，如 {"lookback":12}' />
        <button class="btn mini" @click="upsertModel">添加/更新</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>ID</th><th>类型</th><th>名称</th><th>启用</th><th>参数</th><th>操作</th></tr></thead>
          <tbody>
            <tr v-for="m in models" :key="m.id">
              <td>{{ m.id }}</td><td>{{ modelTypeText(m.model_type) }}</td><td>{{ modelNameText(m.model_type, m.name) }}</td>
              <td>{{ m.enabled ? "是" : "否" }}</td><td><code>{{ JSON.stringify(m.params) }}</code></td>
              <td><button class="btn mini" @click="delModel(m.id)">删除</button></td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section class="card">
      <h2>系统日志</h2>
      <div class="row">
        <input v-model.number="logLines" type="number" min="20" max="2000" />
        <button class="btn mini" @click="loadLogs">查看日志</button>
      </div>
      <div class="table-wrap" style="margin-top: 10px">
        <table>
          <thead><tr><th>日期</th><th>时间</th><th>用户</th><th>操作</th><th>详情</th></tr></thead>
          <tbody>
            <tr v-for="l in logs" :key="l.id">
              <td>{{ l.date }}</td>
              <td>{{ l.time }}</td>
              <td>{{ l.username || "-" }}</td>
              <td>{{ actionText(l.action) }}</td>
              <td>{{ l.detail || "-" }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<script setup>
import axios from "axios";
import { computed, onMounted, ref, watch } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const users = ref([]);
const stats = ref([]);
const models = ref([]);
const logs = ref([]);
const logLines = ref(200);
const searchName = ref("");
const addingUser = ref(false);
const newUser = ref({ username: "", password: "", is_admin: false });
const modelForm = ref({ model_type: "fusion", name: "pca", enabled: true, paramsText: "{}" });

const MODEL_TYPE_LABELS = {
  fusion: "融合模型",
  forecast: "预测模型",
};

const MODEL_NAME_OPTIONS = {
  fusion: [
    { value: "pca", label: "主成分分析（PCA）" },
    { value: "dfm", label: "动态因子模型（DFM）" },
    { value: "entropy", label: "熵权法" },
    { value: "equal", label: "等权法" },
  ],
  forecast: [
    { value: "hw", label: "霍尔特-温特斯（HW）" },
    { value: "lstm", label: "长短期记忆网络（LSTM）" },
  ],
};

function auth() {
  const token = localStorage.getItem("eco_token") || "";
  return { headers: { Authorization: `Bearer ${token}` } };
}

const statMap = computed(() => {
  const out = {};
  for (const s of stats.value) out[s.user_id] = s.run_count;
  return out;
});

const filteredUsers = computed(() => {
  const q = (searchName.value || "").trim().toLowerCase();
  if (!q) return users.value;
  return users.value.filter((u) => String(u.username || "").toLowerCase().includes(q));
});

async function refreshAll() {
  const [u, st, m] = await Promise.all([
    axios.get("/api/admin/users", auth()),
    axios.get("/api/admin/stats/runs", auth()),
    axios.get("/api/admin/models", auth()),
  ]);
  users.value = u.data || [];
  stats.value = st.data || [];
  models.value = m.data || [];
}

async function addUser() {
  const username = String(newUser.value.username || "").trim();
  const password = String(newUser.value.password || "").trim();
  if (!username || !password) {
    alert("请输入用户名和密码。");
    return;
  }
  addingUser.value = true;
  try {
    await axios.post("/api/admin/users", { ...newUser.value, username, password }, auth());
    newUser.value = { username: "", password: "", is_admin: false };
    await refreshAll();
  } catch (e) {
    alert(e.response?.data?.error || e.message || String(e));
  } finally {
    addingUser.value = false;
  }
}

async function delUser(id) {
  const ok = window.confirm("删除后将清空该用户后台所有数据，是否确认删除？");
  if (!ok) return;
  await axios.delete(`/api/admin/users/${id}`, auth());
  await refreshAll();
}

async function upsertModel() {
  if (!modelForm.value.name) {
    modelForm.value.name = currentModelNameOptions.value[0]?.value || "";
  }
  let params = {};
  try { params = JSON.parse(modelForm.value.paramsText || "{}"); } catch (_e) { params = {}; }
  await axios.post(
    "/api/admin/models",
    {
      model_type: modelForm.value.model_type,
      name: modelForm.value.name,
      enabled: modelForm.value.enabled,
      params,
    },
    auth(),
  );
  await refreshAll();
}

const currentModelNameOptions = computed(() => {
  return MODEL_NAME_OPTIONS[modelForm.value.model_type] || [];
});

function modelTypeText(modelType) {
  return MODEL_TYPE_LABELS[modelType] || modelType || "-";
}

function modelNameText(modelType, name) {
  const options = MODEL_NAME_OPTIONS[modelType] || [];
  const hit = options.find((x) => x.value === name);
  return hit?.label || name || "-";
}

watch(
  () => modelForm.value.model_type,
  (t) => {
    const first = (MODEL_NAME_OPTIONS[t] || [])[0]?.value || "";
    modelForm.value.name = first;
  },
);

async function delModel(id) {
  await axios.delete(`/api/admin/models/${id}`, auth());
  await refreshAll();
}

async function loadLogs() {
  const { data } = await axios.get(`/api/admin/logs?lines=${logLines.value}`, auth());
  logs.value = data.logs || [];
}

function actionText(action) {
  const m = {
    user_register: "用户注册",
    user_login: "用户登录",
    analyze_run: "执行分析",
    analyze_sina_run: "抓取新浪并分析",
    upload_dataset: "上传数据集",
    rerun_dataset: "复用数据重跑",
    user_delete_run: "删除分析记录",
    favorite_add: "添加收藏",
    favorite_remove: "取消收藏",
    share_create: "生成分享链接",
    admin_add_user: "管理员添加用户",
    admin_delete_user: "管理员删除用户",
    admin_set_config: "管理员修改配置",
    admin_upsert_model: "管理员新增/更新模型",
    admin_delete_model: "管理员删除模型",
  };
  return m[action] || "其他操作";
}

function goLogin() {
  localStorage.removeItem("eco_token");
  localStorage.removeItem("eco_user");
  router.push("/login");
}

onMounted(async () => {
  try {
    await refreshAll();
    await loadLogs();
  } catch (_e) {
    router.push("/login");
  }
});
</script>

<style scoped>
.page { min-height: 100vh; padding: 1.2rem; background: #f8fafc; }
.top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.actions, .row { display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; }
.card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.55rem; border-bottom: 1px solid #f1f5f9; text-align: left; }
.btn { border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.45rem 0.75rem; background: #fff; cursor: pointer; }
.btn.primary { background: #0d9488; color: #fff; border: none; }
.btn.mini { padding: 0.3rem 0.6rem; font-size: 0.82rem; }
input, select { border: 1px solid #cbd5e1; border-radius: 8px; padding: 0.45rem 0.55rem; min-width: 120px; }
code { font-size: 0.78rem; color: #475569; }
</style>
