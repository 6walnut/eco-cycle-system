<template>
  <div class="login-page">
    <div class="login-card modern">
      <h1>宏观周期分析平台</h1>
      <p class="desc">请选择登录入口，或通过用户名注册新账号</p>

      <div class="mode-tabs">
        <button class="tab" :class="{ active: loginMode === 'user' }" @click="switchMode('user')">用户登录</button>
        <button class="tab" :class="{ active: loginMode === 'admin' }" @click="switchMode('admin')">管理员登录</button>
        <button class="tab" :class="{ active: loginMode === 'register' }" @click="openRegisterDialog">注册账号</button>
      </div>

      <template v-if="loginMode === 'user' || loginMode === 'admin'">
        <form @submit.prevent="login">
          <div class="field">
            <label>{{ loginMode === "admin" ? "管理员用户名" : "用户名" }}</label>
            <input v-model="loginForm.username" class="input short" placeholder="请输入用户名" @keydown.enter.prevent="login" />
          </div>
          <div class="field">
            <label>密码</label>
            <input v-model="loginForm.password" class="input short" type="password" placeholder="请输入密码" @keydown.enter.prevent="login" />
          </div>
          <div class="btn-row">
            <button type="submit" class="btn primary">{{ loginMode === "admin" ? "管理员登录" : "登录" }}</button>
          </div>
        </form>
      </template>

      <template v-else>
        <div class="field">
          <label>注册提示</label>
          <p class="desc-mini">点击下方按钮弹出注册窗口，输入用户名与密码完成注册</p>
        </div>
        <div class="btn-row">
          <button class="btn primary" @click="openRegisterDialog">打开注册弹窗</button>
        </div>
      </template>

      <p v-if="error" class="error">{{ error }}</p>
    </div>

    <div v-if="showRegisterDialog" class="modal-mask" @click.self="showRegisterDialog = false">
      <div class="modal-card">
        <h3>完善注册信息</h3>
        <p class="desc-mini">请输入用户名与密码。用户名需唯一且至少3位。</p>
        <form @submit.prevent="register">
          <div class="field">
            <label>用户名</label>
            <input v-model="registerForm.username" class="input short register-short" placeholder="3位以上用户名" @keydown.enter.prevent="register" />
          </div>
          <div class="field">
            <label>密码</label>
            <input v-model="registerForm.password" class="input short register-short" type="password" placeholder="至少6位密码" @keydown.enter.prevent="register" />
          </div>
          <p v-if="error" class="error">{{ error }}</p>
          <div class="btn-row">
            <button type="submit" class="btn primary">确认注册</button>
            <button type="button" class="btn" @click="showRegisterDialog = false">取消</button>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup>
import axios from "axios";
import { ref } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const loginMode = ref("user");
const loginForm = ref({ username: "", password: "" });
const registerForm = ref({ username: "", password: "" });
const showRegisterDialog = ref(false);
const error = ref("");

function switchMode(m) {
  loginMode.value = m;
  error.value = "";
}

function openRegisterDialog() {
  loginMode.value = "register";
  error.value = "";
  showRegisterDialog.value = true;
}

async function login() {
  error.value = "";
  try {
    const { data } = await axios.post("/api/auth/login", {
      ...loginForm.value,
      role: loginMode.value === "admin" ? "admin" : "user",
    });
    localStorage.setItem("eco_token", data.token);
    localStorage.setItem("eco_user", JSON.stringify(data.user || null));
    router.push(loginMode.value === "admin" ? "/admin" : "/home");
  } catch (e) {
    error.value = e.response?.data?.error || e.message || String(e);
  }
}

async function register() {
  error.value = "";
  if (!registerForm.value.username || !registerForm.value.password) {
    error.value = "请填写用户名和密码。";
    return;
  }
  if (!/^[A-Za-z0-9_\u4e00-\u9fa5]{3,32}$/.test(registerForm.value.username || "")) {
    error.value = "用户名需为3-32位（中文/字母/数字/下划线）。";
    return;
  }
  try {
    const { data } = await axios.post("/api/auth/register", {
      username: registerForm.value.username,
      password: registerForm.value.password,
    });
    localStorage.setItem("eco_token", data.token);
    localStorage.setItem("eco_user", JSON.stringify(data.user || null));
    showRegisterDialog.value = false;
    router.push("/home");
  } catch (e) {
    error.value = e.response?.data?.error || e.message || String(e);
  }
}
</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: grid;
  place-items: center;
  background: linear-gradient(165deg, #f0fdfa 0%, #f8fafc 45%, #eef2ff 100%);
}
.login-card {
  width: clamp(260px, 88vw, 360px);
  background: #fff;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
  padding: 1.4rem;
}
h1 {
  margin: 0 0 0.25rem;
  font-size: 1.25rem;
}
.desc {
  margin: 0 0 1rem;
  color: #64748b;
  font-size: 0.9rem;
}
.login-card.modern {
  border-radius: 18px;
}
.mode-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: 0.45rem;
  margin-bottom: 0.8rem;
}
.tab {
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 0.42rem 0.6rem;
  background: #fff;
  cursor: pointer;
}
.tab.active {
  border-color: transparent;
  color: #fff;
  background: linear-gradient(135deg, #0d9488, #0f766e);
}
.field {
  margin-bottom: 0.75rem;
}
.field label {
  display: block;
  margin-bottom: 0.35rem;
  color: #64748b;
  font-size: 0.8rem;
}
.input {
  width: 100%;
  border: 1px solid #cbd5e1;
  border-radius: 10px;
  padding: 0.52rem 0.65rem;
  background: #f8fafc;
}
.input.short {
  max-width: 240px;
}
.register-short {
  max-width: 210px;
}
.btn-row {
  display: flex;
  gap: 0.6rem;
  margin-top: 0.8rem;
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
.error {
  color: #b91c1c;
  font-size: 0.88rem;
  margin-top: 0.35rem;
}
.modal-mask {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.42);
  display: grid;
  place-items: center;
  padding: 1rem;
}
.modal-card {
  width: min(420px, 92vw);
  background: #fff;
  border-radius: 14px;
  border: 1px solid #e2e8f0;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.18);
  padding: 1rem;
}
.modal-card h3 {
  margin: 0 0 0.45rem;
}
.desc-mini {
  margin: 0 0 0.65rem;
  color: #64748b;
  font-size: 0.84rem;
}
@media (max-width: 420px) {
  .input.short {
    max-width: 100%;
  }
  .mode-tabs {
    grid-template-columns: 1fr;
  }
}
</style>
