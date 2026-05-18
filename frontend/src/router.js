import { createRouter, createWebHistory } from "vue-router";
import App from "./App.vue";
import AdminView from "./views/AdminView.vue";
import HomeView from "./views/HomeView.vue";
import LoginView from "./views/LoginView.vue";
import UserDatasetsView from "./views/UserDatasetsView.vue";
import UserRunsView from "./views/UserRunsView.vue";

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/login" },
    { path: "/login", component: LoginView, meta: { public: true } },
    { path: "/home", component: HomeView },
    { path: "/system", component: App },
    { path: "/me/datasets", component: UserDatasetsView },
    { path: "/me/runs", component: UserRunsView },
    { path: "/admin", component: AdminView, meta: { admin: true } },
  ],
});

router.beforeEach((to) => {
  const token = localStorage.getItem("eco_token");
  const sharedToken = typeof to.query?.shared_token === "string" ? to.query.shared_token : "";
  const userRaw = localStorage.getItem("eco_user");
  let user = null;
  try {
    user = userRaw ? JSON.parse(userRaw) : null;
  } catch (_e) {
    user = null;
  }
  if (to.meta.public) {
    if (token && to.path === "/login") return user?.is_admin ? "/admin" : "/home";
    return true;
  }
  if (to.path === "/system" && sharedToken) return true;
  if (!token) return "/login";
  if (to.meta.admin && !user?.is_admin) return "/home";
  return true;
});

export default router;
