import { createApp } from "vue";
import Root from "./Root.vue";
import router from "./router";

createApp(Root).use(router).mount("#app");
