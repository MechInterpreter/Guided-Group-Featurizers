import vinext from 'vinext';
import {defineConfig} from 'vite';
import {cloudflare} from '@cloudflare/vite-plugin';

export default defineConfig({
  server:{host:'127.0.0.1',port:5173,watch:{ignored:['**/.local/**','**/public/data/**','**/public/images/**','**/public/review-crops/**']}},
  plugins:[vinext(),cloudflare({
    viteEnvironment:{name:'rsc',childEnvironments:['ssr']},inspectorPort:false,
    config:{main:'./worker/index.ts',compatibility_date:'2026-05-15',compatibility_flags:['nodejs_compat']},
  })],
});
