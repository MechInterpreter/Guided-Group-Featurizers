import {env} from 'cloudflare:workers';
export async function GET(){
 const vars=env as unknown as Record<string,string>;
 if(!vars.INFERENCE_ENDPOINT||!vars.INFERENCE_TOKEN)return Response.json({ready:false},{headers:{'Cache-Control':'no-store'}});
 try{const response=await fetch(vars.INFERENCE_ENDPOINT.replace(/\/$/,'')+'/status',{headers:{Authorization:'Bearer '+vars.INFERENCE_TOKEN},signal:AbortSignal.timeout(4000)});if(!response.ok)return Response.json({ready:false},{headers:{'Cache-Control':'no-store'}});const result=await response.json() as any;return Response.json({ready:!!result.ready,device:result.device},{headers:{'Cache-Control':'no-store'}})}catch{return Response.json({ready:false},{headers:{'Cache-Control':'no-store'}})}
}
