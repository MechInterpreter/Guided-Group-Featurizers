import {env} from 'cloudflare:workers';
export async function POST(request:Request){
 const host=new URL(request.url).hostname;
 if(!['127.0.0.1','localhost'].includes(host))return Response.json({error:'Live inference is available on localhost only.'},{status:403});
 const origin=request.headers.get('origin');if(origin&&origin!==new URL(request.url).origin)return Response.json({error:'Cross-origin uploads are not allowed.'},{status:403});
 const vars=env as unknown as Record<string,string>;
 if(!vars.INFERENCE_ENDPOINT||!vars.INFERENCE_TOKEN)return Response.json({error:'The live model service is offline. Recorded exploration is still available.'},{status:503});
 try{
  const chunks:Uint8Array[]=[];let bytes=0;const reader=request.body?.getReader();if(!reader)return Response.json({error:'Choose an image to upload.'},{status:400});
  while(true){const {done,value}=await reader.read();if(done)break;bytes+=value.byteLength;if(bytes>9*1024*1024){await reader.cancel();return Response.json({error:'Choose an image smaller than 8 MB.'},{status:413})}chunks.push(value)}
  const body=new Uint8Array(bytes);let offset=0;for(const chunk of chunks){body.set(chunk,offset);offset+=chunk.length}
  const response=await fetch(vars.INFERENCE_ENDPOINT.replace(/\/$/,'')+'/infer',{method:'POST',headers:{Authorization:'Bearer '+vars.INFERENCE_TOKEN,'Content-Type':request.headers.get('content-type')??''},body,signal:AbortSignal.timeout(90000)});
  if(!response.headers.get('content-type')?.includes('application/json')) {
   await response.body?.cancel();
   const error=response.status===413?'This upload was rejected as too large. Choose an image up to 8 MB.':response.status===429?'The model service is busy. Please try again shortly.':'The model service returned an unexpected response. Please try again.';
   return Response.json({error},{status:response.ok?502:response.status,headers:{'Cache-Control':'no-store'}});
  }
  return new Response(response.body,{status:response.status,headers:{'Content-Type':'application/json','Cache-Control':'no-store'}});
 }catch{return Response.json({error:'The model service did not respond. Please try again.'},{status:502})}
}
