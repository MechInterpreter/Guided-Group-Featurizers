export type UploadView={
  reviewMode?:'pairs'|'explore';pairCursor?:number;
  selection:{method:string;group:number};
  highlightedGroups:Record<string,number>;
  patch:number;imageId:number;
  geometry:'native'|'umap';evidenceSet:'review'|'reference';activityView:boolean;
};

export type CachedUpload={
  key:string;number:number;name:string;live:any;view:UploadView|null;
};

/** Own uploaded image URLs until explicit removal or the page session ends. */
export class UploadSession{
  private entries=new Map<string,CachedUpload>();
  private sequence=0;

  get(key:string){return this.entries.get(key)}
  list(){return [...this.entries.values()]}

  add(key:string,file:File,result:any):CachedUpload{
    const existing=this.entries.get(key);
    if(existing)return existing;
    if(!/^[a-f0-9]{64}$/.test(key)||result.input_sha256!==key)throw Error('The upload result does not match this image. Please try again.');
    const entry:CachedUpload={key,number:++this.sequence,name:file.name,view:null,
      live:{...result,image:{id:file.name,src:URL.createObjectURL(file),geometry:result.geometry}}};
    this.entries.set(key,entry);
    return entry;
  }

  saveView(key:string,view:UploadView){
    const entry=this.entries.get(key);
    if(entry)entry.view={...view,selection:{...view.selection},highlightedGroups:{...view.highlightedGroups}};
  }

  remove(key:string){
    const entry=this.entries.get(key);
    if(!entry)return;
    URL.revokeObjectURL(entry.live.image.src);
    this.entries.delete(key);
  }

  clear(){for(const key of this.entries.keys())this.remove(key)}
}

export async function uploadDigest(file:File){
  const digest=await crypto.subtle.digest('SHA-256',await file.arrayBuffer());
  return [...new Uint8Array(digest)].map(value=>value.toString(16).padStart(2,'0')).join('');
}
