const METHODS=['ggf','slics','supervised_vanilla','vanilla'];
export type Rgb=[number,number,number];

function hsl(hue:number,saturation:number,lightness:number):Rgb{
  const a=saturation*Math.min(lightness,1-lightness);
  const channel=(offset:number)=>{
    const k=(offset+hue/30)%12;
    return lightness-a*Math.max(-1,Math.min(k-3,9-k,1));
  };
  return [channel(0),channel(8),channel(4)];
}
const hex=(rgb:Rgb)=>'#'+rgb.map(value=>Math.round(value*255).toString(16).padStart(2,'0')).join('');

/** Display identity only: no semantic alignment between methods is inferred. */
export function groupColor(method:string,group:number){
  const methodIndex=METHODS.indexOf(method);
  if(methodIndex<0||!Number.isInteger(group)||group<0||group>=75)throw Error('Unknown method/group color identity');
  const hue=(22+(methodIndex*75+group)*137.50776405)%360;
  const rgb=hsl(hue,.76,.60);
  const active=rgb.map(value=>value*.76+.24) as Rgb;
  const context=rgb.map((value,index)=>value*.32+[.46,.57,.60][index]*.68) as Rgb;
  return {hex:hex(rgb),ink:hex(hsl(hue,.64,.32)),rgb,active,context};
}
