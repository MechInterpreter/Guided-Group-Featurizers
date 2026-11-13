import type {Metadata} from 'next';
import './globals.css';
export const metadata:Metadata={title:'GGF Concept Atlas',description:'A private interactive exploration of visual concept groups, measured response geometry, and activation monitoring.',robots:{index:false,follow:false},icons:{icon:'/favicon.svg'}};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}
