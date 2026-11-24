"""Interactive 3D page of the elf's left-leg flesh run (elf_leg_flesh.py): skin per saved frame, coloured by stretch, with the
fitted bones moving inside.  One self-contained HTML file (three.js from a CDN), opened locally.
  E:\\splat\\.venv\\Scripts\\python.exe leg_viewer.py <fits dir> <rig_stage dir> <out.html>"""
import sys, os, json, base64, numpy as np, re
FD, RS, OUT = sys.argv[1:4]
D = np.load(os.path.join(RS, 'elf_leg_flesh.npz')); S, F, X0 = D['skin'], D['faces'].astype(np.uint32), D['rest']
TR = np.load(os.path.join(RS, 'sit_leg_transforms.npz')); step = 15                       # elf_leg_flesh.py keeps every 15th frame
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'], R['obj'], R['F_bone']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
foot = [i for i, n in enumerate(names) if n.endswith('.l') and re.search(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan', n)]
groups = {0: [nid['Hip bone.l']], 1: [nid['Femur.l']], 2: [nid['Tibia.l'], nid['Fibula.l'], nid['Patella.l']] + foot}
bones = []
for g, ids in groups.items():
    fb = FB[np.isin(OB[FB[:, 0]], ids)]; u, inv = np.unique(fb.ravel(), return_inverse=True)
    P = V[u].astype(np.float64); frames = []
    for k in range(len(S)):
        if g == 0: frames.append(P)
        else:
            Rm, t = (TR['thigh_R'], TR['thigh_t']) if g == 1 else (TR['shank_R'], TR['shank_t']); f = min(k * step, len(Rm) - 1); frames.append(P @ Rm[f].T + t[f])
    bones.append(dict(F=inv.reshape(-1, 3).astype(np.uint32), X=np.array(frames, np.float32)))
# stretch per skin vertex per frame (largest edge ratio), and flipped faces
E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; e0 = np.linalg.norm(X0[E[:, 0]] - X0[E[:, 1]], axis=1); ok = e0 > 0
n0 = np.cross(X0[F[:, 1]] - X0[F[:, 0]], X0[F[:, 2]] - X0[F[:, 0]]); ST, FL = [], []
for X in S:
    r = np.ones(len(E)); r[ok] = np.linalg.norm(X[E[ok, 0]] - X[E[ok, 1]], axis=1) / e0[ok]; s = np.ones(len(X0))
    np.maximum.at(s, E[:, 0], np.maximum(r, 1 / np.maximum(r, 1e-6))); np.maximum.at(s, E[:, 1], np.maximum(r, 1 / np.maximum(r, 1e-6))); ST.append(s)
    fl = (n0 * np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])).sum(1) < 0; FL.append(float(fl.mean()))
ST = np.array(ST, np.float32)
allP = np.concatenate([S.reshape(-1, 3)] + [b['X'].reshape(-1, 3) for b in bones]); lo, hi = allP.min(0), allP.max(0); sc = (hi - lo).max()
def q(X): return np.round((X - lo) / sc * 65535).astype(np.uint16)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
data = dict(lo=lo.tolist(), sc=float(sc), nV=int(len(X0)), nF=len(S), step=step, fps=float(TR['fps']), faces=b64(F), skin=b64(q(S)),
            stretch=b64(np.clip(np.log2(ST) / 2, 0, 1).astype(np.float16).view(np.uint16)), flipped=FL,
            hip=[float(np.degrees(0))], bones=[dict(nV=int(b['X'].shape[1]), faces=b64(b['F']), X=b64(q(b['X']))) for b in bones])
A = json.load(open(os.path.join(RS, 'ardy_to_elf_leg_calibrated.json')))
html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Elf leg flesh</title>
<style>
:root{--bg:#f4f2ee;--fg:#222;--panel:rgba(255,255,255,.85)}
@media (prefers-color-scheme:dark){:root{--bg:#16181c;--fg:#e8e6e1;--panel:rgba(30,32,38,.85)}}
html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;overflow:hidden}
#ui{position:absolute;left:12px;top:12px;right:12px;max-width:520px;background:var(--panel);padding:10px 12px;border-radius:8px}
#ui input[type=range]{width:100%} label{margin-right:12px;white-space:nowrap} .k{opacity:.7;font-size:12px}
</style></head><body>
<div id="ui"><b>Elf left leg — Newton flesh, ARDY “sit” clip</b>
<div><input id="t" type="range" min="0" max="1000" value="0"></div>
<div><button id="play">Pause</button> <span id="info"></span></div>
<div><label><input type="checkbox" id="cs" checked> skin</label><label><input type="checkbox" id="cb" checked> bones</label>
<label><input type="checkbox" id="cc" checked> colour by stretch</label><label><input type="checkbox" id="cw"> wireframe</label></div>
<div class="k">Drag to orbit, scroll to zoom. Colour: white = unstretched, red = stretched or squashed 4× or more. Skin from 12 saved frames, blended between them.</div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three'; import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D = __DATA__;
const dec=(s,T)=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)};
const h2f=h=>{const s=(h&0x8000)?-1:1,e=(h>>10)&31,m=h&1023;return e==0?s*Math.pow(2,-14)*(m/1024):e==31?NaN:s*Math.pow(2,e-15)*(1+m/1024)};
const deq=(a)=>{const o=new Float32Array(a.length);for(let i=0;i<a.length;i+=3){o[i]=D.lo[0]+a[i]/65535*D.sc;o[i+1]=D.lo[1]+a[i+1]/65535*D.sc;o[i+2]=D.lo[2]+a[i+2]/65535*D.sc}return o};
const r=new THREE.WebGLRenderer({antialias:true});r.setPixelRatio(devicePixelRatio);r.setSize(innerWidth,innerHeight);document.body.appendChild(r.domElement);
const dark=matchMedia('(prefers-color-scheme: dark)').matches; const scene=new THREE.Scene();scene.background=new THREE.Color(dark?0x16181c:0xf4f2ee);
const cam=new THREE.PerspectiveCamera(35,innerWidth/innerHeight,0.01,20);cam.up.set(0,0,1);
scene.add(new THREE.HemisphereLight(0xffffff,0x555566,1.6));const dl=new THREE.DirectionalLight(0xffffff,1.6);dl.position.set(-1,-2,3);scene.add(dl);
const faces=dec(D.faces,Uint32Array), skin=deq(dec(D.skin,Uint16Array)), st=dec(D.stretch,Uint16Array), nV=D.nV;
const g=new THREE.BufferGeometry();const pos=new Float32Array(nV*3),col=new Float32Array(nV*3);
g.setAttribute('position',new THREE.BufferAttribute(pos,3));g.setAttribute('color',new THREE.BufferAttribute(col,3));g.setIndex(new THREE.BufferAttribute(faces,1));
const sm=new THREE.MeshStandardMaterial({vertexColors:true,roughness:.7,side:THREE.DoubleSide,transparent:true,opacity:.55,depthWrite:false});
const skinMesh=new THREE.Mesh(g,sm);scene.add(skinMesh);
const boneMeshes=D.bones.map(b=>{const bg=new THREE.BufferGeometry();const bp=new Float32Array(b.nV*3);bg.setAttribute('position',new THREE.BufferAttribute(bp,3));
 bg.setIndex(new THREE.BufferAttribute(dec(b.faces,Uint32Array),1));const m=new THREE.Mesh(bg,new THREE.MeshStandardMaterial({color:0xe9e1cf,roughness:.5}));scene.add(m);
 return {m,bg,bp,X:deq(dec(b.X,Uint16Array)),nV:b.nV}});
const stf=new Float32Array(st.length);for(let i=0;i<st.length;i++)stf[i]=h2f(st[i]);
function setT(u){const x=u*(D.nF-1),k=Math.min(Math.floor(x),D.nF-2),a=x-k;
 for(let i=0;i<nV*3;i++)pos[i]=skin[k*nV*3+i]*(1-a)+skin[(k+1)*nV*3+i]*a;
 const cc=document.getElementById('cc').checked;
 for(let i=0;i<nV;i++){const s=cc?stf[k*nV+i]*(1-a)+stf[(k+1)*nV+i]*a:0;col[3*i]=1;col[3*i+1]=1-.85*s;col[3*i+2]=1-.9*s}
 g.attributes.position.needsUpdate=true;g.attributes.color.needsUpdate=true;g.computeVertexNormals();
 for(const b of boneMeshes){const n=b.nV*3;for(let i=0;i<n;i++)b.bp[i]=b.X[k*n+i]*(1-a)+b.X[(k+1)*n+i]*a;b.bg.attributes.position.needsUpdate=true;b.bg.computeVertexNormals()}
 const f=Math.round(x*D.step);document.getElementById('info').textContent=`t = ${(f/D.fps).toFixed(2)} s · flipped skin ${(100*(D.flipped[k]*(1-a)+D.flipped[k+1]*a)).toFixed(1)}%`}
g.computeBoundingSphere();const c0=new THREE.Vector3(),bb=new THREE.Box3().setFromArray(skin.subarray(0,nV*3));bb.getCenter(c0);const R0=bb.getSize(new THREE.Vector3()).length();
cam.position.set(c0.x+R0*1.4,c0.y-R0*1.6,c0.z+R0*.4);const ctl=new OrbitControls(cam,r.domElement);ctl.target.copy(c0);ctl.update();
const sl=document.getElementById('t');let playing=true,u=0;document.getElementById('play').onclick=e=>{playing=!playing;e.target.textContent=playing?'Pause':'Play'};
sl.oninput=()=>{u=sl.value/1000;setT(u)};
document.getElementById('cs').onchange=e=>skinMesh.visible=e.target.checked;document.getElementById('cb').onchange=e=>boneMeshes.forEach(b=>b.m.visible=e.target.checked);
document.getElementById('cw').onchange=e=>{sm.wireframe=e.target.checked};document.getElementById('cc').onchange=()=>setT(u);
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();r.setSize(innerWidth,innerHeight)});
let last=performance.now();setT(0);
r.setAnimationLoop(now=>{if(playing){u+=(now-last)/1000/((D.nF-1)*D.step/D.fps);if(u>1)u=0;sl.value=Math.round(u*1000);setT(u)}last=now;r.render(scene,cam)});
</script></body></html>"""
open(OUT, 'w', encoding='utf-8').write(html.replace('__DATA__', json.dumps(data)))
print('wrote %s (%.1f MB)' % (OUT, os.path.getsize(OUT) / 1e6))
