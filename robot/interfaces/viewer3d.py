"""Self-contained WebGL2 3D editor viewer for the live dashboard (zero external JS dependencies)."""

from __future__ import annotations

VIEWER_JS = r"""(() => {
const root = document.getElementById('view3d-wrap');
const glCanvas = document.getElementById('view3d');
const lblCanvas = document.getElementById('labels3d');
if (!root || !glCanvas || !lblCanvas) return;

const gl = glCanvas.getContext('webgl2', {antialias: true, alpha: false});
if (!gl) {
  if (typeof window.setMapMode === 'function') window.setMapMode('2d');
  const btn3d = document.getElementById('mode-3d');
  if (btn3d) btn3d.disabled = true;
  return;
}

// ---- matrix & quaternion helpers (column-major 4x4) ------------------------------------------------
const M = {
  id() { return new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]); },
  mul(a, b) {
    const o = new Float32Array(16);
    for (let c = 0; c < 4; c++) {
      const b0 = b[c*4], b1 = b[c*4+1], b2 = b[c*4+2], b3 = b[c*4+3];
      o[c*4]   = a[0]*b0 + a[4]*b1 + a[8]*b2  + a[12]*b3;
      o[c*4+1] = a[1]*b0 + a[5]*b1 + a[9]*b2  + a[13]*b3;
      o[c*4+2] = a[2]*b0 + a[6]*b1 + a[10]*b2 + a[14]*b3;
      o[c*4+3] = a[3]*b0 + a[7]*b1 + a[11]*b2 + a[15]*b3;
    }
    return o;
  },
  transpose(m) {
    return new Float32Array([
      m[0],m[4],m[8],m[12], m[1],m[5],m[9],m[13], m[2],m[6],m[10],m[14], m[3],m[7],m[11],m[15]
    ]);
  },
  trs(t, q, s) {
    const [x, y, z, w] = q || [0,0,0,1];
    const [sx, sy, sz] = s || [1,1,1];
    const [tx, ty, tz] = t || [0,0,0];
    const x2 = x+x, y2 = y+y, z2 = z+z;
    const xx = x*x2, xy = x*y2, xz = x*z2, yy = y*y2, yz = y*z2, zz = z*z2;
    const wx = w*x2, wy = w*y2, wz = w*z2;
    return new Float32Array([
      (1-(yy+zz))*sx, (xy+wz)*sx,     (xz-wy)*sx,     0,
      (xy-wz)*sy,     (1-(xx+zz))*sy, (yz+wx)*sy,     0,
      (xz+wy)*sz,     (yz-wx)*sz,     (1-(xx+yy))*sz, 0,
      tx, ty, tz, 1
    ]);
  },
  fromPosYawScale(x, y, z, yaw, sx, sy, sz) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    return new Float32Array([c*sx, s*sx, 0, 0, -s*sy, c*sy, 0, 0, 0, 0, sz, 0, x, y, z, 1]);
  },
  rotX(a) {
    const c = Math.cos(a), s = Math.sin(a);
    return new Float32Array([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]);
  },
  perspective(fovy, aspect, near, far) {
    const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
  },
  lookAt(eye, ctr, up) {
    let fx = ctr[0]-eye[0], fy = ctr[1]-eye[1], fz = ctr[2]-eye[2];
    const fl = Math.hypot(fx, fy, fz) || 1;
    fx /= fl; fy /= fl; fz /= fl;
    let sx = fy*up[2] - fz*up[1], sy = fz*up[0] - fx*up[2], sz = fx*up[1] - fy*up[0];
    const sl = Math.hypot(sx, sy, sz) || 1;
    sx /= sl; sy /= sl; sz /= sl;
    const ux = sy*fz - sz*fy, uy = sz*fx - sx*fz, uz = sx*fy - sy*fx;
    return new Float32Array([
      sx, ux, -fx, 0, sy, uy, -fy, 0, sz, uz, -fz, 0,
      -(sx*eye[0]+sy*eye[1]+sz*eye[2]), -(ux*eye[0]+uy*eye[1]+uz*eye[2]),
      (fx*eye[0]+fy*eye[1]+fz*eye[2]), 1
    ]);
  },
  normalMat3(m) {
    const a00=m[0], a01=m[1], a02=m[2], a10=m[4], a11=m[5], a12=m[6], a20=m[8], a21=m[9], a22=m[10];
    const b01 = a22*a11 - a12*a21, b11 = -a22*a10 + a12*a20, b21 = a21*a10 - a11*a20;
    const det = a00*b01 + a01*b11 + a02*b21;
    if (Math.abs(det) < 1e-8) return new Float32Array([1,0,0, 0,1,0, 0,0,1]);
    const id = 1 / det;
    return new Float32Array([
      b01*id, b11*id, b21*id,
      (-a22*a01+a02*a21)*id, (a22*a00-a02*a20)*id, (-a21*a00+a01*a20)*id,
      (a12*a01-a02*a11)*id, (-a12*a00+a02*a10)*id, (a11*a00-a01*a10)*id
    ]);
  },
};

// ---- shaders ---------------------------------------------------------------------------------------
function makeProgram(vsSrc, fsSrc) {
  const compile = (type, src) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(sh));
    return sh;
  };
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  return p;
}

const meshProg = makeProgram(`#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in vec2 aUV;
uniform mat4 uProj, uView, uModel;
uniform mat3 uNormMat;
uniform vec2 uTexOffset, uTexScale;
uniform float uTexRot;
out vec3 vWorldPos, vNorm;
out vec2 vUV;
void main() {
  vec4 wp = uModel * vec4(aPos, 1.0);
  vWorldPos = wp.xyz;
  vNorm = uNormMat * aNorm;
  float c = cos(uTexRot), s = sin(uTexRot);
  vec2 uv = aUV * uTexScale;
  vUV = vec2(c*uv.x - s*uv.y, s*uv.x + c*uv.y) + uTexOffset;
  gl_Position = uProj * uView * wp;
}`, `#version 300 es
precision highp float;
in vec3 vWorldPos, vNorm;
in vec2 vUV;
uniform sampler2D uTex;
uniform int uHasTex, uUnlit;
uniform vec4 uColor;
uniform float uCutZ;
out vec4 fragColor;
void main() {
  if (uCutZ < 90.0 && vWorldPos.z > uCutZ) discard;
  vec4 base = uColor;
  if (uHasTex == 1) base *= texture(uTex, vUV);
  if (uUnlit == 1) { fragColor = base; return; }
  vec3 n = normalize(vNorm);
  if (!gl_FrontFacing) n = -n;
  float hemi = 0.5 + 0.5 * n.z;
  vec3 amb = mix(vec3(0.42, 0.40, 0.38), vec3(0.76, 0.78, 0.82), hemi);
  float d1 = max(dot(n, normalize(vec3(0.45, 0.55, 0.72))), 0.0);
  float d2 = max(dot(n, normalize(vec3(-0.6, -0.35, 0.5))), 0.0);
  vec3 lit = amb + vec3(0.38) * d1 + vec3(0.18) * d2;
  fragColor = vec4(base.rgb * lit, base.a);
}`);

const lineProg = makeProgram(`#version 300 es
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); }`, `#version 300 es
precision highp float;
uniform vec4 uColor;
out vec4 fragColor;
void main() { fragColor = uColor; }`);

const U = {};
for (const k of ['uProj','uView','uModel','uNormMat','uTexOffset','uTexScale','uTexRot','uTex','uHasTex','uUnlit','uColor','uCutZ']) {
  U[k] = gl.getUniformLocation(meshProg, k);
}
const UL = { uMVP: gl.getUniformLocation(lineProg, 'uMVP'), uColor: gl.getUniformLocation(lineProg, 'uColor') };

// ---- primitive geometries --------------------------------------------------------------------------
function makeMesh(pos, norm, uv, idx) {
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vbo = (loc, arr, size) => {
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, arr, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  };
  vbo(0, pos, 3);
  vbo(1, norm, 3);
  vbo(2, uv, 2);
  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, idx, gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, count: idx.length, idxType: idx instanceof Uint32Array ? gl.UNSIGNED_INT : gl.UNSIGNED_SHORT };
}

function makeBox() {
  const p=[], n=[], u=[], ix=[];
  const faces = [
    [[0,0,1], [1,0,0], [0,1,0]], [[0,0,-1], [-1,0,0], [0,1,0]],
    [[1,0,0], [0,1,0], [0,0,1]], [[-1,0,0], [0,-1,0], [0,0,1]],
    [[0,1,0], [-1,0,0], [0,0,1]], [[0,-1,0], [1,0,0], [0,0,1]],
  ];
  faces.forEach(([fn, fu, fv], f) => {
    const base = f * 4;
    [[-1,-1],[1,-1],[1,1],[-1,1]].forEach(([su, sv]) => {
      p.push((fn[0]+su*fu[0]+sv*fv[0])*0.5, (fn[1]+su*fu[1]+sv*fv[1])*0.5, (fn[2]+su*fu[2]+sv*fv[2])*0.5);
      n.push(fn[0], fn[1], fn[2]);
      u.push((su+1)*0.5, (sv+1)*0.5);
    });
    ix.push(base, base+1, base+2, base, base+2, base+3);
  });
  return makeMesh(new Float32Array(p), new Float32Array(n), new Float32Array(u), new Uint16Array(ix));
}

function makeCyl(segs = 24) {
  const p=[], n=[], u=[], ix=[];
  for (let i = 0; i <= segs; i++) {
    const a = (i / segs) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
    p.push(c, s, -0.5, c, s, 0.5);
    n.push(c, s, 0, c, s, 0);
    u.push(i / segs, 0, i / segs, 1);
  }
  for (let i = 0; i < segs; i++) {
    const b = i * 2;
    ix.push(b, b+2, b+3, b, b+3, b+1);
  }
  for (const [z, nz] of [[0.5, 1], [-0.5, -1]]) {
    const ctr = p.length / 3;
    p.push(0, 0, z); n.push(0, 0, nz); u.push(0.5, 0.5);
    for (let i = 0; i <= segs; i++) {
      const a = (i / segs) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
      p.push(c, s, z); n.push(0, 0, nz); u.push(0.5+0.5*c, 0.5+0.5*s);
    }
    for (let i = 0; i < segs; i++) {
      if (nz > 0) ix.push(ctr, ctr+1+i, ctr+2+i);
      else ix.push(ctr, ctr+2+i, ctr+1+i);
    }
  }
  return makeMesh(new Float32Array(p), new Float32Array(n), new Float32Array(u), new Uint16Array(ix));
}

function makeQuad() {
  return makeMesh(
    new Float32Array([-0.5,-0.5,0, 0.5,-0.5,0, 0.5,0.5,0, -0.5,0.5,0]),
    new Float32Array([0,0,1, 0,0,1, 0,0,1, 0,0,1]),
    new Float32Array([0,1, 1,1, 1,0, 0,0]),
    new Uint16Array([0,1,2, 0,2,3]),
  );
}

function makeFrustumMesh(hfov, aspect, len) {
  const w = len * Math.tan(hfov / 2), h = w / aspect;
  const pts = [0,0,0, len,w,h, len,-w,h, len,-w,-h, len,w,-h];
  const tri = new Float32Array([
    0,0,0, len,w,h, len,-w,h,
    0,0,0, len,-w,h, len,-w,-h,
    0,0,0, len,-w,-h, len,w,-h,
    0,0,0, len,w,-h, len,w,h,
  ]);
  const nrm = new Float32Array(36);
  const uv = new Float32Array(24);
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const b = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.bufferData(gl.ARRAY_BUFFER, tri, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  const nb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, nb);
  gl.bufferData(gl.ARRAY_BUFFER, nrm, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 0, 0);
  const ub = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, ub);
  gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(2); gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 0, 0);

  const wire = new Float32Array([
    0,0,0, len,w,h,  0,0,0, len,-w,h,  0,0,0, len,-w,-h,  0,0,0, len,w,-h,
    len,w,h, len,-w,h,  len,-w,h, len,-w,-h,  len,-w,-h, len,w,-h,  len,w,-h, len,w,h
  ]);
  const wVao = gl.createVertexArray();
  gl.bindVertexArray(wVao);
  const wb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, wb);
  gl.bufferData(gl.ARRAY_BUFFER, wire, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { fillVao: vao, fillCount: 12, wireVao: wVao, wireCount: 16 };
}

const BOX = makeBox();
const CYL = makeCyl(24);
const QUAD = makeQuad();
const FRUSTUM = makeFrustumMesh(2.44, 16 / 9, 0.55);

// dynamic line buffer for trails, grid, bounds
const dynVao = gl.createVertexArray();
const dynBuf = gl.createBuffer();
gl.bindVertexArray(dynVao);
gl.bindBuffer(gl.ARRAY_BUFFER, dynBuf);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);

function makeGridBuf(minX, maxX, minY, maxY) {
  const pts = [];
  for (let x = Math.ceil(minX); x <= Math.floor(maxX); x++) pts.push(x, minY, 0.002, x, maxY, 0.002);
  for (let y = Math.ceil(minY); y <= Math.floor(maxY); y++) pts.push(minX, y, 0.002, maxX, y, 0.002);
  const vao = gl.createVertexArray();
  const buf = gl.createBuffer();
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  return { vao, count: pts.length / 3 };
}
const GRID = makeGridBuf(-12, 12, -12, 12);

// ---- GLB binary loader -----------------------------------------------------------------------------
const COMP_SIZES = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const TYPE_COUNTS = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

function readAccessor(gltf, bin, idx) {
  const acc = gltf.accessors[idx];
  const bv = gltf.bufferViews[acc.bufferView];
  const nComp = TYPE_COUNTS[acc.type];
  const cSize = COMP_SIZES[acc.componentType];
  const stride = bv.byteStride || (nComp * cSize);
  const start = (bv.byteOffset || 0) + (acc.byteOffset || 0);
  const count = acc.count;
  const ArrayCtor = { 5121: Uint8Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }[acc.componentType];
  if (!ArrayCtor) throw new Error('Unsupported GLB componentType ' + acc.componentType);
  if (stride === nComp * cSize && (bin.byteOffset + start) % cSize === 0) {
    return new ArrayCtor(bin.buffer, bin.byteOffset + start, count * nComp);
  }
  const out = new ArrayCtor(count * nComp);
  const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  for (let i = 0; i < count; i++) {
    const base = start + i * stride;
    for (let k = 0; k < nComp; k++) {
      const off = base + k * cSize;
      out[i * nComp + k] = acc.componentType === 5126 ? dv.getFloat32(off, true)
        : acc.componentType === 5125 ? dv.getUint32(off, true)
        : acc.componentType === 5123 ? dv.getUint16(off, true)
        : dv.getUint8(off);
    }
  }
  return out;
}

async function parseGLB(buf) {
  const dv = new DataView(buf);
  if (dv.getUint32(0, true) !== 0x46546C67) throw new Error('Invalid GLB');
  let off = 12, json = null, bin = null;
  while (off < buf.byteLength) {
    const len = dv.getUint32(off, true), type = dv.getUint32(off + 4, true);
    const bytes = new Uint8Array(buf, off + 8, len);
    if (type === 0x4E4F534A) json = JSON.parse(new TextDecoder().decode(bytes));
    else if (type === 0x004E4942) bin = bytes;
    off += 8 + len;
  }
  const textures = await Promise.all((json.textures || []).map(async t => {
    const imgDef = json.images?.[t.source];
    if (!imgDef || imgDef.bufferView === undefined) return null;
    const bv = json.bufferViews[imgDef.bufferView];
    const slice = bin.subarray(bv.byteOffset || 0, (bv.byteOffset || 0) + bv.byteLength);
    const bmp = await createImageBitmap(new Blob([slice], { type: imgDef.mimeType || 'image/png' }));
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, bmp);
    bmp.close();
    const samp = json.samplers?.[t.sampler] || {};
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, samp.wrapS || gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, samp.wrapT || gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.generateMipmap(gl.TEXTURE_2D);
    return tex;
  }));

  const materials = (json.materials || []).map(m => {
    const pbr = m.pbrMetallicRoughness || {};
    const bct = pbr.baseColorTexture;
    const xform = bct?.extensions?.KHR_texture_transform;
    return {
      color: pbr.baseColorFactor || [1, 1, 1, 1],
      tex: bct ? textures[bct.index] : null,
      offset: xform?.offset || [0, 0],
      scale: xform?.scale || [1, 1],
      rot: xform?.rotation || 0,
      doubleSided: Boolean(m.doubleSided),
    };
  });

  const meshes = (json.meshes || []).map(mesh => mesh.primitives.map(pr => {
    const pos = readAccessor(json, bin, pr.attributes.POSITION);
    const norm = pr.attributes.NORMAL !== undefined ? readAccessor(json, bin, pr.attributes.NORMAL) : new Float32Array(pos.length);
    const uv = pr.attributes.TEXCOORD_0 !== undefined ? readAccessor(json, bin, pr.attributes.TEXCOORD_0) : new Float32Array((pos.length / 3) * 2);
    let idx = pr.indices !== undefined ? readAccessor(json, bin, pr.indices)
      : Uint32Array.from({ length: pos.length / 3 }, (_, i) => i);
    if (idx instanceof Uint8Array) idx = new Uint16Array(idx);
    const gpu = makeMesh(pos instanceof Float32Array ? pos : new Float32Array(pos),
                         norm instanceof Float32Array ? norm : new Float32Array(norm),
                         uv instanceof Float32Array ? uv : new Float32Array(uv), idx);
    const mat = pr.material !== undefined ? materials[pr.material]
      : { color: [0.8,0.8,0.8,1], tex: null, offset: [0,0], scale: [1,1], rot: 0, doubleSided: false };
    return { ...gpu, mat };
  }));

  const draws = [];
  const walk = (ni, parentMat) => {
    const n = json.nodes[ni];
    const local = n.matrix ? new Float32Array(n.matrix) : M.trs(n.translation, n.rotation, n.scale);
    const world = M.mul(parentMat, local);
    if (n.mesh !== undefined) {
      for (const prim of meshes[n.mesh]) draws.push({ prim, localMat: world });
    }
    for (const c of (n.children || [])) walk(c, world);
  };
  const sc = json.scenes?.[json.scene || 0];
  for (const ni of (sc?.nodes || [])) walk(ni, M.id());
  return draws;
}

const glbCache = new Map();
function loadGLB(url) {
  if (glbCache.has(url)) return glbCache.get(url);
  const p = fetch(url).then(r => { if (!r.ok) throw new Error(url); return r.arrayBuffer(); }).then(parseGLB);
  glbCache.set(url, p);
  return p;
}

// ---- scene 3D manifest & SLAM floor texture ---------------------------------------------------------
let world3d = null; // { mode:'3d', hab2house, stageDraws, items:[{draws, worldMat}], slamTex, slamRect }
let loadedSceneKey = '';
let loadSeq = 0;

function makeSlamTexture(slam) {
  if (!slam || !slam.png) return null;
  const img = new Image();
  const entry = { tex: null, rect: slam };
  img.onload = () => {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    entry.tex = tex;
  };
  img.src = 'data:image/png;base64,' + slam.png;
  return entry;
}

async function fetchScene3D(sceneName) {
  const seq = ++loadSeq;
  const statusEl = document.getElementById('status3d');
  if (statusEl) statusEl.textContent = 'Loading 3D room…';
  try {
    const r = await fetch('/scene3d', { cache: 'no-store' });
    if (!r.ok || seq !== loadSeq) return;
    const data = await r.json();
    if (seq !== loadSeq) return;
    if (data.mode !== '3d') {
      world3d = { mode: '2d', slam: makeSlamTexture(data.slam) };
      if (statusEl) statusEl.textContent = '';
      return;
    }
    const h2h = new Float32Array([
      0, -1, 0, 0,
      0,  0, 1, 0,
     -1,  0, 0, 0,
     -data.cx, -data.cy, -data.ground_y, 1
    ]);
    const st = { mode: '3d', h2h, stageDraws: [], items: [], slam: makeSlamTexture(data.slam) };
    world3d = st;
    if (data.stage) {
      const stageDraws = await loadGLB(data.stage);
      if (seq !== loadSeq) return;
      st.stageDraws = stageDraws.map(d => ({ prim: d.prim, worldMat: M.mul(h2h, d.localMat), normMat: null }));
      st.stageDraws.forEach(d => { d.normMat = M.normalMat3(d.worldMat); });
    }
    const byUrl = new Map();
    for (const inst of (data.instances || [])) {
      if (!byUrl.has(inst.url)) byUrl.set(inst.url, []);
      byUrl.get(inst.url).push(M.mul(h2h, M.transpose(inst.mat)));
    }
    const entries = [...byUrl.entries()];
    let done = 0;
    const batch = 6;
    for (let i = 0; i < entries.length; i += batch) {
      if (seq !== loadSeq) return;
      await Promise.all(entries.slice(i, i + batch).map(async ([url, mats]) => {
        try {
          const draws = await loadGLB(url);
          if (seq !== loadSeq) return;
          for (const instMat of mats) {
            for (const d of draws) {
              const wm = M.mul(instMat, d.localMat);
              st.items.push({ prim: d.prim, worldMat: wm, normMat: M.normalMat3(wm) });
            }
          }
        } catch {}
        done++;
      }));
      if (statusEl && seq === loadSeq) statusEl.textContent = `3D assets ${done}/${entries.length}`;
    }
    if (statusEl && seq === loadSeq) statusEl.textContent = '';
  } catch {
    if (statusEl && seq === loadSeq) statusEl.textContent = '';
  }
}

// ---- camera, controls & state ----------------------------------------------------------------------
const cam = {
  mode: 'orbit', // orbit | follow | chase | top
  target: [0, 0, 0.25],
  az: -2.35,
  el: 0.62,
  dist: 7.5,
  cutZ: 2.35,
  showFov: true,
  showTrail: true,
  showBeliefs: true,
  showSlam: true,
  showLabels: true,
};
let latestState = null;
let smoothRobot = [0, 0, 0];
let keys = new Set();
let hovered = false;

function toHouse(pt, dock) {
  if (!dock) return [pt[0], pt[1]];
  const [dx, dy, dyaw] = dock, c = Math.cos(dyaw), s = Math.sin(dyaw);
  return [dx + pt[0]*c - pt[1]*s, dy + pt[0]*s + pt[1]*c];
}

function robotWorldPose(s) {
  if (!s) return [0, 0, 0];
  if (Array.isArray(s.true_pose) && s.true_pose.length >= 3) return s.true_pose;
  const p = s.pose || [0, 0, 0];
  const [hx, hy] = toHouse(p, s.dock);
  return [hx, hy, p[2] + (s.dock ? s.dock[2] : 0)];
}

function focusRobot(snap = false) {
  const [rx, ry] = robotWorldPose(latestState);
  if (snap) {
    cam.target = [rx, ry, 0.22];
  } else {
    cam.target[0] = rx; cam.target[1] = ry; cam.target[2] = 0.22;
  }
}

function resetCamera() {
  cam.mode = 'orbit';
  cam.az = -2.35;
  cam.el = 0.62;
  cam.dist = 7.5;
  focusRobot(true);
  syncToolbar();
}

function syncToolbar() {
  for (const m of ['orbit', 'follow', 'chase', 'top']) {
    const b = document.getElementById('cam-' + m);
    if (b) b.classList.toggle('active', cam.mode === m);
  }
  const tog = (id, v) => { const b = document.getElementById(id); if (b) b.classList.toggle('active', v); };
  tog('tog-fov', cam.showFov);
  tog('tog-trail', cam.showTrail);
  tog('tog-beliefs', cam.showBeliefs);
  tog('tog-slam', cam.showSlam);
  tog('tog-labels', cam.showLabels);
}

// pointer & wheel interaction on the overlay canvas
let drag = null;
lblCanvas.addEventListener('contextmenu', e => e.preventDefault());
lblCanvas.addEventListener('pointerdown', e => {
  lblCanvas.setPointerCapture(e.pointerId);
  const pan = e.button === 2 || e.button === 1 || e.shiftKey;
  if (!pan && (cam.mode === 'top' || cam.mode === 'chase')) {
    if (cam.mode === 'top') cam.mode = 'orbit';
    syncToolbar();
  }
  drag = { id: e.pointerId, x: e.clientX, y: e.clientY, pan };
});
lblCanvas.addEventListener('pointermove', e => {
  if (!drag || drag.id !== e.pointerId) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) {
    if (cam.mode === 'follow' || cam.mode === 'chase') { cam.mode = 'orbit'; syncToolbar(); }
    const sp = cam.dist * 0.0014;
    const rx = -Math.sin(cam.az), ry = Math.cos(cam.az);
    const fx = -Math.cos(cam.az), fy = -Math.sin(cam.az);
    cam.target[0] -= (dx * rx - dy * fx) * sp;
    cam.target[1] -= (dx * ry - dy * fy) * sp;
  } else {
    cam.az -= dx * 0.007;
    cam.el = Math.max(0.06, Math.min(1.54, cam.el + dy * 0.007));
  }
});
lblCanvas.addEventListener('pointerup', e => { if (drag && drag.id === e.pointerId) drag = null; });
lblCanvas.addEventListener('pointercancel', () => { drag = null; });
lblCanvas.addEventListener('wheel', e => {
  e.preventDefault();
  cam.dist = Math.max(0.6, Math.min(40, cam.dist * Math.exp(e.deltaY * 0.0011)));
}, { passive: false });
lblCanvas.addEventListener('pointerenter', () => { hovered = true; });
lblCanvas.addEventListener('pointerleave', () => { hovered = false; });

window.addEventListener('keydown', e => {
  if (!hovered || document.activeElement?.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if (['w','a','s','d','q','e'].includes(k)) { keys.add(k); e.preventDefault(); }
  if (k === 'f') { focusRobot(); e.preventDefault(); }
  if (k === 'r') { resetCamera(); e.preventDefault(); }
});
window.addEventListener('keyup', e => keys.delete(e.key.toLowerCase()));

// toolbar wiring
for (const m of ['orbit', 'follow', 'chase', 'top']) {
  const b = document.getElementById('cam-' + m);
  if (b) b.onclick = () => {
    cam.mode = m;
    if (m === 'top') { cam.el = 1.52; cam.dist = Math.max(cam.dist, 9); }
    else if (m === 'chase') { cam.el = 0.36; cam.dist = 2.4; }
    else if (m === 'follow') { cam.el = Math.min(cam.el, 1.0); }
    focusRobot();
    syncToolbar();
  };
}
const btnReset = document.getElementById('cam-reset');
if (btnReset) btnReset.onclick = resetCamera;
for (const [id, key] of [['tog-fov','showFov'],['tog-trail','showTrail'],['tog-beliefs','showBeliefs'],['tog-slam','showSlam'],['tog-labels','showLabels']]) {
  const b = document.getElementById(id);
  if (b) b.onclick = () => { cam[key] = !cam[key]; syncToolbar(); };
}
const cutSlider = document.getElementById('cut-z');
const cutVal = document.getElementById('cut-z-val');
if (cutSlider) cutSlider.oninput = () => {
  const v = parseFloat(cutSlider.value);
  cam.cutZ = v >= 3.45 ? 100.0 : v;
  if (cutVal) cutVal.textContent = v >= 3.45 ? 'Full' : v.toFixed(1) + 'm';
};

const btnExpand = document.getElementById('btn-expand-map');
const mapSec = document.getElementById('mapsec');
if (btnExpand && mapSec) btnExpand.onclick = () => {
  const wide = mapSec.classList.toggle('expanded3d');
  btnExpand.textContent = wide ? 'Shrink' : 'Expand';
};
syncToolbar();

// ---- drawing helpers -------------------------------------------------------------------------------
function drawMeshItem(mesh, modelMat, color, tex = null, doubleSided = false, unlit = false,
                      texOff = [0,0], texSc = [1,1], texRot = 0, cutZ = 100.0, normMat = null) {
  if (doubleSided) gl.disable(gl.CULL_FACE); else gl.enable(gl.CULL_FACE);
  gl.uniformMatrix4fv(U.uModel, false, modelMat);
  gl.uniformMatrix3fv(U.uNormMat, false, normMat || M.normalMat3(modelMat));
  gl.uniform4fv(U.uColor, color);
  gl.uniform1f(U.uCutZ, cutZ);
  gl.uniform1i(U.uUnlit, unlit ? 1 : 0);
  if (tex) {
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(U.uTex, 0);
    gl.uniform1i(U.uHasTex, 1);
    gl.uniform2fv(U.uTexOffset, texOff);
    gl.uniform2fv(U.uTexScale, texSc);
    gl.uniform1f(U.uTexRot, texRot);
  } else {
    gl.uniform1i(U.uHasTex, 0);
  }
  gl.bindVertexArray(mesh.vao);
  gl.drawElements(gl.TRIANGLES, mesh.count, mesh.idxType, 0);
}

function drawLines(vp, pts, color, mode = gl.LINES) {
  if (!pts.length) return;
  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(UL.uMVP, false, vp);
  gl.uniform4fv(UL.uColor, color);
  gl.bindVertexArray(dynVao);
  gl.bindBuffer(gl.ARRAY_BUFFER, dynBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pts), gl.DYNAMIC_DRAW);
  gl.drawArrays(mode, 0, pts.length / 3);
}

function hexToRgba(hex, a = 1) {
  if (!hex || hex[0] !== '#') return [0.55, 0.58, 0.64, a];
  const h = hex.length === 4 ? hex[1]+hex[1]+hex[2]+hex[2]+hex[3]+hex[3] : hex.slice(1);
  const n = parseInt(h, 16);
  return [((n >> 16) & 255)/255, ((n >> 8) & 255)/255, (n & 255)/255, a];
}

function drawRobot(x, y, yaw, vp, ghost = false) {
  const col = (rgb, a) => [rgb[0], rgb[1], rgb[2], ghost ? a * 0.45 : a];
  // base cylinder (r=0.15, h=0.10, z=0.04..0.14)
  drawMeshItem(CYL, M.fromPosYawScale(x, y, 0.09, yaw, 0.15, 0.15, 0.10),
    col(ghost ? [0.22, 0.55, 0.98] : [0.95, 0.55, 0.12], 1));
  // wheels (r=0.05, w=0.03, y=+-0.12, z=0.05)
  for (const side of [-1, 1]) {
    const wx = x - side * 0.12 * Math.sin(yaw), wy = y + side * 0.12 * Math.cos(yaw);
    const wm = M.mul(M.fromPosYawScale(wx, wy, 0.05, yaw, 1, 1, 1),
                     M.mul(M.rotX(Math.PI / 2), M.fromPosYawScale(0, 0, 0, 0, 0.05, 0.05, 0.03)));
    drawMeshItem(CYL, wm, col([0.12, 0.12, 0.14], 1));
  }
  // top deck (r=0.07, h=0.07, z=0.20)
  drawMeshItem(CYL, M.fromPosYawScale(x, y, 0.20, yaw, 0.07, 0.07, 0.07),
    col(ghost ? [0.35, 0.65, 1.0] : [0.55, 0.57, 0.62], 1));
  // 3 camera pods
  for (const cyaw of [0, 2*Math.PI/3, -2*Math.PI/3]) {
    const cx = x + 0.11 * Math.cos(yaw + cyaw), cy = y + 0.11 * Math.sin(yaw + cyaw);
    drawMeshItem(BOX, M.fromPosYawScale(cx, cy, 0.22, yaw + cyaw, 0.025, 0.09, 0.03), col([0.1, 0.12, 0.16], 1));
  }
  // heading bar on deck
  const hx = x + 0.08 * Math.cos(yaw), hy = y + 0.08 * Math.sin(yaw);
  drawMeshItem(BOX, M.fromPosYawScale(hx, hy, 0.245, yaw, 0.14, 0.024, 0.015),
    col(ghost ? [0.4, 0.75, 1.0] : [0.98, 0.98, 1.0], 1), null, false, true);

  if (cam.showFov && !ghost) {
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    gl.disable(gl.CULL_FACE);
    for (const [cyaw, fc, wc] of [
      [0, [0.18, 0.55, 1.0, 0.08], [0.28, 0.68, 1.0, 0.42]],
      [2*Math.PI/3, [0.15, 0.78, 0.72, 0.05], [0.22, 0.85, 0.78, 0.25]],
      [-2*Math.PI/3, [0.15, 0.78, 0.72, 0.05], [0.22, 0.85, 0.78, 0.25]],
    ]) {
      const fx = x + 0.11 * Math.cos(yaw + cyaw), fy = y + 0.11 * Math.sin(yaw + cyaw);
      const fm = M.fromPosYawScale(fx, fy, 0.22, yaw + cyaw, 1, 1, 1);
      gl.useProgram(meshProg);
      gl.uniformMatrix4fv(U.uModel, false, fm);
      gl.uniformMatrix3fv(U.uNormMat, false, M.normalMat3(fm));
      gl.uniform4fv(U.uColor, fc);
      gl.uniform1i(U.uHasTex, 0);
      gl.uniform1i(U.uUnlit, 1);
      gl.uniform1f(U.uCutZ, 100.0);
      gl.bindVertexArray(FRUSTUM.fillVao);
      gl.drawArrays(gl.TRIANGLES, 0, FRUSTUM.fillCount);

      gl.useProgram(lineProg);
      gl.uniformMatrix4fv(UL.uMVP, false, M.mul(vp, fm));
      gl.uniform4fv(UL.uColor, wc);
      gl.bindVertexArray(FRUSTUM.wireVao);
      gl.drawArrays(gl.LINES, 0, FRUSTUM.wireCount);
    }
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }
}

function project(p, vp, w, h) {
  const x = vp[0]*p[0] + vp[4]*p[1] + vp[8]*p[2]  + vp[12];
  const y = vp[1]*p[0] + vp[5]*p[1] + vp[9]*p[2]  + vp[13];
  const z = vp[2]*p[0] + vp[6]*p[1] + vp[10]*p[2] + vp[14];
  const cw = vp[3]*p[0] + vp[7]*p[1] + vp[11]*p[2] + vp[15];
  if (cw <= 0.05) return null;
  const ndcX = x / cw, ndcY = y / cw, ndcZ = z / cw;
  if (ndcZ < -1 || ndcZ > 1 || ndcX < -1.15 || ndcX > 1.15 || ndcY < -1.15 || ndcY > 1.15) return null;
  return [(ndcX * 0.5 + 0.5) * w, (0.5 - ndcY * 0.5) * h];
}

// ---- render loop -----------------------------------------------------------------------------------
let lastTime = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  if (root.hidden || root.offsetParent === null) { lastTime = now; return; }
  const dt = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;

  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cw = Math.max(1, Math.round(glCanvas.clientWidth * dpr));
  const ch = Math.max(1, Math.round(glCanvas.clientHeight * dpr));
  if (glCanvas.width !== cw || glCanvas.height !== ch) {
    glCanvas.width = cw; glCanvas.height = ch;
    lblCanvas.width = cw; lblCanvas.height = ch;
  }

  const s = latestState;
  const rPose = robotWorldPose(s);
  smoothRobot[0] += (rPose[0] - smoothRobot[0]) * Math.min(1, dt * 12);
  smoothRobot[1] += (rPose[1] - smoothRobot[1]) * Math.min(1, dt * 12);
  let dyaw = rPose[2] - smoothRobot[2];
  while (dyaw > Math.PI) dyaw -= 2*Math.PI;
  while (dyaw < -Math.PI) dyaw += 2*Math.PI;
  smoothRobot[2] += dyaw * Math.min(1, dt * 12);

  if (keys.size && hovered) {
    if (cam.mode === 'follow' || cam.mode === 'chase') { cam.mode = 'orbit'; syncToolbar(); }
    const sp = 3.5 * dt;
    const fx = -Math.cos(cam.az), fy = -Math.sin(cam.az);
    const rx = -Math.sin(cam.az), ry = Math.cos(cam.az);
    if (keys.has('w')) { cam.target[0] += fx*sp; cam.target[1] += fy*sp; }
    if (keys.has('s')) { cam.target[0] -= fx*sp; cam.target[1] -= fy*sp; }
    if (keys.has('d')) { cam.target[0] += rx*sp; cam.target[1] += ry*sp; }
    if (keys.has('a')) { cam.target[0] -= rx*sp; cam.target[1] -= ry*sp; }
    if (keys.has('e')) cam.target[2] = Math.min(4, cam.target[2] + sp);
    if (keys.has('q')) cam.target[2] = Math.max(0, cam.target[2] - sp);
  }
  if (cam.mode === 'follow') {
    cam.target[0] = smoothRobot[0]; cam.target[1] = smoothRobot[1]; cam.target[2] = 0.22;
  } else if (cam.mode === 'chase') {
    cam.target[0] = smoothRobot[0] + 0.5*Math.cos(smoothRobot[2]);
    cam.target[1] = smoothRobot[1] + 0.5*Math.sin(smoothRobot[2]);
    cam.target[2] = 0.28;
    const wantAz = smoothRobot[2] + Math.PI;
    let da = wantAz - cam.az;
    while (da > Math.PI) da -= 2*Math.PI;
    while (da < -Math.PI) da += 2*Math.PI;
    cam.az += da * Math.min(1, dt * 6);
  }

  const eye = [
    cam.target[0] + cam.dist * Math.cos(cam.el) * Math.cos(cam.az),
    cam.target[1] + cam.dist * Math.cos(cam.el) * Math.sin(cam.az),
    cam.target[2] + cam.dist * Math.sin(cam.el),
  ];
  const proj = M.perspective(0.88, cw / ch, 0.05, 80.0);
  const view = M.lookAt(eye, cam.target, [0, 0, 1]);
  const vp = M.mul(proj, view);

  const dark = window.matchMedia?.('(prefers-color-scheme: dark)').matches;
  gl.viewport(0, 0, cw, ch);
  gl.clearColor(dark ? 0.09 : 0.91, dark ? 0.09 : 0.90, dark ? 0.10 : 0.88, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.enable(gl.DEPTH_TEST);

  // subtle 1m floor grid
  gl.useProgram(lineProg);
  gl.uniformMatrix4fv(UL.uMVP, false, vp);
  gl.uniform4fv(UL.uColor, dark ? [0.22,0.22,0.24,1] : [0.76,0.75,0.72,1]);
  gl.bindVertexArray(GRID.vao);
  gl.drawArrays(gl.LINES, 0, GRID.count);

  gl.useProgram(meshProg);
  gl.uniformMatrix4fv(U.uProj, false, proj);
  gl.uniformMatrix4fv(U.uView, false, view);

  // draw 3D apartment meshes OR extruded 2D room
  if (world3d && world3d.mode === '3d') {
    for (const d of world3d.stageDraws) {
      const m = d.prim.mat;
      drawMeshItem(d.prim, d.worldMat, m.color, m.tex, m.doubleSided, false,
                   m.offset, m.scale, m.rot, cam.cutZ, d.normMat);
    }
    for (const it of world3d.items) {
      const m = it.prim.mat;
      drawMeshItem(it.prim, it.worldMat, m.color, m.tex, m.doubleSided, false,
                   m.offset, m.scale, m.rot, cam.cutZ, it.normMat);
    }
  } else if (s) {
    const [W, H] = s.size || [8, 6];
    drawMeshItem(QUAD, M.fromPosYawScale(0, 0, -0.002, 0, W, H, 1),
      dark ? [0.16, 0.16, 0.17, 1] : [0.97, 0.96, 0.94, 1], null, true, false);
    const wallH = Math.min(2.3, cam.cutZ > 90 ? 2.3 : cam.cutZ);
    for (const [a, b] of (s.walls || [])) {
      const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
      if (len < 1e-4) continue;
      const wm = M.fromPosYawScale((a[0]+b[0])/2, (a[1]+b[1])/2, wallH/2, Math.atan2(dy, dx), len, 0.08, wallH);
      drawMeshItem(BOX, wm, dark ? [0.36, 0.36, 0.39, 1] : [0.78, 0.77, 0.74, 1]);
    }
    for (const t of (s.objects || [])) {
      const rr = Math.max(0.08, Number(t.r) || 0.25);
      if (t.user) {
        drawMeshItem(CYL, M.fromPosYawScale(t.x, t.y, 0.65, 0, rr*0.7, rr*0.7, 1.15), [0.18, 0.45, 0.92, 1]);
        drawMeshItem(CYL, M.fromPosYawScale(t.x, t.y, 1.42, 0, rr*0.5, rr*0.5, 0.28), [0.24, 0.52, 0.96, 1]);
      } else {
        const col = hexToRgba(t.color, 1);
        const h = t.known ? 0.55 : 0.42;
        if (!t.known) { col[0] = col[0]*0.6+0.2; col[1] = col[1]*0.6+0.2; col[2] = col[2]*0.6+0.2; }
        drawMeshItem(CYL, M.fromPosYawScale(t.x, t.y, h/2, 0, rr, rr, h), col);
      }
    }
  }

  // optional SLAM occupancy floor overlay
  const slam = world3d?.slam;
  if (cam.showSlam && slam && slam.tex && s) {
    const sr = slam.rect;
    const dock = s.dock || [0, 0, 0];
    const [cx, cy] = toHouse([sr.cx, sr.cy], dock);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    drawMeshItem(QUAD, M.fromPosYawScale(cx, cy, 0.008, dock[2], sr.w, sr.h, 1),
      [1, 1, 1, 1], slam.tex, true, true);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
  }

  // trails
  if (cam.showTrail && s) {
    const bPts = [];
    for (const pt of (s.trail || [])) {
      const [hx, hy] = toHouse(pt, s.dock);
      bPts.push(hx, hy, 0.025);
    }
    if (bPts.length >= 6) drawLines(vp, bPts, [0.20, 0.52, 0.98, 1], gl.LINE_STRIP);

    if (Array.isArray(s.true_trail) && s.true_trail.length >= 2) {
      const tPts = [];
      for (const pt of s.true_trail) tPts.push(pt[0], pt[1], 0.032);
      drawLines(vp, tPts, [0.96, 0.52, 0.10, 1], gl.LINE_STRIP);
    }
  }

  // believed vs true robot bodies
  if (s) {
    gl.useProgram(meshProg);
    gl.uniformMatrix4fv(U.uProj, false, proj);
    gl.uniformMatrix4fv(U.uView, false, view);
    if (Array.isArray(s.true_pose) && s.true_pose.length >= 3) {
      drawRobot(smoothRobot[0], smoothRobot[1], smoothRobot[2], vp, false);
      const bp = s.pose || [0, 0, 0];
      const [bx, by] = toHouse(bp, s.dock);
      if (Math.hypot(bx - smoothRobot[0], by - smoothRobot[1]) > 0.04) {
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(meshProg);
        drawRobot(bx, by, bp[2] + (s.dock ? s.dock[2] : 0), vp, true);
        gl.disable(gl.BLEND);
      }
    } else {
      drawRobot(smoothRobot[0], smoothRobot[1], smoothRobot[2], vp, false);
    }

    // believed object markers in 3D apartment mode
    if (cam.showBeliefs && world3d?.mode === '3d') {
      gl.useProgram(meshProg);
      for (const t of (s.objects || [])) {
        const [ox, oy] = toHouse([t.x, t.y], s.dock);
        const rr = Math.max(0.10, Math.min(0.45, Number(t.r) || 0.2));
        drawMeshItem(CYL, M.fromPosYawScale(ox, oy, 0.03, 0, rr, rr, 0.04),
          t.user ? [0.20, 0.55, 1.0, 1] : [0.22, 0.65, 0.98, 1], null, false, true);
      }
    }
  }

  // 2D crisp label overlay
  const g = lblCanvas.getContext('2d');
  g.clearRect(0, 0, cw, ch);
  if (!cam.showLabels || !s) return;
  g.save();
  g.font = `${Math.round(11.5 * dpr)}px system-ui, sans-serif`;
  g.textBaseline = 'middle';
  const placed = [];
  const drawTag = (text, wx, wy, wz, bg, fg) => {
    const sp = project([wx, wy, wz], vp, cw, ch);
    if (!sp) return;
    const tw = g.measureText(text).width + 10 * dpr, th = 18 * dpr;
    const x0 = sp[0] - tw / 2, y0 = sp[1] - th / 2;
    const box = [x0, y0, x0 + tw, y0 + th];
    if (placed.some(b => !(box[2]<b[0] || box[0]>b[2] || box[3]<b[1] || box[1]>b[3]))) return;
    placed.push(box);
    g.fillStyle = bg;
    g.beginPath();
    g.roundRect(x0, y0, tw, th, 4 * dpr);
    g.fill();
    g.fillStyle = fg;
    g.fillText(text, x0 + 5 * dpr, sp[1]);
  };

  drawTag('Sprocket', smoothRobot[0], smoothRobot[1], 0.42, 'rgba(242,140,31,0.92)', '#fff');
  if (cam.showBeliefs) {
    for (const t of (s.objects || [])) {
      const [ox, oy] = toHouse([t.x, t.y], s.dock);
      const lbl = t.user ? 'you' : `${t.label} ${t.id ?? ''}`.trim();
      drawTag(lbl, ox, oy, t.user ? 1.7 : 0.72,
        t.user ? 'rgba(47,111,222,0.92)' : (t.known ? 'rgba(30,45,70,0.85)' : 'rgba(90,90,90,0.7)'), '#fff');
    }
  }
  if (world3d?.mode === '3d' && Array.isArray(s.true_objects)) {
    const seen = new Set();
    for (const o of s.true_objects) {
      if ((o.height || 0) < 0.35 && o.radius < 0.25) continue;
      if (seen.has(o.label + ':' + Math.round(o.x) + ',' + Math.round(o.y))) continue;
      seen.add(o.label + ':' + Math.round(o.x) + ',' + Math.round(o.y));
      drawTag(o.label, o.x, o.y, Math.min(1.6, (o.height || 0.6) + 0.15), 'rgba(24,24,26,0.68)', '#e4e2dc');
    }
  }
  if (s.room_bounds && world3d?.mode !== '3d') {
    for (const [rname, rect] of Object.entries(s.room_bounds)) {
      if (!Array.isArray(rect) || rect.length < 4) continue;
      drawTag(rname, (rect[0]+rect[2])/2, (rect[1]+rect[3])/2, 0.08, 'rgba(0,0,0,0.42)', '#e8e6e0');
    }
  }
  g.restore();
}

window.sprocket3d = {
  update(s) {
    const first = latestState === null;
    latestState = s;
    const key = `${s.scene || ''}|${Array.isArray(s.true_pose) ? '3d' : '2d'}`;
    if (key !== loadedSceneKey) {
      loadedSceneKey = key;
      fetchScene3D(s.scene || '');
      focusRobot(true);
    } else if (first) {
      focusRobot(true);
    }
  },
};

requestAnimationFrame(frame);
})();
"""
