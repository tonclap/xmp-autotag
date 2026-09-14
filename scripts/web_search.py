"""Local web UI for searching the archive: thumbnails, badges, duplicates tab.

    python scripts/web_search.py          # http://localhost:8766
    python scripts/web_search.py 9000     # a different port

Deliberately one file and no web framework: stdlib http.server, one HTML/CSS/JS
page, /api/* returning JSON. Binds to 127.0.0.1 only — it serves images straight
off your disk and has no authentication of any kind.

Search logic lives in search_core.py (shared with the CLI). This module adds
lazy, cached thumbnails and nothing else.
"""
import json
import sys
from hashlib import sha1
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

import config
import search_core as sc
import vlm

for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

THUMB_DIR = config.OUTPUT_DIR / "thumb_cache"
# Roots an image may be served from. Same roots the tagger walks: a path that
# came out of the index still gets checked against them.
ALLOWED_ROOTS = [root for root in (config.PHOTO_ROOT, config.MEDIA_ROOT) if root is not None]


def _thumb_cache_path(path, max_dim):
    st = path.stat()
    key = f"{path}|{st.st_mtime_ns}|{max_dim}"
    return THUMB_DIR / f"{sha1(key.encode('utf-8')).hexdigest()}.jpg"


def make_thumbnail(path, max_dim):
    """JPEG bytes for one image, cached on disk by (path, mtime, size)."""
    cache = _thumb_cache_path(path, max_dim)
    if cache.exists():
        return cache.read_bytes()
    data = path.read_bytes()
    try:
        img = Image.open(BytesIO(data))
        img.load()
    except Exception:
        # RAW file Pillow cannot decode: fall back to its embedded preview.
        jpeg = vlm.extract_embedded_jpeg(data)
        if jpeg is None:
            return None
        img = Image.open(BytesIO(jpeg))
    img = ImageOps.exif_transpose(img)
    img.thumbnail((max_dim, max_dim))
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    out = buf.getvalue()
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(out)
    return out


def safe_photo_path(raw):
    """Resolve a requested path, but only inside ALLOWED_ROOTS.

    The index is trusted data, yet the query string is not: without this check
    /thumb?p=... would read any file on the machine.
    """
    if not raw:
        return None
    try:
        path = Path(raw).resolve()
    except OSError:
        return None
    for root in ALLOWED_ROOTS:
        try:
            path.relative_to(root.resolve())
            return path if path.is_file() else None
        except (ValueError, OSError):
            continue
    return None


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Photo archive — semantic search</title>
<style>
:root{
  --bg:#f4f6fb;--card:#fff;--ink:#0b1020;--ink-soft:#414a63;--muted:#8189a0;
  --line:#e4e8f2;--accent:#2340e0;--accent-soft:#dde3fd;--radius:14px;
}
@media (prefers-color-scheme: dark){
  :root{--bg:#12141c;--card:#1b1e2a;--ink:#eef0f8;--ink-soft:#c3c8db;
    --muted:#7d859e;--line:#2c3040;--accent:#7c93ff;--accent-soft:#25305c;}
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--card);
       border-bottom:1px solid var(--line);padding:14px 20px}
.wrap{max-width:1200px;margin:0 auto}
.head-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.head-titles{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;min-width:0}
h1{font-size:18px;margin:0;white-space:nowrap}
#stats{color:var(--muted);font-size:12.5px;white-space:nowrap}
#q{flex:1;min-width:220px;padding:11px 14px;font-size:15px;border:1.5px solid var(--line);
   border-radius:10px;background:var(--bg);color:var(--ink);outline:none}
#q:focus{border-color:var(--accent)}
main{max-width:1200px;margin:0 auto;padding:18px 20px 60px}
#count{color:var(--muted);font-size:13px;margin:4px 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
      overflow:hidden;cursor:pointer;transition:border-color .15s,transform .1s}
.card:hover{border-color:var(--accent);transform:translateY(-1px)}
.thumb-wrap{position:relative;aspect-ratio:4/3;background:var(--line)}
.thumb-wrap img{width:100%;height:100%;object-fit:cover;display:block}
.score{position:absolute;top:8px;right:8px;background:rgba(11,16,32,.72);color:#fff;
       font:600 11px ui-monospace,monospace;padding:2px 7px;border-radius:999px}
.match-badge{position:absolute;top:8px;left:8px;font:600 11px system-ui,sans-serif;
       padding:2px 8px;border-radius:999px;color:#fff;max-width:calc(100% - 60px);
       overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.match-badge.person{background:rgba(35,64,224,.85)}
.match-badge.location{background:rgba(18,133,90,.85)}
.match-badge.date{background:rgba(180,83,9,.85)}
.card-body{padding:10px 12px 12px}
.desc{font-size:13px;color:var(--ink-soft);display:-webkit-box;
      -webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;min-height:3.9em}
.kw{display:flex;flex-wrap:wrap;gap:4px;margin-top:8px}
.chip{font-size:11px;color:var(--accent);background:var(--accent-soft);
      padding:2px 8px;border-radius:999px}
.chip-more{color:var(--muted);background:transparent}
.empty{color:var(--muted);text-align:center;padding:60px 0}
.overlay{position:fixed;inset:0;background:rgba(11,16,32,.6);display:none;
         align-items:center;justify-content:center;padding:24px;z-index:20}
.overlay.on{display:flex}
.doc{max-width:1000px;max-height:calc(100vh - 48px);width:100%;background:var(--card);
     border-radius:18px;overflow:hidden;display:flex;flex-direction:column}
.doc-h{display:flex;align-items:center;gap:10px;padding:12px 16px;border-bottom:1px solid var(--line)}
.doc-h .x{margin-left:auto;cursor:pointer;border:0;background:none;color:var(--muted);
          font-size:20px;line-height:1;padding:4px 10px;border-radius:8px}
.doc-h .x:hover{color:var(--accent)}
.doc-body{overflow:auto;padding:16px}
.doc-body img{width:100%;border-radius:10px;display:block;background:var(--line)}
.doc-desc{margin-top:14px;font-size:14.5px;color:var(--ink-soft)}
.doc-path{margin-top:10px;font:12px ui-monospace,monospace;color:var(--muted);
          word-break:break-all;user-select:all}
.tabs{display:flex;gap:6px;flex-shrink:0}
.media-toggle{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--ink-soft);
     white-space:nowrap;cursor:pointer;user-select:none;flex-shrink:0;
     padding:9px 14px;border-radius:10px;border:1.5px solid var(--line);background:var(--bg)}
.media-toggle:hover{border-color:var(--accent)}
.media-toggle input{cursor:pointer;accent-color:var(--accent)}
.source-badge{position:absolute;bottom:8px;right:8px;background:rgba(124,147,255,.85);color:#fff;
     font:600 10.5px system-ui,sans-serif;padding:2px 7px;border-radius:999px}
.tab{padding:8px 16px;border-radius:999px;border:1.5px solid var(--line);background:var(--bg);
     color:var(--ink-soft);font:600 13px system-ui,sans-serif;cursor:pointer}
.tab.on{background:var(--accent);border-color:var(--accent);color:#fff}
.view{display:none}
.view.on{display:block}
.dup-group{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);
     padding:14px;margin:0 0 14px}
.dup-head{display:flex;align-items:baseline;gap:10px;margin-bottom:10px;font-size:13px;color:var(--muted)}
.dup-head b{color:var(--ink);font-size:14px}
.dup-row{display:flex;flex-wrap:wrap;gap:10px}
.dup-thumb{width:130px;cursor:pointer}
.dup-thumb .thumb-wrap{aspect-ratio:4/3;border-radius:10px;overflow:hidden}
.dup-thumb .fname{font-size:11px;color:var(--muted);margin-top:4px;overflow:hidden;
     text-overflow:ellipsis;white-space:nowrap}
</style></head>
<body>
<header><div class="wrap head-row">
  <div class="head-titles">
    <h1>Photo archive</h1>
    <span id="stats">loading index…</span>
  </div>
  <input id="q" placeholder="plain language, e.g. &laquo;kids playing in the snow&raquo;"
         autofocus autocomplete="off">
  <label class="media-toggle" id="mediaToggleWrap" style="display:none">
    <input type="checkbox" id="mediaToggle"> Media library</label>
  <div class="tabs">
    <button class="tab on" id="tabSearch">Search</button>
    <button class="tab" id="tabDup">Duplicates</button>
  </div>
</div></header>
<main>
  <div class="view on" id="searchView">
    <div id="count"></div>
    <div class="grid" id="results"></div>
  </div>
  <div class="view" id="dupView">
    <div id="dupInfo"></div>
    <div id="dupGroups"></div>
  </div>
</main>
<div class="overlay" id="overlay"><div class="doc">
  <div class="doc-h"><b id="docTitle">Photo</b>
    <button class="x" id="docClose" title="Close (Esc)">&#10005;</button></div>
  <div class="doc-body" id="docBody"></div>
</div></div>
<script>
const $=s=>document.querySelector(s), q=$("#q"), results=$("#results"),
      count=$("#count"), overlay=$("#overlay"), mediaToggle=$("#mediaToggle");
const esc=t=>String(t).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function thumbUrl(p,s){return "/thumb?p="+encodeURIComponent(p)+"&s="+s;}
function basename(p){return p.split(/[\\\\/]/).pop();}

fetch("/api/stats").then(r=>r.json()).then(s=>{
  const bySource=s.bySource||{};
  if(!s.indexed){$("#stats").textContent="no index yet — python scripts/build_search_index.py";return;}
  let text=`${bySource.photos||s.total} photos`;
  if(bySource.media){
    $("#mediaToggleWrap").style.display="";
    text+=` \\u00b7 ${bySource.media} media files (off)`;
  }
  $("#stats").textContent=text;
});
if(mediaToggle) mediaToggle.addEventListener("change",()=>{clearTimeout(timer);search();});

function cards(list){
  return list.map(x=>{
    const kws=x.keywords||[];
    const shown=kws.slice(0,5).map(k=>`<span class="chip">${esc(k)}</span>`).join("");
    const more=kws.length>5?`<span class="chip chip-more">+${kws.length-5}</span>`:"";
    let badge="";
    if(x.matchType==="person") badge=`<span class="match-badge person">${esc(x.matchLabel)}</span>`;
    else if(x.matchType==="location") badge=`<span class="match-badge location">${esc(x.matchLabel)}</span>`;
    else if(x.matchType==="date") badge=`<span class="match-badge date">${esc(x.matchLabel)}</span>`;
    const srcBadge=x.source&&x.source!=="photos"?`<span class="source-badge">${esc(x.source)}</span>`:"";
    return `<div class="card" data-path="${esc(x.path)}" data-score="${x.score.toFixed(3)}">
      <div class="thumb-wrap">
        <img loading="lazy" src="${thumbUrl(x.path,320)}"
             onerror="this.closest('.thumb-wrap').style.background='var(--line)'">
        <span class="score">${x.score.toFixed(2)}</span>
        ${badge}${srcBadge}
      </div>
      <div class="card-body">
        <div class="desc">${esc(x.description)}</div>
        <div class="kw">${shown}${more}</div>
      </div></div>`;
  }).join("");
}

let timer=null;
async function search(){
  const query=q.value.trim();
  if(!query){count.textContent="";results.innerHTML="";return;}
  try{
    const media=mediaToggle&&mediaToggle.checked?"&media=1":"";
    const r=await fetch("/api/search?q="+encodeURIComponent(query)+media);
    const d=await r.json();
    if(d.error){count.textContent="";results.innerHTML=`<div class="empty">${esc(d.error)}</div>`;return;}
    if(!d.results.length){count.textContent="";
      results.innerHTML=`<div class="empty">Nothing found for &laquo;${esc(query)}&raquo;</div>`;return;}
    count.textContent=`Top ${d.results.length} of ${d.total} by relevance`;
    results.innerHTML=cards(d.results);
  }catch(e){results.innerHTML=`<div class="empty">Error: ${e}</div>`;}
}
q.addEventListener("input",()=>{clearTimeout(timer);timer=setTimeout(search,300);});
q.addEventListener("keydown",e=>{if(e.key==="Enter"){clearTimeout(timer);search();}});

function openPhoto(path,desc,kws){
  $("#docTitle").textContent=basename(path);
  $("#docBody").innerHTML=`<img src="${thumbUrl(path,1600)}">
    ${desc?`<div class="doc-desc">${esc(desc)}</div>`:""}
    ${kws?`<div class="doc-desc">${esc(kws)}</div>`:""}
    <div class="doc-path">${esc(path)}</div>`;
  overlay.classList.add("on"); document.body.style.overflow="hidden";
}

results.addEventListener("click",e=>{
  const card=e.target.closest(".card"); if(!card)return;
  const desc=card.querySelector(".desc").textContent;
  const kws=[...card.querySelectorAll(".chip")].map(x=>x.textContent).join(", ");
  openPhoto(card.dataset.path,desc,kws);
});
function closeDoc(){overlay.classList.remove("on");document.body.style.overflow="";$("#docBody").innerHTML="";}
$("#docClose").onclick=closeDoc;
overlay.addEventListener("click",e=>{if(e.target===overlay)closeDoc();});
document.addEventListener("keydown",e=>{if(e.key==="Escape")closeDoc();});

// Duplicates tab: near_duplicates.json produced by find_near_duplicates.py.
const tabSearch=$("#tabSearch"), tabDup=$("#tabDup"), searchView=$("#searchView"), dupView=$("#dupView");
let dupLoaded=false;
tabSearch.onclick=()=>{tabSearch.classList.add("on");tabDup.classList.remove("on");
  searchView.classList.add("on");dupView.classList.remove("on");};
tabDup.onclick=()=>{tabDup.classList.add("on");tabSearch.classList.remove("on");
  dupView.classList.add("on");searchView.classList.remove("on");
  if(!dupLoaded){dupLoaded=true;loadDuplicates();}};

async function loadDuplicates(){
  const info=$("#dupInfo"), groups=$("#dupGroups");
  info.textContent="Loading…";
  try{
    const r=await fetch("/api/duplicates");
    const d=await r.json();
    if(d.error){info.innerHTML=`<div class="empty">${esc(d.error)}</div>`;return;}
    const dupCount=d.clusters.reduce((n,c)=>n+c.size,0);
    info.textContent=`Similarity threshold ${d.threshold} \\u00b7 ${d.clusters.length} groups \\u00b7 `
      +`${dupCount} images resemble another one in the same folder (bursts, not exact copies)`;
    groups.innerHTML=d.clusters.map(c=>{
      const thumbs=c.photos.map(p=>`<div class="dup-thumb" data-path="${esc(p.path)}">
        <div class="thumb-wrap"><img loading="lazy" src="${thumbUrl(p.path,220)}"></div>
        <div class="fname">${esc(basename(p.path))}</div></div>`).join("");
      return `<div class="dup-group">
        <div class="dup-head"><b>${c.size} images</b> similarity ~${c.avg_similarity}</div>
        <div class="dup-row">${thumbs}</div></div>`;
    }).join("") || `<div class="empty">No duplicates found</div>`;
  }catch(e){info.innerHTML=`<div class="empty">Error: ${e}</div>`;}
}
$("#dupGroups").addEventListener("click",e=>{
  const t=e.target.closest(".dup-thumb"); if(!t)return;
  openPhoto(t.dataset.path,"","");
});
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype, cache=False):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if cache:
            self.send_header("Cache-Control", "public, max-age=604800, immutable")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        params = parse_qs(url.query)
        if url.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif url.path == "/api/stats":
            self._send(200, json.dumps(sc.stats()), "application/json; charset=utf-8")
        elif url.path == "/api/search":
            query = (params.get("q") or [""])[0]
            with_media = (params.get("media") or [""])[0] == "1"
            sources = sc.DEFAULT_SOURCES | {"media"} if with_media else sc.DEFAULT_SOURCES
            self._send(200, json.dumps(sc.search(query, sources=sources)),
                       "application/json; charset=utf-8")
        elif url.path == "/api/duplicates":
            dup_path = config.OUTPUT_DIR / "near_duplicates.json"
            if not dup_path.exists():
                body = {"error": "no report yet — python scripts/find_near_duplicates.py"}
            else:
                body = json.loads(dup_path.read_text(encoding="utf-8"))
            self._send(200, json.dumps(body), "application/json; charset=utf-8")
        elif url.path == "/thumb":
            self._serve_thumb(params)
        else:
            self._send(404, "not found", "text/plain; charset=utf-8")

    def _serve_thumb(self, params):
        try:
            size = min(2000, max(64, int((params.get("s") or ["320"])[0])))
        except ValueError:
            size = 320
        photo = safe_photo_path((params.get("p") or [""])[0])
        if photo is None:
            self._send(404, "not found", "text/plain; charset=utf-8")
            return
        try:
            data = make_thumbnail(photo, size)
        except Exception as exc:
            self._send(500, f"thumbnail error: {exc}", "text/plain; charset=utf-8")
            return
        if data is None:
            self._send(415, "unsupported image format", "text/plain; charset=utf-8")
        else:
            self._send(200, data, "image/jpeg", cache=True)

    def log_message(self, *args):
        pass  # one line per thumbnail is noise, not a log


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else config.WEB_SEARCH_PORT
    if not ALLOWED_ROOTS:
        sys.exit("PHOTO_ROOT is not set (see .env.example) - nothing to serve images from")
    if not sc.INDEX_PATH.exists():
        print(f"WARNING: no index at {sc.INDEX_PATH} - run: python scripts/build_search_index.py")
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Archive search: http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
