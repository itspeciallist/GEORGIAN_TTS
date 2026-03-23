"""
Georgian TTS Pro — Complete App
=================================
Tabs:
  1. TTS          — Text → Georgian MP3
  2. Video Dub    — English video → Georgian dubbed video

Install:
    pip install flask edge-tts pydub librosa soundfile numpy scipy
    pip install openai-whisper deep-translator moviepy

    sudo apt install ffmpeg

Run:
    python app.py  →  http://localhost:5000
"""

import os, re, asyncio, tempfile, time, json, subprocess
import numpy as np
import librosa
import soundfile as sf
import edge_tts
import whisper
from deep_translator import GoogleTranslator
from flask import Flask, request, jsonify, send_file, render_template_string
from pydub import AudioSegment
try:
    from moviepy.editor import VideoFileClip, AudioFileClip
except ImportError:
    from moviepy import VideoFileClip, AudioFileClip

app       = Flask(__name__)
OUT_MP3   = "output.mp3"
OUT_WAV   = "_tts_22k.wav"
OUT_VIDEO = "output_ka.mp4"
TARGET_SR = 22050
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

VOICES = {
    "male":   "ka-GE-GiorgiNeural",
    "female": "ka-GE-EkaNeural",
}

# ─── Whisper model (loaded once) ─────────────────────────
_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        print("⏳ Loading Whisper model (base)...")
        _whisper_model = whisper.load_model("base")
        print("✅ Whisper ready")
    return _whisper_model

# ═══════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="ka">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ქართული TTS Pro</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+Georgian:wght@300;400;700;900&family=JetBrains+Mono:wght@300;400;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#080810;--s1:#0e0e1a;--s2:#13131f;--border:#252535;
  --gold:#d4a843;--purple:#8b6be8;--cyan:#43c9d4;
  --red:#e85050;--green:#50e878;--blue:#4a9eff;
  --text:#ddd8cc;--muted:#5a5870;--r:10px;
  --mono:'JetBrains Mono',monospace;--serif:'Noto Serif Georgian',serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--mono);min-height:100vh;overflow-x:hidden}
body::after{content:'';position:fixed;inset:0;pointer-events:none;
  background:radial-gradient(ellipse 60% 40% at 15% 20%,rgba(139,107,232,.07) 0%,transparent 60%),
             radial-gradient(ellipse 50% 35% at 85% 80%,rgba(212,168,67,.07) 0%,transparent 60%)}

/* ── layout ── */
.shell{position:relative;z-index:1;max-width:1000px;margin:0 auto;padding:36px 20px 100px}
header{margin-bottom:36px;display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap}
.eyebrow{font-size:9px;letter-spacing:4px;color:var(--gold);text-transform:uppercase;margin-bottom:8px;opacity:.8}
h1{font-family:var(--serif);font-size:clamp(1.6rem,3.5vw,2.4rem);font-weight:900;
  background:linear-gradient(135deg,var(--text) 30%,var(--gold));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border:1px solid var(--border);
  border-radius:20px;font-size:9px;letter-spacing:2px;color:var(--muted)}
.dot{width:6px;height:6px;border-radius:50%;background:var(--green);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}

/* ── TABS ── */
.tabs{display:flex;gap:4px;margin-bottom:24px;background:var(--s1);
  padding:5px;border-radius:var(--r);border:1px solid var(--border)}
.tab{flex:1;padding:11px 16px;border:none;border-radius:7px;cursor:pointer;
  font-family:var(--mono);font-size:11px;letter-spacing:1.5px;text-transform:uppercase;
  background:transparent;color:var(--muted);transition:all .2s;font-weight:700}
.tab:hover{color:var(--text)}
.tab.active{background:var(--s2);color:var(--gold);border:1px solid var(--border)}
.tab-icon{margin-right:6px}
.page{display:none}.page.active{display:block}

/* ── sections ── */
.sec{background:var(--s2);border:1px solid var(--border);border-radius:var(--r);
  padding:24px;margin-bottom:16px;transition:border-color .2s}
.sec:hover{border-color:rgba(212,168,67,.2)}
.sh{font-size:9px;letter-spacing:3px;text-transform:uppercase;color:var(--gold);
  margin-bottom:18px;display:flex;align-items:center;gap:10px}
.sh::before{content:'';display:block;width:20px;height:1px;background:var(--gold)}
.sub{font-size:10px;color:var(--muted);margin-top:-12px;margin-bottom:16px}

/* ── inputs ── */
textarea{width:100%;min-height:150px;background:var(--s1);border:1px solid var(--border);
  border-radius:8px;color:var(--text);font-family:var(--serif);font-size:14px;
  line-height:1.8;padding:14px;resize:vertical;outline:none;transition:border-color .2s,box-shadow .2s}
textarea:focus{border-color:var(--gold);box-shadow:0 0 0 3px rgba(212,168,67,.08)}
textarea::placeholder{color:var(--muted)}
.row-meta{display:flex;justify-content:space-between;margin-top:6px;font-size:10px;color:var(--muted)}

select{width:100%;background:var(--s1);border:1px solid var(--border);border-radius:8px;
  color:var(--text);font-family:var(--mono);font-size:12px;padding:10px 12px;
  outline:none;cursor:pointer;-webkit-appearance:none;transition:border-color .2s}
select:focus{border-color:var(--gold)}

/* ── voice cards ── */
.vcards{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.vc input{position:absolute;opacity:0;pointer-events:none}
.vc label{display:flex;align-items:center;gap:12px;padding:14px 16px;background:var(--s1);
  border:1px solid var(--border);border-radius:8px;cursor:pointer;transition:all .2s}
.vc label:hover{border-color:var(--gold)}
.vc input:checked+label{border-color:var(--gold);background:rgba(212,168,67,.06);box-shadow:0 0 20px rgba(212,168,67,.1)}
.vi{font-size:22px}.vn strong{display:block;font-size:12px;margin-bottom:1px}
.vn small{font-size:9px;color:var(--muted);letter-spacing:1px}

/* ── sliders ── */
.sl-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:18px}
.sll{display:flex;justify-content:space-between;font-size:9px;letter-spacing:1.5px;
  text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.slv{color:var(--gold);font-weight:700;font-size:11px}
input[type=range]{-webkit-appearance:none;width:100%;height:3px;background:var(--border);border-radius:2px;outline:none;cursor:pointer}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;
  background:var(--gold);border-radius:50%;box-shadow:0 0 8px rgba(212,168,67,.5);transition:transform .15s}
input[type=range]::-webkit-slider-thumb:hover{transform:scale(1.3)}
.sld{font-size:9px;color:var(--muted);margin-top:5px}

/* ── toggles ── */
.trow{display:flex;align-items:center;justify-content:space-between;padding:10px 0;border-bottom:1px solid var(--border)}
.trow:last-child{border-bottom:none}
.ti strong{display:block;font-size:11px;margin-bottom:2px}.ti small{font-size:9px;color:var(--muted)}
.tog{position:relative;width:38px;height:20px;flex-shrink:0}
.tog input{opacity:0;width:0;height:0}
.ts{position:absolute;inset:0;background:var(--border);border-radius:10px;cursor:pointer;transition:.2s}
.ts::before{content:'';position:absolute;width:14px;height:14px;top:3px;left:3px;background:#fff;border-radius:50%;transition:.2s}
.tog input:checked+.ts{background:var(--gold)}
.tog input:checked+.ts::before{transform:translateX(18px)}

/* ── recording ── */
.rzone{border:2px dashed var(--border);border-radius:var(--r);padding:20px;text-align:center;transition:all .3s}
.rzone.active{border-color:var(--red);background:rgba(232,80,80,.04)}
.rzone.done{border-color:var(--green);background:rgba(80,232,120,.04)}
.rbtn{display:inline-flex;align-items:center;gap:8px;padding:11px 26px;border-radius:8px;border:none;cursor:pointer;
  font-family:var(--mono);font-size:11px;letter-spacing:2px;text-transform:uppercase;transition:all .2s;font-weight:700}
.rbtn.s{background:rgba(232,80,80,.15);color:var(--red);border:1px solid rgba(232,80,80,.3)}
.rbtn.s:hover{background:rgba(232,80,80,.25)}
.rbtn.r{background:rgba(80,232,120,.15);color:var(--green);border:1px solid rgba(80,232,120,.3);animation:rp 1s infinite}
@keyframes rp{0%,100%{box-shadow:0 0 0 0 rgba(232,80,80,.3)}50%{box-shadow:0 0 0 8px rgba(232,80,80,0)}}
.rtimer{font-size:26px;font-weight:700;color:var(--red);margin:8px 0}
.rhint{font-size:10px;color:var(--muted);margin-top:8px}
#vpreview{margin-top:12px;width:100%;border-radius:6px;accent-color:var(--gold);display:none}
.abox{margin-top:12px;background:var(--s1);border:1px solid var(--border);border-radius:8px;padding:14px 16px;display:none}
.atitle{font-size:9px;letter-spacing:2px;color:var(--cyan);text-transform:uppercase;margin-bottom:10px}
.agrid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.ai{text-align:center}.av{font-size:18px;font-weight:700;color:var(--cyan)}.ak{font-size:8px;color:var(--muted);letter-spacing:1px;margin-top:2px}
.abtn{margin-top:12px;width:100%;padding:10px;background:rgba(67,201,212,.1);border:1px solid rgba(67,201,212,.3);
  color:var(--cyan);border-radius:6px;font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;cursor:pointer;transition:.2s}
.abtn:hover{background:rgba(67,201,212,.2)}

/* ── VIDEO UPLOAD ── */
.drop-zone{border:2px dashed var(--border);border-radius:var(--r);padding:40px 24px;
  text-align:center;cursor:pointer;transition:all .3s;position:relative}
.drop-zone:hover,.drop-zone.drag{border-color:var(--blue);background:rgba(74,158,255,.04)}
.drop-zone input[type=file]{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.dz-icon{font-size:40px;margin-bottom:10px}
.dz-title{font-size:13px;margin-bottom:4px}
.dz-sub{font-size:10px;color:var(--muted)}
.file-info{margin-top:12px;background:var(--s1);border:1px solid var(--border);border-radius:8px;
  padding:12px 14px;display:none;text-align:left}
.fi-name{font-size:12px;margin-bottom:2px}.fi-meta{font-size:10px;color:var(--muted)}

/* ── pipeline steps (video) ── */
.pipeline{display:flex;gap:0;margin-bottom:20px;overflow-x:auto}
.pstep-v{flex:1;min-width:90px;text-align:center;padding:10px 4px;position:relative}
.pstep-v::after{content:'→';position:absolute;right:-8px;top:50%;transform:translateY(-50%);
  color:var(--muted);font-size:10px}
.pstep-v:last-child::after{display:none}
.pstep-v .icon{font-size:20px;margin-bottom:4px}
.pstep-v .label{font-size:8px;letter-spacing:1px;color:var(--muted);text-transform:uppercase}
.pstep-v.active .label{color:var(--gold)}
.pstep-v.done .label{color:var(--green)}
.pstep-v.active .icon,.pstep-v.done .icon{animation:none}

/* ── transcript box ── */
.transcript-box{background:var(--s1);border:1px solid var(--border);border-radius:8px;
  padding:14px;font-family:var(--serif);font-size:13px;line-height:1.7;
  max-height:200px;overflow-y:auto;display:none}
.t-en{color:var(--text);margin-bottom:12px}
.t-ka{color:var(--gold)}
.t-label{font-size:8px;letter-spacing:2px;color:var(--muted);text-transform:uppercase;margin-bottom:6px;font-family:var(--mono)}

/* ── buttons ── */
.btn{padding:13px 40px;border:none;border-radius:8px;font-family:var(--mono);font-size:12px;
  font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;transition:all .2s}
.btn-gold{background:linear-gradient(135deg,var(--gold),#b8892a);color:#080810}
.btn-gold:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(212,168,67,.25)}
.btn-gold:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-blue{background:linear-gradient(135deg,var(--blue),#2a7fe0);color:#fff}
.btn-blue:hover{transform:translateY(-2px);box-shadow:0 10px 30px rgba(74,158,255,.25)}
.btn-blue:disabled{opacity:.4;cursor:not-allowed;transform:none}
.wrap-btn{text-align:center;margin:8px 0}

/* ── progress ── */
.prog-wrap{display:none;margin-top:20px}
.pbar-bg{height:3px;background:var(--border);border-radius:2px;overflow:hidden;margin-bottom:8px}
.pbar-fill{height:100%;background:linear-gradient(90deg,var(--purple),var(--gold));
  border-radius:2px;width:0;transition:width .5s}
.prog-msg{font-size:10px;color:var(--muted);text-align:center;letter-spacing:1px}

/* ── result ── */
.res-card{background:rgba(80,232,120,.04);border:1px solid rgba(80,232,120,.2);
  border-radius:var(--r);padding:24px;text-align:center;margin-top:20px;display:none}
.res-tick{font-size:36px;margin-bottom:8px}
.res-info{font-size:10px;letter-spacing:2px;color:var(--green);margin-bottom:14px}
audio,video{width:100%;border-radius:6px;accent-color:var(--gold);margin-bottom:12px}
.dl-row{display:flex;gap:10px;justify-content:center;flex-wrap:wrap}
.dl-btn{padding:10px 20px;border:1px solid var(--green);color:var(--green);
  background:transparent;border-radius:6px;font-family:var(--mono);font-size:10px;
  letter-spacing:2px;cursor:pointer;text-decoration:none;display:inline-block;transition:.2s}
.dl-btn:hover{background:rgba(80,232,120,.1)}

/* ── error ── */
.err{display:none;margin-top:12px;padding:12px 16px;border-radius:8px;
  background:rgba(232,80,80,.07);border:1px solid rgba(232,80,80,.25);
  font-size:11px;color:var(--red)}

.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.fmeta{margin-top:14px;display:grid;grid-template-columns:repeat(auto-fill,minmax(100px,1fr));gap:8px}
.fc{background:var(--s1);border:1px solid var(--border);border-radius:6px;padding:8px 10px;text-align:center}
.fv{font-size:14px;font-weight:700}.fk{font-size:8px;color:var(--muted);letter-spacing:1px;margin-top:2px}
@media(max-width:600px){.grid-2,.vcards,.sl-grid,.agrid{grid-template-columns:1fr}
  .pipeline{flex-wrap:wrap}.pstep-v{min-width:70px}}
</style>
</head>
<body>
<div class="shell">

<header>
  <div>
    <div class="eyebrow">Neural Voice · Edge TTS · Whisper</div>
    <h1>ქართული TTS Pro</h1>
  </div>
  <div class="badge"><span class="dot"></span>ka-GE · 22050 Hz</div>
</header>

<!-- ══ TABS ══ -->
<div class="tabs">
  <button class="tab active" onclick="switchTab('tts',this)">
    <span class="tab-icon">🎙</span>TTS / ტექსტი → ხმა
  </button>
  <button class="tab" onclick="switchTab('video',this)">
    <span class="tab-icon">🎬</span>Video Dub / ვიდეო → ქართული
  </button>
</div>

<!-- ══════════════════════════════════════════
     PAGE 1 — TTS
══════════════════════════════════════════ -->
<div id="page-tts" class="page active">

  <div class="sec">
    <div class="sh">ტექსტი / Text</div>
    <textarea id="text" placeholder="ჩაწერეთ ქართული ტექსტი...&#10;Enter Georgian text here..."></textarea>
    <div class="row-meta"><span id="wc">0 სიტყვა</span><span id="cc">0 სიმბოლო</span></div>
  </div>

  <div class="sec">
    <div class="sh">ხმა / Voice</div>
    <div class="vcards">
      <div class="vc"><input type="radio" name="voice" id="vm" value="male" checked>
        <label for="vm"><span class="vi">👨</span><div class="vn"><strong>გიორგი</strong><small>GiorgiNeural · Male</small></div></label></div>
      <div class="vc"><input type="radio" name="voice" id="vf" value="female">
        <label for="vf"><span class="vi">👩</span><div class="vn"><strong>ეკა</strong><small>EkaNeural · Female</small></div></label></div>
    </div>
  </div>

  <div class="sec">
    <div class="sh">ხმის პარამეტრები / Voice Parameters</div>
    <div class="sl-grid">
      <div><div class="sll">სიჩქარე / Rate <span class="slv" id="v-rate">+0%</span></div>
        <input type="range" id="rate" min="-50" max="100" value="-10" step="5">
        <div class="sld">−50% ნელი · +100% სწრაფი</div></div>
      <div><div class="sll">სიხშირე / Pitch <span class="slv" id="v-pitch">+0Hz</span></div>
        <input type="range" id="pitch" min="-30" max="30" value="0" step="1">
        <div class="sld">−30Hz დაბალი · +30Hz მაღალი</div></div>
      <div><div class="sll">მოცულობა / Volume <span class="slv" id="v-volume">+0%</span></div>
        <input type="range" id="volume" min="-50" max="50" value="0" step="5">
        <div class="sld">−50% ჩუმი · +50% ხმამაღალი</div></div>
      <div><div class="sll">პაუზა / Pause <span class="slv" id="v-pause">400ms</span></div>
        <input type="range" id="pause" min="0" max="1500" value="600" step="50">
        <div class="sld">პაუზა წინადადებებს შორის</div></div>
      <div><div class="sll">Pitch Shift <span class="slv" id="v-pshift">0.0 st</span></div>
        <input type="range" id="pshift" min="-12" max="12" value="0" step="0.5">
        <div class="sld">Post-process pitch (semitones)</div></div>
      <div><div class="sll">Time Stretch <span class="slv" id="v-tstretch">1.00×</span></div>
        <input type="range" id="tstretch" min="0.5" max="2.0" value="1.0" step="0.05">
        <div class="sld">Post-process speed</div></div>
    </div>
  </div>

  <div class="sec">
    <div class="sh">გამოსვლა / Output Config</div>
    <div class="grid-2" style="margin-bottom:16px">
      <div><div class="sll" style="margin-bottom:8px">MP3 Bitrate</div>
        <select id="bitrate"><option value="128k">128k</option><option value="192k" selected>192k</option><option value="320k">320k</option></select></div>
      <div><div class="sll" style="margin-bottom:8px">Sample Rate</div>
        <select id="srate"><option value="16000">16000 Hz</option><option value="22050" selected>22050 Hz ★</option><option value="44100">44100 Hz</option></select></div>
    </div>
    <div class="trow"><div class="ti"><strong>Normalize</strong><small>Peak normalization</small></div>
      <label class="tog"><input type="checkbox" id="normalize" checked><span class="ts"></span></label></div>
    <div class="trow"><div class="ti"><strong>Trim Silence</strong><small>Remove silence</small></div>
      <label class="tog"><input type="checkbox" id="trim"><span class="ts"></span></label></div>
    <div class="trow"><div class="ti"><strong>Fade In/Out</strong><small>200ms fades</small></div>
      <label class="tog"><input type="checkbox" id="fade"><span class="ts"></span></label></div>
  </div>

  <div class="sec">
    <div class="sh">ხმის ჩაწერა / Voice Match</div>
    <div class="sub">ჩაიწერეთ თქვენი ხმა → ავტო-მოარგება TTS პარამეტრებს</div>
    <div class="rzone" id="rec-zone">
      <div class="rtimer" id="rec-timer" style="display:none">0:00</div>
      <button class="rbtn s" id="rec-btn" onclick="toggleRec()">🎙 ჩაწერის დაწყება</button>
      <div class="rhint" id="rec-hint">მინ. 3 წამი · ისაუბრეთ ბუნებრივად ქართულად</div>
      <audio id="vpreview" controls></audio>
      <div class="abox" id="analysis-box">
        <div class="atitle">ანალიზის შედეგები</div>
        <div class="agrid">
          <div class="ai"><div class="av" id="a-pitch">—</div><div class="ak">Pitch (Hz)</div></div>
          <div class="ai"><div class="av" id="a-speed">—</div><div class="ak">Speed (×)</div></div>
          <div class="ai"><div class="av" id="a-level">—</div><div class="ak">Level (dB)</div></div>
        </div>
        <button class="abtn" onclick="applyMatch()">⚡ Apply to Parameters</button>
      </div>
    </div>
  </div>

  <div class="wrap-btn">
    <button class="btn btn-gold" id="gen-btn" onclick="generateTTS()">▶ გენერაცია / Generate</button>
  </div>
  <div class="prog-wrap" id="tts-prog">
    <div class="pbar-bg"><div class="pbar-fill" id="tts-pbar"></div></div>
    <div class="prog-msg" id="tts-msg">მუშავდება...</div>
  </div>
  <div class="err" id="tts-err"></div>
  <div class="res-card" id="tts-result">
    <div class="res-tick">✅</div>
    <div class="res-info" id="tts-info"></div>
    <audio id="tts-audio" controls></audio>
    <div class="fmeta" id="tts-meta"></div>
    <div class="dl-row" style="margin-top:14px">
      <a id="tts-dl" class="dl-btn" download="output.mp3">⬇ MP3 გადმოწერა</a>
    </div>
  </div>

</div><!-- /page-tts -->


<!-- ══════════════════════════════════════════
     PAGE 2 — VIDEO DUB
══════════════════════════════════════════ -->
<div id="page-video" class="page">

  <!-- Pipeline visualization -->
  <div class="sec">
    <div class="sh">მუშაობის პრინციპი / Pipeline</div>
    <div class="pipeline">
      <div class="pstep-v" id="vp1"><div class="icon">🎬</div><div class="label">Upload Video</div></div>
      <div class="pstep-v" id="vp2"><div class="icon">🎧</div><div class="label">Extract Audio</div></div>
      <div class="pstep-v" id="vp3"><div class="icon">📝</div><div class="label">Transcribe</div></div>
      <div class="pstep-v" id="vp4"><div class="icon">🌐</div><div class="label">Translate</div></div>
      <div class="pstep-v" id="vp5"><div class="icon">🗣</div><div class="label">Georgian TTS</div></div>
      <div class="pstep-v" id="vp6"><div class="icon">🎞</div><div class="label">Merge Video</div></div>
    </div>
  </div>

  <!-- Video upload -->
  <div class="sec">
    <div class="sh">ვიდეო / Upload Video</div>
    <div class="drop-zone" id="drop-zone">
      <input type="file" id="video-file" accept="video/*" onchange="onFileSelect(this)">
      <div class="dz-icon">🎬</div>
      <div class="dz-title">ვიდეო ჩასვით ან დააჭირეთ</div>
      <div class="dz-sub">MP4 · MKV · AVI · MOV · WebM — max 500MB</div>
    </div>
    <div class="file-info" id="file-info">
      <div class="fi-name" id="fi-name">—</div>
      <div class="fi-meta" id="fi-meta">—</div>
    </div>
  </div>

  <!-- Dub settings -->
  <div class="sec">
    <div class="sh">დუბლირების პარამეტრები / Dub Settings</div>
    <div class="grid-2" style="margin-bottom:20px">
      <div>
        <div class="sll" style="margin-bottom:8px">ქართული ხმა / Georgian Voice</div>
        <div class="vcards">
          <div class="vc"><input type="radio" name="dvoice" id="dvm" value="male" checked>
            <label for="dvm"><span class="vi">👨</span><div class="vn"><strong>გიორგი</strong><small>Male</small></div></label></div>
          <div class="vc"><input type="radio" name="dvoice" id="dvf" value="female">
            <label for="dvf"><span class="vi">👩</span><div class="vn"><strong>ეკა</strong><small>Female</small></div></label></div>
        </div>
      </div>
      <div>
        <div class="sll" style="margin-bottom:8px">Whisper მოდელი</div>
        <select id="whisper-model">
          <option value="tiny">tiny — სწრაფი (დაბალი ხარისხი)</option>
          <option value="base" selected>base — ბალანსი ★</option>
          <option value="small">small — მაღალი ხარისხი (ნელი)</option>
          <option value="medium">medium — საუკეთესო (ძალიან ნელი)</option>
        </select>
        <div class="sld" style="margin-top:8px">პირველ გაშვებაზე ჩამოიტვირთება</div>
      </div>
    </div>

    <div class="sl-grid" style="margin-bottom:16px">
      <div><div class="sll">TTS სიჩქარე <span class="slv" id="vd-rate">+0%</span></div>
        <input type="range" id="drate" min="-50" max="100" value="0" step="5">
        <div class="sld">TTS reading speed</div></div>
      <div><div class="sll">TTS Pitch <span class="slv" id="vd-pitch">+0Hz</span></div>
        <input type="range" id="dpitch" min="-30" max="30" value="0" step="1">
        <div class="sld">TTS pitch</div></div>
    </div>

    <div class="trow"><div class="ti"><strong>ორიგინალური ხმა</strong><small>შეინახოს ფონური ხმა (დაბალ მოცულობაზე)</small></div>
      <label class="tog"><input type="checkbox" id="keep-audio" checked><span class="ts"></span></label></div>
    <div class="trow"><div class="ti"><strong>სუბტიტრები</strong><small>შეიქმნას .srt სუბტიტრების ფაილი</small></div>
      <label class="tog"><input type="checkbox" id="make-srt" checked><span class="ts"></span></label></div>
  </div>

  <!-- Transcript preview -->
  <div class="sec" id="transcript-sec" style="display:none">
    <div class="sh">ტრანსკრიფცია / Transcript</div>
    <div class="transcript-box" id="transcript-box">
      <div class="t-label">English (Original)</div>
      <div class="t-en" id="t-en">—</div>
      <div class="t-label" style="margin-top:12px">ქართული (Translated)</div>
      <div class="t-ka" id="t-ka">—</div>
    </div>
  </div>

  <div class="wrap-btn">
    <button class="btn btn-blue" id="dub-btn" onclick="startDub()" disabled>🎬 დუბლირება / Start Dubbing</button>
  </div>

  <div class="prog-wrap" id="vid-prog">
    <div class="pbar-bg"><div class="pbar-fill" id="vid-pbar"></div></div>
    <div class="prog-msg" id="vid-msg">მუშავდება...</div>
  </div>
  <div class="err" id="vid-err"></div>

  <div class="res-card" id="vid-result">
    <div class="res-tick">🎬</div>
    <div class="res-info" id="vid-info">ვიდეო მზადაა</div>
    <video id="vid-player" controls></video>
    <div class="dl-row">
      <a id="vid-dl" class="dl-btn" download="output_ka.mp4">⬇ MP4 გადმოწერა</a>
      <a id="srt-dl" class="dl-btn" download="subtitles_ka.srt" style="display:none">⬇ SRT სუბტიტრები</a>
    </div>
  </div>

</div><!-- /page-video -->
</div><!-- /shell -->

<script>
/* ── Tab switching ── */
function switchTab(id, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('page-' + id).classList.add('active');
}

/* ── TTS counters ── */
const ta = document.getElementById('text');
ta.addEventListener('input', () => {
  const v = ta.value;
  document.getElementById('cc').textContent = v.length + ' სიმბოლო';
  document.getElementById('wc').textContent = (v.trim() ? v.trim().split(/\s+/).length : 0) + ' სიტყვა';
});

/* ── Slider binding ── */
function bind(id, sfx, dec=0) {
  const el = document.getElementById(id);
  const vl = document.getElementById('v-' + id);
  if (!el || !vl) return;
  const up = () => { const v = parseFloat(el.value); vl.textContent = (v >= 0 ? '+' : '') + v.toFixed(dec) + sfx; };
  el.addEventListener('input', up); up();
}
['rate','pitch','volume','pause'].forEach(id => bind(id, id==='pause'?'ms':'%'));
bind('pshift', ' st', 1); bind('tstretch', '×', 2);

function bind2(id, sfx, vId) {
  const el = document.getElementById(id), vl = document.getElementById(vId);
  if (!el || !vl) return;
  const up = () => { const v = parseFloat(el.value); vl.textContent = (v >= 0 ? '+' : '') + v.toFixed(0) + sfx; };
  el.addEventListener('input', up); up();
}
bind2('drate', '%', 'vd-rate'); bind2('dpitch', 'Hz', 'vd-pitch');

/* ══════════════════════════════════════════
   RECORDING (AudioContext → WAV — no webm)
══════════════════════════════════════════ */
let audioCtx, mediaStream, scriptProc;
let recBufs = [], recSR = 44100, isRec = false, recSecs = 0, recTimer;
let analysisData = null;

function f32toWAV(bufs, sr) {
  let n = 0; bufs.forEach(b => n += b.length);
  const flat = new Float32Array(n); let o = 0;
  bufs.forEach(b => { flat.set(b, o); o += b.length; });
  const ab = new ArrayBuffer(44 + n * 2), v = new DataView(ab);
  const u32 = (p, x) => v.setUint32(p, x, true);
  const u16 = (p, x) => v.setUint16(p, x, true);
  const str = (p, s) => { for (let i = 0; i < s.length; i++) v.setUint8(p + i, s.charCodeAt(i)); };
  str(0,'RIFF'); u32(4,36+n*2); str(8,'WAVE');
  str(12,'fmt '); u32(16,16); u16(20,1); u16(22,1);
  u32(24,sr); u32(28,sr*2); u16(32,2); u16(34,16);
  str(36,'data'); u32(40,n*2);
  let p = 44;
  for (let i = 0; i < flat.length; i++) {
    const s = Math.max(-1, Math.min(1, flat[i]));
    v.setInt16(p, s < 0 ? s * 0x8000 : s * 0x7FFF, true); p += 2;
  }
  return new Blob([ab], { type: 'audio/wav' });
}

async function toggleRec() {
  const btn = document.getElementById('rec-btn');
  const zone = document.getElementById('rec-zone');
  const hint = document.getElementById('rec-hint');
  if (!isRec) {
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      recSR = audioCtx.sampleRate; recBufs = []; recSecs = 0; isRec = true;
      const src = audioCtx.createMediaStreamSource(mediaStream);
      scriptProc = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProc.onaudioprocess = e => { if (isRec) recBufs.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
      src.connect(scriptProc); scriptProc.connect(audioCtx.destination);
      clearInterval(recTimer);
      recTimer = setInterval(() => {
        recSecs++;
        const m = Math.floor(recSecs / 60), s = recSecs % 60;
        document.getElementById('rec-timer').textContent = m + ':' + String(s).padStart(2, '0');
      }, 1000);
      document.getElementById('rec-timer').style.display = 'block';
      btn.className = 'rbtn r'; btn.textContent = '⏹ გაჩერება / Stop';
      zone.className = 'rzone active'; hint.textContent = '● ჩაწერა მიმდინარეობს...';
    } catch (e) { alert('მიკროფონი: ' + e.message); }
  } else {
    isRec = false; clearInterval(recTimer);
    if (scriptProc) scriptProc.disconnect();
    if (audioCtx) audioCtx.close();
    if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
    document.getElementById('rec-timer').style.display = 'none';
    btn.className = 'rbtn s'; btn.textContent = '🎙 თავიდან ჩაწერა';
    hint.textContent = '⏳ WAV კოდირება...';
    const wav = f32toWAV(recBufs, recSR);
    document.getElementById('vpreview').src = URL.createObjectURL(wav);
    document.getElementById('vpreview').style.display = 'block';
    zone.className = 'rzone done';
    hint.textContent = '✅ ჩაწერა დასრულდა · ანალიზი...';
    await doAnalyze(wav);
  }
}

async function doAnalyze(blob) {
  const hint = document.getElementById('rec-hint');
  const fd = new FormData(); fd.append('audio', blob, 'rec.wav');
  try {
    hint.textContent = '⏳ ანალიზი...';
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    const d = await res.json();
    if (d.error) { hint.textContent = '⚠ ' + d.error; return; }
    analysisData = d;
    document.getElementById('a-pitch').textContent = d.pitch_hz.toFixed(0);
    document.getElementById('a-speed').textContent = d.speed_ratio.toFixed(2);
    document.getElementById('a-level').textContent = d.level_db.toFixed(1);
    document.getElementById('analysis-box').style.display = 'block';
    hint.textContent = '✅ ანალიზი დასრულდა';
  } catch (e) { hint.textContent = '⚠ ' + e.message; }
}

function applyMatch() {
  if (!analysisData) return;
  document.getElementById('pshift').value = analysisData.suggested_pshift;
  document.getElementById('tstretch').value = analysisData.suggested_tstretch;
  document.getElementById('volume').value = analysisData.suggested_vol;
  ['pshift','tstretch','volume'].forEach(id => document.getElementById(id).dispatchEvent(new Event('input')));
  const b = document.querySelector('.abtn');
  b.textContent = '✅ გამოყენებულია!';
  setTimeout(() => b.textContent = '⚡ Apply to Parameters', 2500);
}

/* ══════════════════════════════════════════
   TTS GENERATE
══════════════════════════════════════════ */
async function generateTTS() {
  const text = ta.value.trim();
  if (!text) { showErr('tts-err', 'გთხოვთ შეიყვანოთ ტექსტი'); return; }
  document.getElementById('gen-btn').disabled = true;
  document.getElementById('tts-result').style.display = 'none';
  document.getElementById('tts-err').style.display = 'none';
  setProg('tts-prog', 'tts-pbar', 'tts-msg', true, 10, 'სინთეზი...');
  try {
    const p = {
      text,
      voice: document.querySelector('input[name=voice]:checked').value,
      rate: +document.getElementById('rate').value,
      pitch: +document.getElementById('pitch').value,
      volume: +document.getElementById('volume').value,
      pause: +document.getElementById('pause').value,
      pshift: +document.getElementById('pshift').value,
      tstretch: +document.getElementById('tstretch').value,
      bitrate: document.getElementById('bitrate').value,
      srate: +document.getElementById('srate').value,
      normalize: document.getElementById('normalize').checked,
      trim: document.getElementById('trim').checked,
      fade: document.getElementById('fade').checked,
    };
    setP('tts-pbar','tts-msg', 40, 'სინთეზი...');
    const res = await fetch('/generate', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(p) });
    setP('tts-pbar','tts-msg', 85, 'Post-FX...');
    const d = await res.json();
    if (!res.ok || d.error) { showErr('tts-err', d.error || 'Error'); return; }
    setP('tts-pbar','tts-msg', 100, 'დასრულდა!');
    await sleep(400);
    setProg('tts-prog','tts-pbar','tts-msg', false);
    const ts = Date.now();
    document.getElementById('tts-audio').src = '/download/mp3?t=' + ts;
    document.getElementById('tts-dl').href = '/download/mp3?t=' + ts;
    document.getElementById('tts-info').textContent = `✓ ${d.duration_sec}s · ${d.size_mb}MB · ${p.srate}Hz · ${p.bitrate}`;
    document.getElementById('tts-result').style.display = 'block';
  } catch (e) { showErr('tts-err', e.message); }
  finally { document.getElementById('gen-btn').disabled = false; }
}

/* ══════════════════════════════════════════
   VIDEO FILE SELECT
══════════════════════════════════════════ */
let selectedFile = null;

function onFileSelect(input) {
  const f = input.files[0];
  if (!f) return;
  selectedFile = f;
  document.getElementById('fi-name').textContent = f.name;
  document.getElementById('fi-meta').textContent =
    `${(f.size/(1024*1024)).toFixed(1)} MB · ${f.type || 'video'}`;
  document.getElementById('file-info').style.display = 'block';
  document.getElementById('drop-zone').style.borderColor = 'var(--blue)';
  document.getElementById('dub-btn').disabled = false;
}

/* drag-drop */
const dz = document.getElementById('drop-zone');
['dragenter','dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('drag'); }));
['dragleave','drop'].forEach(ev => dz.addEventListener(ev, () => dz.classList.remove('drag')));
dz.addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) { document.getElementById('video-file').files = e.dataTransfer.files; onFileSelect(document.getElementById('video-file')); }
});

/* ══════════════════════════════════════════
   VIDEO DUB
══════════════════════════════════════════ */
async function startDub() {
  if (!selectedFile) { showErr('vid-err', 'გთხოვთ ატვირთოთ ვიდეო'); return; }

  document.getElementById('dub-btn').disabled = true;
  document.getElementById('vid-result').style.display = 'none';
  document.getElementById('vid-err').style.display = 'none';
  document.getElementById('transcript-sec').style.display = 'none';
  setVPipeline(0);

  const fd = new FormData();
  fd.append('video', selectedFile);
  fd.append('voice', document.querySelector('input[name=dvoice]:checked').value);
  fd.append('whisper_model', document.getElementById('whisper-model').value);
  fd.append('rate', document.getElementById('drate').value);
  fd.append('pitch', document.getElementById('dpitch').value);
  fd.append('keep_audio', document.getElementById('keep-audio').checked);
  fd.append('make_srt', document.getElementById('make-srt').checked);

  setProg('vid-prog','vid-pbar','vid-msg', true, 5, 'ვიდეო იტვირთება...');
  setVPipeline(1);

  try {
    const res = await fetch('/dub', { method:'POST', body:fd });
    setP('vid-pbar','vid-msg', 60, 'სინთეზი / დუბლირება...');
    setVPipeline(4);
    const d = await res.json();
    if (!res.ok || d.error) { showErr('vid-err', d.error || 'Server error'); return; }

    setP('vid-pbar','vid-msg', 100, 'დასრულდა!');
    setVPipeline(6);

    /* show transcript */
    if (d.transcript_en) {
      document.getElementById('t-en').textContent = d.transcript_en;
      document.getElementById('t-ka').textContent = d.transcript_ka;
      document.getElementById('transcript-box').style.display = 'block';
      document.getElementById('transcript-sec').style.display = 'block';
    }

    await sleep(400);
    setProg('vid-prog','vid-pbar','vid-msg', false);
    const ts = Date.now();
    document.getElementById('vid-player').src = '/download/video?t=' + ts;
    document.getElementById('vid-dl').href = '/download/video?t=' + ts;
    document.getElementById('vid-info').textContent =
      `✓ ${d.duration}s · ${d.size_mb}MB · ${selectedFile.name} → output_ka.mp4`;

    if (d.has_srt) {
      document.getElementById('srt-dl').style.display = 'inline-block';
      document.getElementById('srt-dl').href = '/download/srt?t=' + ts;
    }

    document.getElementById('vid-result').style.display = 'block';
  } catch (e) { showErr('vid-err', e.message); }
  finally { document.getElementById('dub-btn').disabled = false; }
}

function setVPipeline(step) {
  for (let i = 1; i <= 6; i++) {
    const el = document.getElementById('vp' + i);
    el.className = 'pstep-v' + (i < step ? ' done' : i === step ? ' active' : '');
  }
}

/* ── Helpers ── */
function setProg(wrapId, barId, msgId, show, pct=0, msg='') {
  document.getElementById(wrapId).style.display = show ? 'block' : 'none';
  if (show) { setP(barId, msgId, pct, msg); }
}
function setP(barId, msgId, pct, msg) {
  document.getElementById(barId).style.width = pct + '%';
  document.getElementById(msgId).textContent = msg;
}
function showErr(id, msg) {
  setProg(id.replace('err','prog'), id.replace('err','pbar'), id.replace('err','msg'), false);
  document.getElementById(id.replace('err','prog')).style.display = 'none';
  const e = document.getElementById(id); e.textContent = '❌ ' + msg; e.style.display = 'block';
}
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
</script>
</body>
</html>"""

# ═══════════════════════════════════════════════════════════
#  TTS HELPERS
# ═══════════════════════════════════════════════════════════

def ensure_punct(text):
    """Add period if text has no ending punctuation — improves TTS prosody."""
    t = text.strip()
    if t and t[-1] not in '.!?…,;:':
        t += '.' 
    return t

def split_chunks(text, max_chars=1000):
    """Split by blank lines (paragraphs) first, then by sentence endings."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        paragraphs = [l.strip() for l in text.splitlines() if l.strip()]
    chunks = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            sentences = re.split(r'(?<=[.!?…])\s+', para)
            cur = ""
            for s in sentences:
                if len(cur) + len(s) + 1 <= max_chars:
                    cur += (" " if cur else "") + s
                else:
                    if cur: chunks.append(cur)
                    cur = s
            if cur: chunks.append(cur)
    return chunks

async def synth_async(text, path, voice, rate, pitch, volume):
    c = edge_tts.Communicate(text, voice,
        rate=f"{int(rate):+d}%", pitch=f"{int(pitch):+d}Hz", volume=f"{int(volume):+d}%")
    await c.save(path)

def build_tts_audio(p):
    """
    Pure TTS pipeline — NO speed change, NO pitch shift, NO processing.
    edge-tts output is used exactly as generated.
    Chunks are joined with silence pauses only.
    """
    voice    = VOICES.get(p["voice"], VOICES["male"])
    chunks   = split_chunks(p["text"])
    pause_ms = int(p.get("pause", 600))
    combined = AudioSegment.empty()
    silence  = AudioSegment.silent(duration=pause_ms)

    for i, chunk in enumerate(chunks):
        tmp_mp3 = f"_chunk_{i:04d}.mp3"
        # rate=0, pitch=0, volume=0 → original edge-tts output unchanged
        asyncio.run(synth_async(ensure_punct(chunk), tmp_mp3, voice, 0, 0, 0))
        combined += AudioSegment.from_mp3(tmp_mp3)
        os.remove(tmp_mp3)
        if i < len(chunks) - 1:
            combined += silence

    combined.export(OUT_MP3, format="mp3", bitrate=p.get("bitrate", "192k"))

    return len(chunks), round(len(combined) / 1000, 2)


# ═══════════════════════════════════════════════════════════
#  VOICE ANALYSIS
# ═══════════════════════════════════════════════════════════

def analyze_voice(wav_path):
    audio, sr = librosa.load(wav_path, sr=22050, mono=True)
    f0, voiced, _ = librosa.pyin(audio, sr=sr,
        fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    vf0      = f0[voiced & ~np.isnan(f0)] if f0 is not None else []
    pitch_hz = float(np.median(vf0)) if len(vf0) > 0 else 150.0
    vr       = float(np.sum(voiced)/len(voiced)) if len(voiced) > 0 else 0.5
    speed    = round(vr * 2, 2)
    rms      = float(np.sqrt(np.mean(audio**2)))
    level_db = float(20 * np.log10(rms + 1e-9))
    st       = 12 * np.log2(max(pitch_hz, 1) / 130.0)
    return {
        "pitch_hz":           round(pitch_hz, 1),
        "speed_ratio":        speed,
        "level_db":           round(level_db, 1),
        "suggested_pshift":   round(float(np.clip(st, -12, 12)), 1),
        "suggested_tstretch": round(float(np.clip(1.0/max(speed, 0.3), 0.6, 1.8)), 2),
        "suggested_vol":      int(np.clip((-18.0 - level_db) * 1.5, -50, 50)),
    }


# ═══════════════════════════════════════════════════════════
#  VIDEO DUBBING PIPELINE
# ═══════════════════════════════════════════════════════════

def make_srt_content(segments_ka):
    """Build SRT file content from translated segments."""
    lines = []
    for i, seg in enumerate(segments_ka, 1):
        start = seg["start"]
        end   = seg["end"]
        text  = seg["text"].strip()
        def fmt(s):
            h = int(s//3600); m = int((s%3600)//60); sec = s % 60
            return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")
        lines.append(f"{i}\n{fmt(start)} --> {fmt(end)}\n{text}\n")
    return "\n".join(lines)


def dub_video(video_path, voice_key, whisper_model_name,
              rate=0, pitch=0, keep_audio=True, make_srt_flag=True):
    """
    Full pipeline:
      1. Extract audio from video
      2. Transcribe with Whisper (English)
      3. Translate each segment to Georgian
      4. Generate Georgian TTS for each segment
      5. Compose dubbed audio (fit to original timing)
      6. Merge back into video
    """
    voice = VOICES.get(voice_key, VOICES["male"])
    work  = tempfile.mkdtemp()

    # ── 1. Extract audio ──────────────────────────────────
    orig_audio_path = os.path.join(work, "orig_audio.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        orig_audio_path
    ], check=True, capture_output=True)

    # ── 2. Transcribe ─────────────────────────────────────
    model = whisper.load_model(whisper_model_name)
    result = model.transcribe(orig_audio_path, language="en", task="transcribe")
    segments_en = result["segments"]
    full_en     = result["text"].strip()

    # ── 3. Translate each segment ─────────────────────────
    translator = GoogleTranslator(source="en", target="ka")
    segments_ka = []
    for seg in segments_en:
        try:
            ka_text = translator.translate(seg["text"].strip())
        except Exception:
            ka_text = seg["text"]  # fallback to English if translation fails
        segments_ka.append({
            "start": seg["start"],
            "end":   seg["end"],
            "text":  ka_text,
        })
    full_ka = " ".join(s["text"] for s in segments_ka)

    # ── 4. Generate TTS for each segment ──────────────────
    # Build a single continuous audio track
    video_clip    = VideoFileClip(video_path)
    total_dur_ms  = int(video_clip.duration * 1000)
    video_clip.close()

    dub_track = AudioSegment.silent(duration=total_dur_ms)

    for i, seg in enumerate(segments_ka):
        txt = seg["text"].strip()
        if not txt:
            continue
        seg_dur_ms  = int((seg["end"] - seg["start"]) * 1000)
        tmp_mp3     = os.path.join(work, f"seg_{i:04d}.mp3")

        asyncio.run(synth_async(ensure_punct(txt), tmp_mp3, voice, rate, pitch, 0))
        tts_seg = AudioSegment.from_mp3(tmp_mp3)

        # Fit TTS to segment duration by time-stretching
        if len(tts_seg) > 0 and abs(len(tts_seg) - seg_dur_ms) > 100:
            stretch_rate = len(tts_seg) / max(seg_dur_ms, 1)
            stretch_rate = max(0.5, min(3.0, stretch_rate))
            tmp_wav = tmp_mp3 + ".wav"
            tts_seg.export(tmp_wav, format="wav")
            audio_arr, sr_ = librosa.load(tmp_wav, sr=TARGET_SR, mono=True)
            if abs(stretch_rate - 1.0) > 0.05:
                audio_arr = librosa.effects.time_stretch(audio_arr, rate=stretch_rate)
            sf.write(tmp_wav, audio_arr, TARGET_SR)
            tts_seg = AudioSegment.from_wav(tmp_wav)
            os.remove(tmp_wav)

        # Insert at correct timestamp
        start_ms = int(seg["start"] * 1000)
        dub_track = dub_track.overlay(tts_seg, position=start_ms)

    # ── 5. Mix original + dubbed ───────────────────────────
    if keep_audio:
        orig_seg = AudioSegment.from_wav(orig_audio_path)
        orig_seg = orig_seg - 18   # lower original by 18dB
        # match lengths
        if len(orig_seg) < len(dub_track):
            orig_seg = orig_seg + AudioSegment.silent(len(dub_track) - len(orig_seg))
        else:
            orig_seg = orig_seg[:len(dub_track)]
        final_audio = dub_track.overlay(orig_seg)
    else:
        final_audio = dub_track

    # Normalize
    pk = final_audio.max_dBFS
    if pk < -0.5:
        final_audio = final_audio.apply_gain(-pk - 0.5)

    dubbed_audio_path = os.path.join(work, "dubbed_audio.wav")
    final_audio.export(dubbed_audio_path, format="wav")

    # ── 6. Merge back into video ───────────────────────────
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", dubbed_audio_path,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        OUT_VIDEO,
    ], check=True, capture_output=True)

    # ── SRT ───────────────────────────────────────────────
    srt_path = None
    if make_srt_flag:
        srt_path = "subtitles_ka.srt"
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(make_srt_content(segments_ka))

    # cleanup
    import shutil
    shutil.rmtree(work, ignore_errors=True)

    size_mb = round(os.path.getsize(OUT_VIDEO) / (1024*1024), 2)
    clip    = VideoFileClip(OUT_VIDEO)
    dur     = round(clip.duration, 1)
    clip.close()

    return {
        "ok":           True,
        "transcript_en": full_en,
        "transcript_ka": full_ka,
        "duration":      dur,
        "size_mb":       size_mb,
        "has_srt":       srt_path is not None,
    }


# ═══════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        p = request.get_json()
        if not p.get("text","").strip():
            return jsonify({"error":"ტექსტი ცარიელია"}), 400
        chunks, dur = build_tts_audio(p)
        return jsonify({"ok":True, "chunks":chunks, "duration_sec":dur,
                        "size_mb": round(os.path.getsize(OUT_MP3)/(1024*1024),2)})
    except Exception as e:
        return jsonify({"error":str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze():
    tmp = None
    try:
        f = request.files.get("audio")
        if not f: return jsonify({"error":"No audio"}), 400
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd); f.save(tmp)
        return jsonify(analyze_voice(tmp))
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    finally:
        if tmp and os.path.exists(tmp): os.remove(tmp)


@app.route("/dub", methods=["POST"])
def dub():
    tmp_video = None
    try:
        vf = request.files.get("video")
        if not vf: return jsonify({"error":"No video file"}), 400

        ext = os.path.splitext(vf.filename)[1] or ".mp4"
        fd, tmp_video = tempfile.mkstemp(suffix=ext)
        os.close(fd); vf.save(tmp_video)

        result = dub_video(
            video_path        = tmp_video,
            voice_key         = request.form.get("voice", "male"),
            whisper_model_name= request.form.get("whisper_model", "base"),
            rate              = int(request.form.get("rate", 0)),
            pitch             = int(request.form.get("pitch", 0)),
            keep_audio        = request.form.get("keep_audio") == "true",
            make_srt_flag     = request.form.get("make_srt") == "true",
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_video and os.path.exists(tmp_video): os.remove(tmp_video)


@app.route("/download/mp3")
def dl_mp3():
    if not os.path.exists(OUT_MP3): return "Not found", 404
    return send_file(OUT_MP3, as_attachment=True, download_name="output.mp3")

@app.route("/download/video")
def dl_video():
    if not os.path.exists(OUT_VIDEO): return "Not found", 404
    return send_file(OUT_VIDEO, as_attachment=True, download_name="output_ka.mp4")

@app.route("/download/srt")
def dl_srt():
    if not os.path.exists("subtitles_ka.srt"): return "Not found", 404
    return send_file("subtitles_ka.srt", as_attachment=True, download_name="subtitles_ka.srt")


if __name__ == "__main__":
    print("=" * 50)
    print("  Georgian TTS Pro  →  http://localhost:5000")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=False)
