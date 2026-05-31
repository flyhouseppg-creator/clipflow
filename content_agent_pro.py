#!/usr/bin/env python3
"""Content Agent Pro v8.5 - Single Render per Config"""
import os, sys, json, threading, webbrowser, subprocess, tempfile, time, re, shutil, math, logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

try:
    from faster_whisper import WhisperModel
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    logger.warning("⚠️ faster-whisper non installato")

ALLOWED_ORIGINS = ["http://localhost:8000", "http://127.0.0.1:8000"]
OUTPUT_DIR = Path.home() / "content_agent_output"
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()

WHISPER_MODEL_INSTANCE = None
def get_whisper():
    global WHISPER_MODEL_INSTANCE
    if WHISPER_MODEL_INSTANCE is None and WHISPER_OK:
        try:
            WHISPER_MODEL_INSTANCE = WhisperModel("base", device="cpu", compute_type="int8")
            logger.info("✅ Whisper model caricato (base)")
        except Exception as e:
            logger.error(f"⚠️ Whisper error: {e}")
    return WHISPER_MODEL_INSTANCE

def _run_ffmpeg_safe(cmd: list, timeout: int = 300) -> bool:
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logger.error("⏱️ FFmpeg timeout")
        return False
    except Exception as e:
        logger.error(f"❌ FFmpeg errore: {e}")
        return False

PLATFORMS = {
    "tiktok":    {"w": 1080, "h": 1920, "max_sec": 60,  "fps": 30, "label": "TikTok",               "icon": "🎵"},
    "instagram": {"w": 1080, "h": 1920, "max_sec": 60,  "fps": 30, "label": "Instagram Reels",      "icon": "📸"},
    "youtube":   {"w": 1080, "h": 1920, "max_sec": 60,  "fps": 30, "label": "YouTube Shorts",       "icon": "▶️"},
    "twitter":   {"w": 1080, "h": 1080, "max_sec": 140, "fps": 30, "label": "Twitter/X",            "icon": "🐦"},
    "linkedin":  {"w": 1920, "h": 1080, "max_sec": 180, "fps": 30, "label": "LinkedIn",             "icon": "💼"},
    "youtube_long": {"w": 1920, "h": 1080, "max_sec": 180, "fps": 30, "label": "YouTube Video (3 min)", "icon": "🎥"},
    "podcast":   {"w": 1080, "h": 1920, "max_sec": 180, "fps": 30, "label": "Podcast Clip (3 min)",  "icon": "🎙️"},
}

def check_ffmpeg() -> bool:
    try: subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=10); return True
    except: return False

def has_audio_stream(path: Path) -> bool:
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)], capture_output=True, text=True, timeout=30)
        return any(s.get("codec_type") == "audio" for s in json.loads(r.stdout).get("streams", []))
    except: return False

def get_video_info(path: Path) -> Dict[str, float]:
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-show_format", str(path)], capture_output=True, text=True, timeout=30)
        info = json.loads(r.stdout)
    except: return {"duration": 0, "width": 0, "height": 0}
    duration = float(info.get("format", {}).get("duration", 0) or 0)
    for s in info.get("streams", []):
        if s.get("codec_type") == "video": return {"duration": duration, "width": s.get("width", 0), "height": s.get("height", 0)}
    return {"duration": duration, "width": 0, "height": 0}

def detect_silences(path: Path, noise_db: int = -30, min_dur: float = 0.5) -> List[Dict]:
    logger.info("🔍 Analisi silenzi...")
    cmd = ["ffmpeg", "-i", str(path), "-af", f"silencedetect=n={noise_db}dB:d={min_dur}", "-f", "null", "-"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        silences = []
        for line in proc.stderr:
            if "silence_start" in line: silences.append({"start": float(re.search(r"silence_start:\s*([\d.]+)", line).group(1)), "end": None})
            elif "silence_end" in line and silences and silences[-1]["end"] is None:
                silences[-1]["end"] = float(re.search(r"silence_end:\s*([\d.]+)", line).group(1))
        proc.wait(timeout=120)
    except Exception as e:
        logger.warning(f"⚠️ Silenzio detection fallita: {e}")
        silences = [{"start": 0, "end": get_video_info(path)["duration"]}]

    if silences and silences[-1]["end"] is None: silences[-1]["end"] = get_video_info(path)["duration"]
    keep_segments, current_pos = [], 0.0
    for sil in silences:
        if sil["start"] > current_pos: keep_segments.append({"start": current_pos, "end": sil["start"]})
        current_pos = sil["end"]
    info = get_video_info(path)
    if current_pos < info["duration"]: keep_segments.append({"start": current_pos, "end": info["duration"]})
    logger.info(f"✅ {len(keep_segments)} segmenti trovati")
    return keep_segments if keep_segments else [{"start": 0, "end": info["duration"]}]

def transcribe_words(video_path: Path) -> Optional[List[Dict]]:
    if not has_audio_stream(video_path): return None
    model = get_whisper()
    if not model: return None
    logger.info("🎤 Trascrizione Whisper...")
    try:
        segments, _ = model.transcribe(str(video_path), language="it", word_timestamps=True)
        words = [{"start": w.start, "end": w.end, "text": w.word.strip()} for seg in segments for w in seg.words if w.word.strip()]
        logger.info(f"✅ {len(words)} parole trascritte")
        return words
    except Exception as e:
        logger.error(f"⚠️ Whisper fallita: {e}")
        return None

def adjust_words_for_cuts(words, keep_segments, max_sec):
    if not words or not keep_segments: return []
    adjusted, pos = [], 0.0
    for seg in keep_segments:
        if pos >= max_sec: break
        for w in words:
            if w["start"] >= seg["start"] and w["end"] <= seg["end"]:
                ns, ne = w["start"] - seg["start"] + pos, w["end"] - seg["start"] + pos
                if ns < max_sec: adjusted.append({"start": ns, "end": min(ne, max_sec), "text": w["text"]})
        pos += (seg["end"] - seg["start"])
    return adjusted

def chunk_words(words, max_words=4, max_gap=1.0):
    if not words: return []
    chunks, cur = [], [words[0]]
    for i in range(1, len(words)):
        if (words[i]["start"] - cur[-1]["end"] > max_gap) or len(cur) >= max_words:
            chunks.append(cur); cur = [words[i]]
        else: cur.append(words[i])
    if cur: chunks.append(cur)
    return chunks

def generate_ass_file(words, output_path, is_portrait):
    fs = 88 if is_portrait else 70; mv = 700 if is_portrait else 350
    ac, ic = "FFFFFF", "BBBBBB"
    header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\nStyle: Default,Arial,{fs},&H00{ic},&H00{ac},&H00000000,&HA0000000,-1,0,3,15,0,2,10,10,{mv}\n\n[Events]\nFormat: Start, End, Style, Text\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        if any(len(w["text"].split())>2 for w in words):
            for w in words: f.write(f"Dialogue: {format_ass_time(w['start'])},{format_ass_time(w['end'])},Default,{{\\c&H{ac}&}}{w['text'].replace('{','\\{').replace('}','\\}')}\n")
        else:
            for chunk in chunk_words(words):
                for i, aw in enumerate(chunk):
                    parts = [f"{{\\c&H{ac if j==i else ic}&}}{w['text'].replace('{','\\{').replace('}','\\}')}" for j,w in enumerate(chunk)]
                    f.write(f"Dialogue: {format_ass_time(aw['start'])},{format_ass_time(aw['end'])},Default,{' '.join(parts)}\n")

def format_ass_time(sec):
    h,m,s = int(sec//3600), int((sec%3600)//60), int(sec%60)
    return f"{h}:{m:02d}:{s:02d}.{int((sec-int(sec))*100):02d}"

def build_video(input_path: Path, audio_path: Optional[Path], output_path: Path, platform_cfg: Dict, keep_segments: List[Dict], words: Optional[List[Dict]], max_sec: int, auto_zoom: bool) -> None:
    w, h, fps, is_portrait = platform_cfg["w"], platform_cfg["h"], platform_cfg["fps"], platform_cfg["h"] > platform_cfg["w"]
    tmp_dir = Path(tempfile.mkdtemp(prefix="cap_"))
    try:
        segment_files, total_dur = [], 0.0
        logger.info(f"✂️ Taglio segmenti...")
        
        clean_meta = [
            "-map", "0:v:0", "-map", "0:a:0?",
            "-map", "-0:s", "-map", "-0:d", "-map", "-0:t",
            "-dn", "-sn", "-map_chapters", "-1",
            "-metadata", "title= ", "-metadata", "comment= ",
            "-metadata:s:v:0", "title= ", "-metadata:s:v:0", "handler_name= ",
            "-metadata:s:a:0", "title= ", "-metadata:s:a:0", "handler_name= "
        ]
        
        for i, seg in enumerate(keep_segments):
            if total_dur >= max_sec: break
            dur = min(seg["end"] - seg["start"], max_sec - total_dur)
            if dur <= 0.2: continue
            seg_path = tmp_dir / f"seg_{i}.mp4"
            cmd = ["ffmpeg", "-y", "-ss", str(seg["start"]), "-t", str(dur), "-i", str(input_path)] + clean_meta + \
                  ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", str(seg_path)]
            if not _run_ffmpeg_safe(cmd, 120): raise RuntimeError("Taglio fallito")
            if seg_path.exists(): segment_files.append(seg_path); total_dur += dur
            
        if not segment_files: raise RuntimeError("Nessun segmento valido")
        if len(segment_files) > 1:
            list_file = tmp_dir / "list.txt"; list_file.write_text("\n".join(f"file '{p}'" for p in segment_files))
            cut_video = tmp_dir / "cut.mp4"
            logger.info("🔗 Unione segmenti...")
            if not _run_ffmpeg_safe(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy", str(cut_video)], 60): raise RuntimeError("Unione fallita")
            cut_video = cut_video if cut_video.exists() else segment_files[0]
        else: cut_video = segment_files[0]
        logger.info(f"✅ Video base pronto")

        if is_portrait: vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}:(in_w-out_w)/2:0"
        else: vf = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black"
            
        if auto_zoom:
            logger.info("🔍 Auto-Zoom applicato")
            cycle = 10; speed = 2 * math.pi / (cycle * fps)
            vf += f",fps={fps},zoompan=z='1.125-0.125*cos(on*{speed:.6f})':d=0:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={w}x{h}"
        else:
            vf += f",fps={fps}"
            
        vf += ",setsar=1,format=yuv420p"

        if words:
            logger.info("📝 Sottotitoli applicati...")
            ass_path = tmp_dir / "subs.ass"
            generate_ass_file(words, ass_path, is_portrait)
            safe_path = str(ass_path).replace("\\", "/").replace(":", "\\:")
            vf += f",subtitles='{safe_path}'"

        final_video = tmp_dir / "final.mp4"
        cut_has_audio = has_audio_stream(cut_video)
        mp3_provided = audio_path is not None and audio_path.exists()
        
        logger.info("🎬 Rendering FFmpeg...")
        base_cmd = ["ffmpeg", "-y", "-i", str(cut_video)] + clean_meta
        if mp3_provided: base_cmd += ["-i", str(audio_path)]
        base_cmd += ["-map_metadata", "-1", "-tune", "zerolatency", "-movflags", "+faststart"]

        if mp3_provided and cut_has_audio:
            vf_filter, amix = f"[0:v]{vf}[vout]", "[0:a]volume=1.0[a1];[1:a]volume=0.25[a2];[a1][a2]amix=inputs=2:duration=first[aout]"
            cmd = base_cmd + ["-filter_complex", f"{vf_filter};{amix}", "-map", "[vout]", "-map", "[aout]",
                              "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(final_video)]
        elif mp3_provided:
            cmd = base_cmd + ["-vf", vf, "-map", "0:v:0", "-map", "1:a:0",
                              "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(final_video)]
        else:
            ac = "aac" if cut_has_audio else "an"
            cmd = base_cmd + ["-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-c:a", ac, "-b:a", "128k", str(final_video)]

        logger.info("⏳ Rendering in corso...")
        if not _run_ffmpeg_safe(cmd, 600): raise RuntimeError("Rendering fallito")
        logger.info(f"✅ Rendering completato: {output_path.name}")
        shutil.copy2(str(final_video), str(output_path))
    finally: 
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

def run_job(job_id, video_path, audio_path, video_filename, user_prompt, selected_platforms, auto_zoom):
    def update(p, m):
        with jobs_lock:
            if job_id in jobs: jobs[job_id]["progress"], jobs[job_id]["message"] = p, m
    try:
        with jobs_lock: jobs[job_id]["status"] = "running"
        logger.info(f"🚀 Job {job_id} PARTITO")
        update(5, "Analisi..."); get_video_info(video_path)
        update(10, "✂️ Silenzi..."); keep_segments = detect_silences(video_path)
        words = None
        if WHISPER_OK and has_audio_stream(video_path):
            update(20, "🎤 Trascrizione..."); raw = transcribe_words(video_path)
            if raw: update(25, "⏱️ Sync..."); words = adjust_words_for_cuts(raw, keep_segments, 180); update(35, "✅ Sottotitoli")
            else: update(35, "⚠️ Nessun audio")
        else: update(35, "⚠️ Video muto")
        
        results, n = {}, len(selected_platforms)
        config_cache = {}
        platform_to_key = {}
        
        for i, plat in enumerate(selected_platforms):
            cfg = PLATFORMS[plat]
            cfg_key = (cfg["w"], cfg["h"], cfg["max_sec"], cfg["fps"], auto_zoom)
            ps, pe = 35 + int(i * 60 / n), 35 + int((i + 1) * 60 / n)
            
            if cfg_key in config_cache:
                ok, err = config_cache[cfg_key]
                first_plat_for_key = next(p for p, k in platform_to_key.items() if k == cfg_key)
                target_path = OUTPUT_DIR / f"content_{plat}.mp4"
                src_path = OUTPUT_DIR / f"content_{first_plat_for_key}.mp4"
                if ok and plat != first_plat_for_key:
                    shutil.copy2(str(src_path), str(target_path))
                update(ps + 10, f"📦 Copia {cfg['label']}...")
            else:
                update(ps + 10, f"🔥 Rendering {cfg['label']}...")
                logger.info(f"📦 Avvio piattaforma: {cfg['label']}")
                out = OUTPUT_DIR / f"content_{plat}.mp4"
                pw = []
                if words:
                    for w in words:
                        if w["start"] < cfg["max_sec"]: pw.append({"start": w["start"], "end": min(w["end"], cfg["max_sec"]), "text": w["text"]})
                try: 
                    build_video(video_path, audio_path, out, cfg, keep_segments, pw, cfg["max_sec"], auto_zoom)
                    ok, err = True, None
                except Exception as e: 
                    ok, err = False, str(e); logger.error(f"❌ {plat}: {e}")
                config_cache[cfg_key] = (ok, err)
                platform_to_key[plat] = cfg_key
                
            results[plat] = {"label": cfg["label"], "icon": cfg["icon"], "video_ok": ok, "video_error": err, "video_name": f"content_{plat}.mp4" if ok else None, "resolution": f"{cfg['w']}x{cfg['h']}", "transcript": " ".join(w["text"] for w in words) if words else None}
            update(pe, f"✅ {cfg['label']} fatto")
            
        with jobs_lock: jobs[job_id].update({"status": "done", "results": results, "progress": 100, "message": "✅ Fatto!"})
        logger.info("🏁 JOB COMPLETATO")
    except Exception as e:
        logger.error(f"❌ Job fallito: {e}")
        with jobs_lock: jobs[job_id].update({"status": "error", "message": str(e)})
    finally:
        for p in [video_path, audio_path]:
            if p and str(p).startswith(tempfile.gettempdir()):
                try: p.unlink(missing_ok=True)
                except: pass

HTML = r"""<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8"><title>Content Agent Pro v8.5</title><style>*{margin:0;padding:0;box-sizing:border-box}:root{--bg:#0f172a;--card:#1e293b;--input:#334155;--primary:#6366f1;--text:#f8fafc;--muted:#94a3b8;--border:#475569;--green:#22c55e;--red:#ef4444}body{font-family:Segoe UI,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}.app{max-width:1100px;margin:0 auto;padding:2rem}h1{font-size:2.2rem;background:linear-gradient(135deg,#6366f1,#8b5cf6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.header{text-align:center;padding:2rem 0;border-bottom:1px solid var(--border);margin-bottom:2rem}.status-bar{display:flex;justify-content:center;gap:1rem;margin-top:1rem;flex-wrap:wrap}.status-item{display:flex;align-items:center;gap:0.5rem;padding:0.4rem 1rem;background:var(--card);border-radius:0.5rem;font-size:0.85rem}.dot{width:8px;height:8px;border-radius:50%;background:var(--green)}.dot.off{background:var(--red)}.card{background:var(--card);border-radius:1rem;padding:1.5rem;border:1px solid var(--border);margin-bottom:1.5rem}.card h2{font-size:1.1rem;margin-bottom:1rem;color:var(--muted)}.drop-zone{border:2px dashed var(--border);border-radius:0.75rem;padding:2rem;text-align:center;cursor:pointer;transition:border-color 0.2s}.drop-zone:hover,.drop-zone.drag{border-color:var(--primary)}.drop-zone.has-file{border-color:var(--green);border-style:solid}.drop-zone input{display:none}.drop-zone .icon{font-size:2rem;margin-bottom:0.4rem}.drop-zone p{color:var(--muted);font-size:0.85rem}.video-preview{width:100%;max-height:200px;border-radius:0.5rem;margin-top:1rem;display:none}.audio-preview{width:100%;margin-top:0.75rem;display:none}.platforms{display:grid;grid-template-columns:repeat(auto-fill,minmax(155px,1fr));gap:0.75rem;margin-top:0.5rem}.plat{display:flex;align-items:center;gap:0.5rem;padding:0.6rem 1rem;background:var(--input);border-radius:0.5rem;cursor:pointer;border:2px solid transparent;transition:border-color 0.2s;user-select:none}.plat input{width:16px;height:16px;cursor:pointer}.plat.selected{border-color:var(--primary)}.btn{width:100%;padding:0.9rem 2rem;font-size:1rem;font-weight:600;border:none;border-radius:0.75rem;cursor:pointer;background:linear-gradient(135deg,#6366f1,#8b5cf6);color:white}.btn:disabled{opacity:0.4;cursor:not-allowed}.progress-wrap{display:none;margin-top:1.25rem}.progress-wrap.active{display:block}.progress-bar-bg{background:var(--input);border-radius:1rem;height:14px;overflow:hidden}.progress-bar{height:100%;background:linear-gradient(135deg,#6366f1,#8b5cf6);width:0%;transition:width 0.5s;border-radius:1rem}.progress-msg{font-size:0.85rem;color:var(--muted);margin-top:0.5rem;min-height:1.2em}.results{display:none;margin-top:1.5rem}.results.active{display:block}.result-card{background:var(--input);border-radius:0.75rem;padding:1.25rem;margin-bottom:1rem}.result-header{display:flex;align-items:center;gap:0.5rem;font-size:1rem;font-weight:600;margin-bottom:1rem;flex-wrap:wrap}.badge{font-size:0.75rem;padding:0.2rem 0.6rem;border-radius:1rem;background:var(--green);color:#000;font-weight:600}.badge.err{background:var(--red);color:#fff}.dl-btn{display:inline-flex;align-items:center;gap:0.4rem;font-size:0.9rem;padding:0.5rem 1.25rem;background:var(--green);color:#000;border:none;border-radius:0.5rem;cursor:pointer;text-decoration:none;font-weight:700;margin-top:0.25rem}.transcript-box{background:var(--card);border-radius:0.5rem;padding:0.75rem;font-size:0.85rem;color:var(--muted);max-height:130px;overflow-y:auto;line-height:1.6;margin-top:0.5rem}.two-col{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}@media(max-width:750px){.two-col{grid-template-columns:1fr}}</style></head><body><div class="app"><header class="header"><h1>🎬 Content Agent Pro v8.5</h1><p style="color:var(--muted);margin-top:0.3rem">Optimized Render • Clean</p><div class="status-bar"><div class="status-item"><div class="dot" id="dotFfmpeg"></div><span id="sFfmpeg">ffmpeg...</span></div><div class="status-item"><div class="dot" id="dotWhisper"></div><span id="sWhisper">Whisper...</span></div></div></header><div class="two-col"><div><div class="card"><h2>🎥 1. Carica il video</h2><div class="drop-zone" id="dropVideo"><input type="file" id="fileVideo" accept="video/*,.mp4,.mov,.avi,.mkv,.webm"><div class="icon">🎬</div><p id="dropVideoTxt">Clicca o trascina il video</p></div><video id="previewVideo" class="video-preview" controls></video></div><div class="card"><h2>🎵 2. Musica (opzionale)</h2><div class="drop-zone" id="dropAudio"><input type="file" id="fileAudio" accept="audio/*,.mp3,.wav"><div class="icon">🎵</div><p id="dropAudioTxt">Clicca o trascina un file audio</p></div><audio id="previewAudio" class="audio-preview" controls></audio></div></div><div><div class="card"><h2>📱 3. Piattaforme & Effetti</h2><div class="platforms"><label class="plat selected"><input type="checkbox" name="p" value="tiktok,instagram,youtube" checked onchange="togglePlat(this)"> 📱 YT, IG, TikTok (1080x1920)</label><label class="plat"><input type="checkbox" name="p" value="twitter" onchange="togglePlat(this)"> 🐦 Twitter/X (1080x1080)</label><label class="plat"><input type="checkbox" name="p" value="linkedin" onchange="togglePlat(this)"> 💼 LinkedIn (1920x1080)</label><label class="plat"><input type="checkbox" name="p" value="youtube_long" onchange="togglePlat(this)"> 🎥 YT Video 3m (1920x1080)</label><label class="plat"><input type="checkbox" name="p" value="podcast" onchange="togglePlat(this)"> 🎙️ Podcast 3m (1080x1920)</label><label class="plat"><input type="checkbox" id="autoZoom" onchange="togglePlat(this)"> 🔍 Auto-Zoom</label></div></div><div class="card"><h2>💬 Sottotitoli</h2><p style="color:var(--muted);font-size:0.9rem">Generazione automatica se presente audio.</p></div></div></div><button class="btn" id="genBtn" onclick="startJob()" disabled>🚀 Avvia</button><div class="progress-wrap" id="progressWrap"><div class="progress-bar-bg"><div class="progress-bar" id="progressBar"></div></div><div class="progress-msg" id="progressMsg">Avvio...</div></div><div class="results" id="results"><div class="card"><h2>📋 Trascrizione</h2><div class="transcript-box" id="transcriptBox">—</div></div><div id="resultCards"></div></div></div><script>
let selVideo=null,selAudio=null,pollInterval=null,currentJobId=null;
document.getElementById("dropVideo").addEventListener("click",()=>document.getElementById("fileVideo").click());
document.getElementById("dropVideo").addEventListener("dragover",(e)=>{e.preventDefault();e.currentTarget.classList.add("drag");});
document.getElementById("dropVideo").addEventListener("dragleave",(e)=>{e.currentTarget.classList.remove("drag");});
document.getElementById("dropVideo").addEventListener("drop",(e)=>{e.preventDefault();e.currentTarget.classList.remove("drag");if(e.dataTransfer.files[0])onVideo(e.dataTransfer.files[0]);});
document.getElementById("fileVideo").addEventListener("change",(e)=>{if(e.target.files[0])onVideo(e.target.files[0]);});
document.getElementById("dropAudio").addEventListener("click",()=>document.getElementById("fileAudio").click());
document.getElementById("dropAudio").addEventListener("dragover",(e)=>{e.preventDefault();e.currentTarget.classList.add("drag");});
document.getElementById("dropAudio").addEventListener("dragleave",(e)=>{e.currentTarget.classList.remove("drag");});
document.getElementById("dropAudio").addEventListener("drop",(e)=>{e.preventDefault();e.currentTarget.classList.remove("drag");if(e.dataTransfer.files[0])onAudio(e.dataTransfer.files[0]);});
document.getElementById("fileAudio").addEventListener("change",(e)=>{if(e.target.files[0])onAudio(e.target.files[0]);});
function onVideo(f){if(!f)return;selVideo=f;document.getElementById("previewVideo").src=URL.createObjectURL(f);document.getElementById("previewVideo").style.display="block";document.getElementById("dropVideoTxt").innerHTML="✅ <strong>"+f.name+"</strong><br><small>"+(f.size/1024/1024).toFixed(1)+" MB</small>";document.getElementById("dropVideo").classList.add("has-file");checkReady();}
function onAudio(f){if(!f)return;selAudio=f;document.getElementById("previewAudio").src=URL.createObjectURL(f);document.getElementById("previewAudio").style.display="block";document.getElementById("dropAudioTxt").innerHTML="✅ <strong>"+f.name+"</strong>";document.getElementById("dropAudio").classList.add("has-file");}
function checkReady(){document.getElementById("genBtn").disabled=!selVideo;}
function togglePlat(cb){cb.closest(".plat").classList.toggle("selected",cb.checked);}
async function checkStatus(){try{const j=await(await fetch("/api/status")).json();setDot("dotFfmpeg","sFfmpeg",j.ffmpeg,"ffmpeg: "+(j.ffmpeg?"✓":"Mancante"));setDot("dotWhisper","sWhisper",j.whisper,"Whisper: "+(j.whisper?"✓":"No"));}catch(e){}}
function setDot(d,l,ok,t){document.getElementById(d).classList.toggle("off",!ok);document.getElementById(l).textContent=t;}
async function startJob(){
    if(pollInterval) clearInterval(pollInterval); pollInterval=null; currentJobId=null;
    const rawPlatforms = Array.from(document.querySelectorAll("input[name=p]:checked")).map(c=>c.value);
    const platforms = []; rawPlatforms.forEach(p=>p.split(',').forEach(v=>platforms.push(v.trim())));
    if(!selVideo||!platforms.length)return;
    const autoZoom=document.getElementById("autoZoom").checked;
    document.getElementById("genBtn").disabled=true;
    document.getElementById("progressWrap").classList.add("active");
    document.getElementById("results").classList.remove("active");
    document.getElementById("resultCards").innerHTML="";
    document.getElementById("progressBar").style.width="0%";
    const formData=new FormData();
    formData.append("video",selVideo); if(selAudio)formData.append("audio",selAudio);
    formData.append("auto_zoom",autoZoom); platforms.forEach(p=>formData.append("platforms",p));
    try{
        const res=await fetch("/api/job/start",{method:"POST",body:formData});
        if(!res.ok){const e=await res.json().catch(()=>({}));throw new Error(e.detail||"Errore "+res.status);}
        const j=await res.json(); currentJobId=j.job_id;
        pollInterval=setInterval(()=>pollJob(currentJobId),1500);
    }catch(e){alert("❌ "+e.message);resetJob();}
}
async function pollJob(id){
    if(!currentJobId||id!==currentJobId) return;
    try{
        const res=await fetch("/api/job/status?job_id="+id);
        if(!res.ok) return;
        const j=await res.json();
        document.getElementById("progressBar").style.width=j.progress+"%";
        document.getElementById("progressMsg").textContent=j.message;
        if(j.status==="done"){clearInterval(pollInterval);pollInterval=null;showResults(j.results);resetJob();}
        else if(j.status==="error"){clearInterval(pollInterval);pollInterval=null;document.getElementById("progressMsg").textContent="❌ "+j.message;resetJob();}
    }catch(e){}
}
function resetJob(){if(pollInterval)clearInterval(pollInterval);pollInterval=null;document.getElementById("genBtn").disabled=!selVideo;}
function showResults(r){document.getElementById("results").classList.add("active");const f=Object.values(r)[0];document.getElementById("transcriptBox").textContent=f?.transcript||"Nessun audio.";let h="";for(const[k,v]of Object.entries(r)){h+=`<div class="result-card"><div class="result-header"><span>${v.icon} ${v.label}</span><span class="badge ${v.video_ok?"":"err"}">${v.video_ok?"✅":"❌"}</span><span style="font-size:0.75rem;color:var(--muted);margin-left:auto">${v.resolution}</span></div>${v.video_ok?`<a class="dl-btn" href="/api/download?file=${encodeURIComponent(v.video_name)}" download="${v.video_name}">⬇️ Scarica</a>`:`<p style="color:var(--red)">${v.video_error}</p>`}</div>`;}document.getElementById("resultCards").innerHTML=h;}
checkStatus();setInterval(checkStatus,15000);
</script></body></html>"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("="*40); logger.info("  🎬  Content Agent Pro v8.5"); logger.info("="*40)
    if not check_ffmpeg(): logger.error("❌ ffmpeg mancante!")
    else: logger.info("✅ ffmpeg OK")
    logger.info("✅ Whisper OK" if WHISPER_OK else "⚠️ Whisper no")
    logger.info(f"📁 Output: {OUTPUT_DIR}"); yield; logger.info("👋 Shutdown")

app = FastAPI(title="Content Agent Pro", version="8.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def root(): return HTMLResponse(content=HTML)
@app.get("/api/status")
async def get_status(): return {"ffmpeg": check_ffmpeg(), "whisper": WHISPER_OK}

@app.post("/api/job/start")
async def start_job(background_tasks: BackgroundTasks, video: UploadFile = File(...), audio: Optional[UploadFile] = File(None), auto_zoom: str = Form("false"), platforms: List[str] = Form(...)):
    if not platforms: raise HTTPException(400, "Seleziona piattaforma")
    tmp = Path(tempfile.mkdtemp(prefix="cap_"))
    vp = tmp / f"input{Path(video.filename).suffix or '.mp4'}"
    with open(vp, "wb") as f:
        while c := await video.read(8192): f.write(c)
    ap = None
    if audio:
        ap = tmp / f"music{Path(audio.filename).suffix or '.mp3'}"
        with open(ap, "wb") as f:
            while c := await audio.read(8192): f.write(c)
    jid = f"job_{int(time.time()*1000)}"
    with jobs_lock: jobs[jid] = {"status": "queued", "progress": 0, "message": "In attesa...", "results": {}}
    background_tasks.add_task(run_job, jid, vp, ap, video.filename, "", platforms, auto_zoom.lower()=="true")
    return {"job_id": jid}

@app.get("/api/job/status")
async def get_job_status(job_id: str):
    with jobs_lock: j = jobs.get(job_id)
    if not j: raise HTTPException(404, "Job non trovato")
    return {"job_id": job_id, "status": j["status"], "progress": j["progress"], "message": j["message"], "results": j.get("results")}

@app.get("/api/download")
async def download_file(file: str):
    fp = (OUTPUT_DIR / file).resolve()
    try: fp.relative_to(OUTPUT_DIR.resolve())
    except ValueError: raise HTTPException(403, "Negato")
    if not fp.exists(): raise HTTPException(404, "Non trovato")
    return FileResponse(path=str(fp), filename=file, media_type="video/mp4")

def main():
    import argparse
    default_host = os.environ.get("HOST", "0.0.0.0")
    default_port = int(os.environ.get("PORT") or 8000)

    p = argparse.ArgumentParser()
    p.add_argument("--host", default=default_host)
    p.add_argument("--port", type=int, default=default_port)
    args = p.parse_args()

    if args.host in ("127.0.0.1", "localhost"):
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()
    logger.info(f"🚀 http://{args.host}:{args.port}")
    uvicorn.run("content_agent_pro:app", host=args.host, port=args.port, log_level="info")

if __name__ == "__main__": main()
