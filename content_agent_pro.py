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
            WHISPER_MODEL_INSTANCE = WhisperModel("small", device="cpu", compute_type="int8")
            logger.info("✅ Whisper model caricato (small)")
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

def clean_hex_color(value: str, default: str) -> str:
    if not isinstance(value, str):
        return default
    c = value.strip().lstrip("#")
    if len(c) == 3 and all(ch in "0123456789ABCDEFabcdef" for ch in c):
        c = "".join(ch * 2 for ch in c)
    if len(c) == 6 and all(ch in "0123456789ABCDEFabcdef" for ch in c):
        return c.upper()
    return default

def validate_subtitle_font(font: str) -> str:
    allowed = {"Arial", "Impact", "Montserrat", "Oswald", "Bebas Neue"}
    return font if font in allowed else "Arial"

def validate_bg_style(style: str) -> str:
    return style if style in {"none", "black", "white", "glow"} else "none"

def get_stream_codecs(path: Path) -> Dict[str, Optional[str]]:
    try:
        r = subprocess.run(["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", str(path)], capture_output=True, text=True, timeout=30)
        streams = json.loads(r.stdout).get("streams", [])
        result = {"video": None, "audio": None}
        for s in streams:
            if s.get("codec_type") == "video" and result["video"] is None:
                result["video"] = s.get("codec_name")
            elif s.get("codec_type") == "audio" and result["audio"] is None:
                result["audio"] = s.get("codec_name")
        return result
    except:
        return {"video": None, "audio": None}

def detect_silences(path: Path, noise_db: int = -30, min_dur: float = 0.5, duration: Optional[float] = None) -> List[Dict]:
    logger.info("🔍 Analisi silenzi...")
    if duration is None:
        duration = get_video_info(path)["duration"]
    cmd = ["ffmpeg", "-i", str(path), "-vn", "-sn", "-nostats", "-hide_banner", "-af", f"silencedetect=n={noise_db}dB:d={min_dur}", "-f", "null", "-"]
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
        silences = [{"start": 0, "end": duration}]

    if silences and silences[-1]["end"] is None: silences[-1]["end"] = duration
    keep_segments, current_pos = [], 0.0
    for sil in silences:
        if sil["start"] > current_pos: keep_segments.append({"start": current_pos, "end": sil["start"]})
        current_pos = sil["end"]
    if current_pos < duration: keep_segments.append({"start": current_pos, "end": duration})
    logger.info(f"✅ {len(keep_segments)} segmenti trovati")
    return keep_segments if keep_segments else [{"start": 0, "end": duration}]

def transcribe_words(video_path: Path) -> Optional[List[Dict]]:
    if not has_audio_stream(video_path): return None
    model = get_whisper()
    if not model: return None
    logger.info("🎤 Trascrizione Whisper...")
    try:
        segments, _ = model.transcribe(str(video_path), language="it", word_timestamps=True)
        words = [{"start": w.start, "end": w.end, "text": w.word.strip()} for seg in segments for w in seg.words if w.word.strip()]
        logger.info(f"✅ {len(words)} parole trascritte")
        if words:
            logger.info("🔎 Ultime 3 parole transcritte: %s", [(w['text'], w['start'], w['end']) for w in words[-3:]])
        return words
    except Exception as e:
        logger.error(f"⚠️ Whisper fallita: {e}")
        return None

def adjust_words_for_cuts(words, keep_segments, max_sec):
    if not words or not keep_segments: return []

    # posizione di inizio di ogni segmento nella timeline compressa
    seg_pos, pos = [], 0.0
    for seg in keep_segments:
        seg_pos.append(pos)
        pos += seg["end"] - seg["start"]

    try:
        logger.info("🔎 keep_segments: %s", [(s['start'], s['end']) for s in keep_segments])
    except Exception:
        pass

    MIN_DUR = 0.3  # durata minima visibile per le parole recuperate (anti-lampeggio)

    adjusted = []
    assigned = [False] * len(words)

    # PRIMA PASSATA: il punto medio cade dentro un segmento -> remap lineare
    for si, seg in enumerate(keep_segments):
        base, s0, s1 = seg_pos[si], seg["start"], seg["end"]
        for wi, w in enumerate(words):
            if assigned[wi]:
                continue
            mid = (w["start"] + w["end"]) / 2.0
            if s0 <= mid <= s1:
                ns = max(w["start"], s0) - s0 + base
                ne = min(w["end"], s1) - s0 + base
                if ns < max_sec:
                    adjusted.append({"start": ns, "end": min(ne, max_sec), "text": w["text"]})
                    assigned[wi] = True

    # SECONDA PASSATA: parole rimaste nei buchi -> start ancorato al bordo
    # del segmento più vicino (inizio se il buco è prima, fine se è dopo).
    for wi, w in enumerate(words):
        if assigned[wi]:
            continue
        mid = (w["start"] + w["end"]) / 2.0
        dur = max(w["end"] - w["start"], 0.0)

        best_idx, best_dist, before = None, None, True
        for si, s in enumerate(keep_segments):
            if mid < s["start"]:
                dist, side_before = s["start"] - mid, True
            elif mid > s["end"]:
                dist, side_before = mid - s["end"], False
            else:
                dist, side_before = 0.0, True
            if best_dist is None or dist < best_dist:
                best_idx, best_dist, before = si, dist, side_before

        if best_idx is None:
            continue

        base = seg_pos[best_idx]
        seg_dur = keep_segments[best_idx]["end"] - keep_segments[best_idx]["start"]
        ns = base if before else base + seg_dur   # bordo: inizio o fine del segmento
        ne = ns + max(dur, MIN_DUR)
        if ns < max_sec:
            adjusted.append({"start": ns, "end": min(ne, max_sec), "text": w["text"]})
            assigned[wi] = True

    # ordine cronologico per chunking / karaoke
    adjusted.sort(key=lambda x: (x["start"], x["end"]))

    # VISIBILITÀ MINIMA SENZA DRIFT: lo start di ogni parola resta al suo tempo
    # reale (niente spinta in avanti), così l'evidenziazione non accumula ritardo.
    # La durata minima si ottiene estendendo la FINE: può creare un piccolo overlap
    # con la parola seguente (preferibile al lag dell'evidenziazione).
    result = []
    for it in adjusted:
        s = it["start"]
        e = min(max(it["end"], s + MIN_DUR), max_sec)   # >= MIN_DUR e <= max_sec
        result.append({"start": s, "end": e, "text": it["text"]})

    discarded = [words[i]["text"] for i in range(len(words)) if not assigned[i]]
    logger.info("🔎 adjust_words_for_cuts: input=%d output=%d scartate=%d",
                len(words), len(result), len(discarded))
    if discarded:
        logger.info("🔎 Parole scartate (oltre max_sec): %s", discarded)

    return result

def chunk_words(words, max_words=4, max_gap=1.0):
    if not words: return []
    chunks, cur = [], [words[0]]
    for i in range(1, len(words)):
        if (words[i]["start"] - cur[-1]["end"] > max_gap) or len(cur) >= max_words:
            chunks.append(cur); cur = [words[i]]
        else: cur.append(words[i])
    if cur: chunks.append(cur)
    return chunks

def generate_ass_file(words, output_path, is_portrait, font: str = "Arial", highlight_color: str = "#FFD700", text_color: str = "#FFFFFF", bg_style: str = "none"):
    fs = 88 if is_portrait else 70; mv = 700 if is_portrait else 350
    font = validate_subtitle_font(font)
    text_color = clean_hex_color(text_color, "FFFFFF")
    highlight_color = clean_hex_color(highlight_color, "FFD700")
    bg_style = validate_bg_style(bg_style)
    back_color = "FF000000"
    outline_color = "000000"
    border_style = 1
    outline = 2
    shadow = 0

    if bg_style == "black":
        back_color = "80000000"
        border_style = 3
        outline = 2
        shadow = 0
    elif bg_style == "white":
        back_color = "80FFFFFF"
        border_style = 3
        outline = 2
        shadow = 0
    elif bg_style == "glow":
        back_color = "FF000000"
        outline_color = highlight_color
        border_style = 1
        outline = 8
        shadow = 10

    header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\nStyle: Default,{font},{fs},&H00{text_color},&H00{highlight_color},&H{outline_color},&H{back_color},-1,0,{border_style},{outline},{shadow},2,10,10,{mv}\n\n[Events]\nFormat: Start, End, Style, Text\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        if any(len(w["text"].split())>2 for w in words):
            for w in words:
                text = w['text'].replace('{', '\\{').replace('}', '\\}')
                f.write(f"Dialogue: {format_ass_time(w['start'])},{format_ass_time(w['end'])},Default,{{\\c&H{highlight_color}&}}{text}\n")
        else:
            for chunk in chunk_words(words):
                for i, aw in enumerate(chunk):
                    parts = []
                    for j, w in enumerate(chunk):
                        text = w['text'].replace('{', '\\{').replace('}', '\\}')
                        if j == i:
                            parts.append(f"{{\\c&H{highlight_color}&}}{text}{{\\c&H{text_color}&}}")
                        else:
                            parts.append(text)
                    f.write(f"Dialogue: {format_ass_time(aw['start'])},{format_ass_time(aw['end'])},Default,{' '.join(parts)}\n")

def format_ass_time(sec):
    h,m,s = int(sec//3600), int((sec%3600)//60), int(sec%60)
    return f"{h}:{m:02d}:{s:02d}.{int((sec-int(sec))*100):02d}"

def build_video(input_path: Path, audio_path: Optional[Path], output_path: Path, platform_cfg: Dict, keep_segments: List[Dict], words: Optional[List[Dict]], max_sec: int, auto_zoom: bool, subtitle_font: str = "Arial", subtitle_highlight_color: str = "#FFD700", subtitle_text_color: str = "#FFFFFF", subtitle_bg_style: str = "none") -> None:
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
        source_codecs = get_stream_codecs(input_path)
        
        for i, seg in enumerate(keep_segments):
            if total_dur >= max_sec: break
            dur = min(seg["end"] - seg["start"], max_sec - total_dur)
            if dur <= 0.2: continue
            seg_path = tmp_dir / f"seg_{i}.mp4"
            cmd = ["ffmpeg", "-y", "-ss", str(seg["start"]), "-t", str(dur), "-i", str(input_path)] + clean_meta + ["-c:v", "libx264", "-preset", "superfast", "-crf", "23", "-c:a", "aac", str(seg_path)]
            if not _run_ffmpeg_safe(cmd, 300): raise RuntimeError("Taglio fallito")
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
            generate_ass_file(words, ass_path, is_portrait, font=subtitle_font, highlight_color=subtitle_highlight_color, text_color=subtitle_text_color, bg_style=subtitle_bg_style)
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
                              "-c:v", "libx264", "-preset", "superfast", "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(final_video)]
        elif mp3_provided:
            audio_codec = get_stream_codecs(audio_path).get("audio") if audio_path else None
            audio_args = ["-c:a", "copy"] if audio_codec == "aac" else ["-c:a", "aac", "-b:a", "128k"]
            cmd = base_cmd + ["-vf", vf, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "superfast", "-crf", "23"] + audio_args + [str(final_video)]
        else:
            if cut_has_audio and get_stream_codecs(cut_video).get("audio") == "aac":
                ac_args = ["-c:a", "copy"]
            else:
                ac_args = ["-c:a", "aac", "-b:a", "128k"] if cut_has_audio else ["-c:a", "an"]
            cmd = base_cmd + ["-vf", vf, "-c:v", "libx264", "-preset", "superfast", "-crf", "23"] + ac_args + [str(final_video)]

        logger.info("⏳ Rendering in corso...")
        if not _run_ffmpeg_safe(cmd, 600): raise RuntimeError("Rendering fallito")
        logger.info(f"✅ Rendering completato: {output_path.name}")
        shutil.copy2(str(final_video), str(output_path))
    finally: 
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

def run_job(job_id, video_path, audio_path, video_filename, user_prompt, selected_platforms, auto_zoom, cut_silences: bool, subtitle_font: str = "Arial", subtitle_highlight_color: str = "#FFD700", subtitle_text_color: str = "#FFFFFF", subtitle_bg_style: str = "none"):
    def update(p, m):
        with jobs_lock:
            if job_id in jobs: jobs[job_id]["progress"], jobs[job_id]["message"] = p, m
    try:
        with jobs_lock: jobs[job_id]["status"] = "running"
        logger.info(f"🚀 Job {job_id} PARTITO")
        update(5, "Analisi..."); video_info = get_video_info(video_path)
        if cut_silences:
            update(10, "✂️ Silenzi..."); keep_segments = detect_silences(video_path, duration=video_info["duration"])
        else:
            update(10, "⏹️ Taglio silenzi disattivato"); keep_segments = [{"start": 0.0, "end": video_info["duration"]}]
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
                    build_video(video_path, audio_path, out, cfg, keep_segments, pw, cfg["max_sec"], auto_zoom, subtitle_font, subtitle_highlight_color, subtitle_text_color, subtitle_bg_style)
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

HTML = r"""<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Clipflow</title><style>*{margin:0;padding:0;box-sizing:border-box}:root{--primary:#FF006E;--secondary:#FB5607;--accent:#FFBE0B;--success:#8BC34A;--error:#FF4757;--dark:#121212;--darker:#090909;--light:#F5F5F5;--text:#FFFFFF;--muted:#A8A8A8;--border:#2D2D2D;--surface:rgba(255,255,255,0.06)}html,body{min-height:100%;font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:linear-gradient(135deg,var(--darker) 0%,#0e0a13 100%);color:var(--text);overflow-x:hidden}body::before{content:'';position:fixed;inset:0;background:radial-gradient(circle at 18% 30%,rgba(255,0,110,0.18) 0%,transparent 28%),radial-gradient(circle at 85% 10%,rgba(251,86,7,0.16) 0%,transparent 25%);pointer-events:none;z-index:-1}.container{max-width:960px;margin:0 auto;padding:1rem}.header{margin-bottom:1.5rem;padding:1.8rem 1.2rem;border-radius:2rem;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);backdrop-filter:blur(16px)}.logo{display:inline-flex;align-items:center;gap:0.85rem;font-size:1.15rem;font-weight:800;color:var(--text)}.logo span{font-size:1.8rem;display:inline-flex;align-items:center;justify-content:center}.header h1{font-size:clamp(2rem,4vw,2.8rem);margin:0.75rem 0 0.35rem;letter-spacing:-0.04em}.header p{font-size:0.95rem;color:var(--muted);max-width:640px;line-height:1.6}.status-bar{display:flex;flex-wrap:wrap;gap:0.85rem;margin-top:1.4rem}.status-item{display:flex;align-items:center;gap:0.65rem;padding:0.75rem 1rem;border-radius:999px;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);font-size:0.9rem}.dot{width:10px;height:10px;border-radius:50%;background:var(--success);box-shadow:0 0 8px rgba(139,195,74,0.4)}.dot.off{background:var(--error);box-shadow:0 0 8px rgba(255,71,87,0.5)}.card{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:1.75rem;padding:1.5rem;backdrop-filter:blur(18px);margin-bottom:1.5rem;transition:transform .25s ease,box-shadow .25s ease}.card:hover{transform:translateY(-2px);box-shadow:0 20px 40px rgba(0,0,0,0.18)}.card h2{font-size:1rem;font-weight:700;color:var(--accent);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem}.drop-zone{position:relative;border:2px dashed rgba(255,255,255,0.15);border-radius:1.5rem;padding:1.8rem 1.4rem;text-align:center;cursor:pointer;transition:all .25s ease;background:rgba(255,255,255,0.04)}.drop-zone:hover{border-color:rgba(255,0,110,0.5);background:rgba(255,0,110,0.08)}.drop-zone.drag{border-color:var(--secondary);background:rgba(251,86,7,0.08)}.drop-zone.has-file{border-color:var(--success);background:rgba(139,195,74,0.12)}.drop-zone .icon{font-size:2.4rem;margin-bottom:0.8rem;display:inline-flex}.drop-zone p{font-size:0.95rem;color:var(--muted);line-height:1.6}.drop-zone small{display:block;margin-top:0.45rem;color:var(--muted)}.file-info{margin-top:1rem;color:var(--accent);font-weight:700;letter-spacing:0.02em;line-height:1.4}.video-thumb{margin-top:1rem;border-radius:1.5rem;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.1);padding:1rem;display:grid;grid-template-columns:auto 1fr;gap:1rem;align-items:center;cursor:pointer;transition:all .25s ease}.video-thumb:hover{background:rgba(255,255,255,0.12)}.video-thumb .thumb-icon{width:56px;height:56px;border-radius:1.1rem;background:linear-gradient(135deg,var(--primary),var(--secondary));display:grid;place-items:center;font-size:1.8rem;color:#0f0f0f}.video-thumb .thumb-details{display:flex;flex-direction:column;gap:0.25rem;text-align:left}.video-thumb .thumb-title{font-weight:700;font-size:0.95rem}.video-thumb .thumb-label{color:var(--muted);font-size:0.9rem;line-height:1.4}.video-thumb .thumb-action{font-size:0.85rem;color:var(--accent);font-weight:700}.video-preview{width:100%;border-radius:1.5rem;margin-top:1rem;display:none;max-height:260px;object-fit:cover;background:#000}.platforms{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:0.85rem;margin-top:0.75rem}.plat{display:flex;align-items:center;gap:0.65rem;padding:0.8rem 1rem;border-radius:1rem;border:2px solid rgba(255,255,255,0.12);background:rgba(255,255,255,0.05);color:var(--text);cursor:pointer;transition:all .25s ease}.plat:hover{border-color:rgba(255,255,255,0.2)}.plat input{cursor:pointer}.plat.selected{background:linear-gradient(135deg,var(--primary),var(--secondary));border-color:transparent;color:#090909;font-weight:700;box-shadow:0 18px 30px rgba(255,0,110,0.18)}.options{display:flex;flex-wrap:wrap;gap:1rem;align-items:center;margin-top:1rem}.checkbox-group{display:flex;align-items:center;gap:0.6rem;cursor:pointer;user-select:none;color:var(--text)}.checkbox-group input{cursor:pointer}.button{width:100%;padding:1rem 1.1rem;border:none;border-radius:1rem;font-size:1rem;font-weight:700;cursor:pointer;transition:all .25s ease;text-transform:uppercase;letter-spacing:0.05em;background:linear-gradient(135deg,var(--primary),var(--secondary));color:#090909;box-shadow:0 18px 30px rgba(255,0,110,0.2)}.button:hover{transform:translateY(-2px)}.button:disabled{opacity:.5;cursor:not-allowed;transform:none;box-shadow:none}.progress-wrap{display:none;margin-top:1.75rem}.progress-wrap.active{display:block}.stepper{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:0.8rem}.step{padding:1rem;border-radius:1.25rem;border:1px solid rgba(255,255,255,0.08);background:rgba(255,255,255,0.04);color:var(--text);display:flex;align-items:flex-start;gap:0.85rem;transition:all .25s ease}.step.active{background:linear-gradient(135deg,var(--primary),var(--secondary));color:#090909;border-color:transparent;box-shadow:0 20px 40px rgba(255,0,110,0.18)}.step-icon{width:40px;height:40px;border-radius:14px;background:rgba(255,255,255,0.12);display:grid;place-items:center;font-size:1.1rem;flex-shrink:0}.step.active .step-icon{background:#fff}.step-info{display:flex;flex-direction:column;gap:0.2rem}.step-info h3{font-size:0.92rem;margin:0;font-weight:700}.step-info p{margin:0;color:inherit;opacity:.85;font-size:0.82rem;line-height:1.4}.progress-msg{margin-top:1rem;text-align:center;color:var(--muted);font-size:0.95rem;min-height:1.5rem}.results{display:none;margin-top:1.75rem}.results.active{display:block}.transcript-section{background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:1.5rem;padding:1rem;backdrop-filter:blur(12px);margin-bottom:1rem}.transcript-section h3{font-size:.9rem;text-transform:uppercase;letter-spacing:.08em;color:var(--accent);margin-bottom:.75rem}.transcript-box{color:var(--muted);font-size:.95rem;line-height:1.7;max-height:170px;overflow-y:auto}.result-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:1rem}.result-card{display:flex;gap:1rem;align-items:center;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);border-radius:1.25rem;padding:1rem;backdrop-filter:blur(12px);transition:all .25s ease}.result-card:hover{transform:translateY(-2px)}.result-thumb{width:56px;height:56px;border-radius:1.25rem;display:grid;place-items:center;font-size:1.35rem;font-weight:800;color:#090909;background:linear-gradient(135deg,var(--primary),var(--secondary));box-shadow:0 18px 30px rgba(255,0,110,0.2)}.result-content{flex:1;display:flex;flex-direction:column;gap:0.5rem}.result-header{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}.result-header span{font-weight:700;font-size:0.95rem}.badge{padding:.35rem .75rem;border-radius:.75rem;font-size:.78rem;font-weight:700;background:rgba(139,195,74,0.15);color:var(--success)}.badge.err{background:rgba(255,71,87,0.12);color:var(--error)}.dl-btn{display:inline-flex;align-items:center;justify-content:center;padding:.8rem 1rem;border-radius:.95rem;background:linear-gradient(135deg,var(--success),#8BC34A);color:#090909;font-weight:700;text-decoration:none;font-size:.88rem;transition:all .25s ease}.dl-btn:hover{transform:translateY(-2px);box-shadow:0 12px 24px rgba(139,195,74,0.3)}.error-text{color:var(--error);font-size:.9rem}.hidden{display:none}.@media(max-width:720px){.stepper{grid-template-columns:repeat(2,minmax(0,1fr))}.header{padding:1.4rem 1rem}.card{padding:1.25rem}.video-thumb{grid-template-columns:1fr;}.video-thumb .thumb-icon{margin:0 auto}.result-card{flex-direction:column;align-items:stretch}}@media(max-width:520px){.container{padding:.85rem}} </style></head><body><div class="container"><header class="header"><div class="logo"><span>✂️</span>Clipflow</div><h1>Creazione video smart per social</h1><p>Carica, scegli piattaforme e genera clip ottimizzate con un’interfaccia moderna e mobile-first.</p><div class="status-bar"><div class="status-item"><span class="dot" id="dotFfmpeg"></span><span id="sFfmpeg">Verifica ffmpeg...</span></div><div class="status-item"><span class="dot" id="dotWhisper"></span><span id="sWhisper">Verifica Whisper...</span></div></div></header><main><section class="card"><h2>📹 Video</h2><div class="drop-zone" id="dropVideo"><div class="icon">📤</div><p>Trascina il video qui oppure tocca per selezionarlo</p><small>MP4, MOV, WEBM</small><div class="file-info" id="dropVideoTxt"></div><input type="file" id="fileVideo" accept="video/*"><div class="video-thumb" id="videoThumb"><div class="thumb-icon">🎬</div><div class="thumb-details"><div class="thumb-title">Anteprima video</div><div class="thumb-label" id="thumbText">Seleziona un video per visualizzare</div><div class="thumb-action">Tocca per espandere</div></div></div><video id="previewVideo" class="video-preview" controls playsinline></video></div></section><section class="card"><h2>🎵 Audio (Opzionale)</h2><div class="drop-zone" id="dropAudio"><div class="icon">🎧</div><p>Trascina l’audio o tocca per selezionarlo</p><small>MP3, WAV</small><div class="file-info" id="dropAudioTxt"></div><input type="file" id="fileAudio" accept="audio/*"><audio id="previewAudio" class="audio-preview" controls></audio></div></section><section class="card"><h2>📱 Piattaforme</h2><div class="platforms"><label class="plat"><input type="checkbox" name="p" value="tiktok" checked><span>🎵 TikTok</span></label><label class="plat"><input type="checkbox" name="p" value="instagram"><span>📸 Instagram</span></label><label class="plat"><input type="checkbox" name="p" value="youtube"><span>▶️ YouTube</span></label><label class="plat"><input type="checkbox" name="p" value="twitter"><span>🐦 Twitter/X</span></label><label class="plat"><input type="checkbox" name="p" value="linkedin"><span>💼 LinkedIn</span></label><label class="plat"><input type="checkbox" name="p" value="youtube_long"><span>🎥 YT Long</span></label><label class="plat"><input type="checkbox" name="p" value="podcast"><span>🎙️ Podcast</span></label></div><div class="options"><label class="checkbox-group"><input type="checkbox" id="autoZoom"><span>✨ Auto-Zoom</span></label><label class="checkbox-group"><input type="checkbox" id="autoCutSilences" checked><span>✂️ Taglia silenzi automaticamente</span></label></div></section><button class="button" id="genBtn" disabled>🚀 Genera clip</button><div class="progress-wrap" id="progressWrap"><div class="stepper"><div class="step" id="step1"><div class="step-icon">✂️</div><div class="step-info"><h3>Taglio</h3><p>Silenzi automatici</p></div></div><div class="step" id="step2"><div class="step-icon">🎤</div><div class="step-info"><h3>Trascrizione</h3><p>Audio in testo</p></div></div><div class="step" id="step3"><div class="step-icon">📝</div><div class="step-info"><h3>Sottotitoli</h3><p>Overlay smart</p></div></div><div class="step" id="step4"><div class="step-icon">📦</div><div class="step-info"><h3>Export</h3><p>Video pronto</p></div></div></div><div class="progress-msg" id="progressMsg">Preparazione...</div></div><div class="results" id="results"><section class="transcript-section"><h3>Trascrizione</h3><div class="transcript-box" id="transcriptBox">Nessun audio.</div></section><section class="result-cards" id="resultCards"></section></div></main></div><script>
let selVideo=null,selAudio=null,pollInterval=null,currentJobId=null;
const stepOrder=["step1","step2","step3","step4"];
const stepMessages=[/silenz/i,/trascrizion/i,/sottotitol/i,/(render|export|completato|fatto)/i];
const stepThreshold=[0,25,50,75];
const videoThumb=document.getElementById("videoThumb");
const previewVideo=document.getElementById("previewVideo");
const thumbText=document.getElementById("thumbText");

function setActiveStep(index){stepOrder.forEach((id,i)=>{const el=document.getElementById(id);if(!el)return;el.classList.toggle("active",i<=index);});}
function updateProgressSteps(message, progress){let index=0;const msg=message||"";for(let i=0;i<stepMessages.length;i++){if(stepMessages[i].test(msg))index=i;}if(progress!==undefined){for(let i=stepThreshold.length-1;i>=0;i--){if(progress>=stepThreshold[i]){index=i;break;}}}
setActiveStep(index);}

function handleDatasetVideo(file){if(!file)return;selVideo=file;const size=(file.size/1024/1024).toFixed(1)+" MB";thumbText.innerHTML=`<strong>${file.name}</strong><span>${size}</span><span class='thumb-action'>Tocca per espandere</span>`;videoThumb.classList.add("has-file");document.getElementById("dropVideoTxt").innerHTML="Thumbnail pronta. Premi per visualizzare.";previewVideo.src=URL.createObjectURL(file);previewVideo.style.display="none";checkReady();}

document.getElementById("dropVideo").addEventListener("click",()=>document.getElementById("fileVideo").click());
document.getElementById("dropVideo").addEventListener("dragover",e=>{e.preventDefault();e.currentTarget.classList.add("drag");});
document.getElementById("dropVideo").addEventListener("dragleave",e=>{e.currentTarget.classList.remove("drag");});
document.getElementById("dropVideo").addEventListener("drop",e=>{e.preventDefault();e.currentTarget.classList.remove("drag");if(e.dataTransfer.files[0])handleDatasetVideo(e.dataTransfer.files[0]);});
document.getElementById("fileVideo").addEventListener("change",e=>{if(e.target.files[0])handleDatasetVideo(e.target.files[0]);});

document.getElementById("videoThumb").addEventListener("click",()=>{if(!selVideo){document.getElementById("fileVideo").click();return;}previewVideo.style.display=previewVideo.style.display==="block"?"none":"block";});

document.getElementById("dropAudio").addEventListener("click",()=>document.getElementById("fileAudio").click());
document.getElementById("dropAudio").addEventListener("dragover",e=>{e.preventDefault();e.currentTarget.classList.add("drag");});
document.getElementById("dropAudio").addEventListener("dragleave",e=>{e.currentTarget.classList.remove("drag");});
document.getElementById("dropAudio").addEventListener("drop",e=>{e.preventDefault();e.currentTarget.classList.remove("drag");if(e.dataTransfer.files[0])onAudio(e.dataTransfer.files[0]);});
document.getElementById("fileAudio").addEventListener("change",e=>{if(e.target.files[0])onAudio(e.target.files[0]);});
document.querySelectorAll("input[name='p']").forEach(cb=>cb.addEventListener("change",e=>togglePlat(e.target)));
;(function(){
    try{
        const optionsEl = document.querySelector('.options');
        const section = document.createElement('section');
        section.className = 'card';
        section.innerHTML = `<h2>🎨 Stile Sottotitoli</h2><div class="subtitle-style" style="display:flex;flex-wrap:wrap;gap:0.75rem;margin-top:0.75rem;"><label style="display:flex;flex-direction:column;font-size:0.9rem;"><span style="font-weight:700;margin-bottom:0.35rem;">Font</span><select id="subtitleFont" name="subtitle_font"><option>Arial</option><option>Impact</option><option>Montserrat</option><option>Oswald</option><option>Bebas Neue</option></select></label><label style="display:flex;flex-direction:column;font-size:0.9rem;"><span style="font-weight:700;margin-bottom:0.35rem;">Colore evidenziata</span><input type="color" id="subtitleHighlight" name="subtitle_highlight_color" value="#FFD700"></label><label style="display:flex;flex-direction:column;font-size:0.9rem;"><span style="font-weight:700;margin-bottom:0.35rem;">Colore testo</span><input type="color" id="subtitleTextColor" name="subtitle_text_color" value="#FFFFFF"></label><label style="display:flex;flex-direction:column;font-size:0.9rem;"><span style="font-weight:700;margin-bottom:0.35rem;">Sfondo</span><select id="subtitleBgStyle" name="subtitle_bg_style"><option value="none">Nessuno</option><option value="black">Nero semi-trasparente</option><option value="white">Bianco semi-trasparente</option><option value="glow">Alone sfocato</option></select></label></div>`;
        if(optionsEl && optionsEl.parentNode) optionsEl.parentNode.insertBefore(section, optionsEl);
    }catch(e){console.warn('subtitle UI insert failed', e)}
})();

function onVideo(f){handleDatasetVideo(f);}
function onAudio(f){if(!f)return;selAudio=f;document.getElementById("previewAudio").src=URL.createObjectURL(f);document.getElementById("previewAudio").style.display="block";document.getElementById("dropAudioTxt").innerHTML="✅ "+f.name;document.getElementById("dropAudio").classList.add("has-file");}
function checkReady(){document.getElementById("genBtn").disabled=!selVideo;}
function togglePlat(cb){cb.closest(".plat").classList.toggle("selected",cb.checked);} 
async function checkStatus(){try{const j=await(await fetch("/api/status")).json();setDot("dotFfmpeg","sFfmpeg",j.ffmpeg,"ffmpeg: "+(j.ffmpeg?"✓":"Mancante"));setDot("dotWhisper","sWhisper",j.whisper,"Whisper: "+(j.whisper?"✓":"No"));}catch(e){}}
function setDot(d,l,ok,t){document.getElementById(d).classList.toggle("off",!ok);document.getElementById(l).textContent=t;}
async function startJob(){if(pollInterval)clearInterval(pollInterval);pollInterval=null;currentJobId=null;const rawPlatforms=Array.from(document.querySelectorAll("input[name=p]:checked")).map(c=>c.value);const platforms=[];rawPlatforms.forEach(p=>p.split(',').forEach(v=>platforms.push(v.trim())));if(!selVideo||!platforms.length){alert("⚠️ Seleziona almeno una piattaforma");return;}const autoZoom=document.getElementById("autoZoom").checked;const autoCutSilences=document.getElementById("autoCutSilences").checked;document.getElementById("genBtn").disabled=true;document.getElementById("progressWrap").classList.add("active");document.getElementById("results").classList.remove("active");document.getElementById("resultCards").innerHTML="";document.getElementById("progressMsg").textContent="Inizio processo...";updateProgressSteps("",0);const formData=new FormData();formData.append("video",selVideo);if(selAudio)formData.append("audio",selAudio);formData.append("auto_zoom",autoZoom);formData.append("cut_silences",autoCutSilences);platforms.forEach(p=>formData.append("platforms",p));const subFont=(document.getElementById("subtitleFont")&&document.getElementById("subtitleFont").value)||"Arial";const subHighlight=(document.getElementById("subtitleHighlight")&&document.getElementById("subtitleHighlight").value)||"#FFD700";const subText=(document.getElementById("subtitleTextColor")&&document.getElementById("subtitleTextColor").value)||"#FFFFFF";const subBg=(document.getElementById("subtitleBgStyle")&&document.getElementById("subtitleBgStyle").value)||"none";formData.append("subtitle_font",subFont);formData.append("subtitle_highlight_color",subHighlight);formData.append("subtitle_text_color",subText);formData.append("subtitle_bg_style",subBg);try{const res=await fetch("/api/job/start",{method:"POST",body:formData});if(!res.ok){const e=await res.json().catch(()=>({}));throw new Error(e.detail||"Errore "+res.status);}const j=await res.json();currentJobId=j.job_id;pollInterval=setInterval(()=>pollJob(currentJobId),1500);}catch(e){alert("❌ "+e.message);resetJob();}}
async function pollJob(id){if(!currentJobId||id!==currentJobId)return;try{const res=await fetch("/api/job/status?job_id="+id);if(!res.ok)return;const j=await res.json();const message=j.message||"";document.getElementById("progressMsg").textContent=message;updateProgressSteps(message,j.progress||0);if(j.status==="done"){clearInterval(pollInterval);pollInterval=null;showResults(j.results);resetJob();}else if(j.status==="error"){clearInterval(pollInterval);pollInterval=null;document.getElementById("progressMsg").textContent="❌ "+j.message;resetJob();}}catch(e){}}
function resetJob(){if(pollInterval)clearInterval(pollInterval);pollInterval=null;document.getElementById("genBtn").disabled=!selVideo;}
function showResults(r){document.getElementById("results").classList.add("active");const f=Object.values(r)[0];document.getElementById("transcriptBox").textContent=f?.transcript||"Nessun audio.";let h="";for(const[k,v]of Object.entries(r)){h+=`<div class="result-card"><div class="result-thumb">${v.icon}</div><div class="result-content"><div class="result-header"><span>${v.label}</span><span class="badge ${v.video_ok?"":"err"}">${v.video_ok?"✅ Pronto":"❌ Errore"}</span></div>${v.video_ok?`<a class="dl-btn" href="/api/download?file=${encodeURIComponent(v.video_name)}" download="${v.video_name}">Scarica ${v.label}</a>`:`<p class="error-text">${v.video_error}</p>`}</div></div>`;}document.getElementById("resultCards").innerHTML=h;}
document.getElementById("genBtn").addEventListener("click",startJob);checkStatus();setInterval(checkStatus,15000);
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
async def start_job(background_tasks: BackgroundTasks, video: UploadFile = File(...), audio: Optional[UploadFile] = File(None), auto_zoom: str = Form("false"), cut_silences: str = Form("true"), platforms: List[str] = Form(...), subtitle_font: str = Form("Arial"), subtitle_highlight_color: str = Form("#FFD700"), subtitle_text_color: str = Form("#FFFFFF"), subtitle_bg_style: str = Form("none")):
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
    background_tasks.add_task(run_job, jid, vp, ap, video.filename, "", platforms, auto_zoom.lower()=="true", cut_silences.lower()=="true", subtitle_font, subtitle_highlight_color, subtitle_text_color, subtitle_bg_style)
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
