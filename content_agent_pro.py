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

# Posizione verticale dei sottotitoli: alignment resta sempre 2 (bottom-center),
# cambia SOLO il MarginV. Fonte di verità unica = frazione dell'altezza dall'ALTO
# dove cade il CENTRO del sottotitolo. DEVE combaciare con i marker dell'anteprima
# (top:% nel blocco HTML): top 10% · 1/4 25% · metà 50% · 3/4 75% · basso 85%.
SUBTITLE_POSITION_FRAC_FROM_TOP = {
    "top": 0.10,
    "one_quarter": 0.25,
    "middle": 0.50,
    "three_quarter": 0.75,
    "bottom": 0.85,
}

def validate_subtitle_position(pos: str) -> str:
    return pos if pos in SUBTITLE_POSITION_FRAC_FROM_TOP else "bottom"

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

    # CLAMP SENZA DRIFT: lo start resta al tempo reale (niente latenza) e la fine
    # viene tagliata all'inizio della parola successiva -> una parola finisce sempre
    # appena prima che parta la prossima: niente overlap, niente doppioni, niente lag.
    MIN_VIS = 0.05  # durata di sicurezza per il raro clamp degenere (start coincidenti)
    result = []
    n = len(adjusted)
    for i, it in enumerate(adjusted):
        s = it["start"]
        e = it["end"]
        if i + 1 < n:
            e = min(e, adjusted[i + 1]["start"])   # taglia all'inizio della successiva
        e = min(e, max_sec)
        if e <= s:                                 # clamp degenere -> durata minima
            e = min(s + MIN_VIS, max_sec)
        result.append({"start": s, "end": e, "text": it["text"]})

    discarded = [words[i]["text"] for i in range(len(words)) if not assigned[i]]
    logger.info("✅ Sottotitoli sincronizzati: %d parole (%d oltre il limite)",
                len(result), len(discarded))
    if discarded:
        logger.warning("⚠️ Parole oltre max_sec, non sottotitolate: %s", discarded)

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

def generate_ass_file(words, output_path, is_portrait, font: str = "Arial", highlight_color: str = "#FFD700", text_color: str = "#FFFFFF", bg_style: str = "none", position: str = "bottom"):
    fs = 88 if is_portrait else 70
    # Solo il margine verticale cambia in base alla posizione scelta; alignment resta 2.
    # MarginV (dal basso) = PlayResY*(1-f) - fs/2, così il CENTRO del testo cade alla
    # frazione f dall'alto, esattamente come il marker mostrato in anteprima.
    position = validate_subtitle_position(position)
    f = SUBTITLE_POSITION_FRAC_FROM_TOP[position]
    mv = max(0, round(1920 * (1 - f) - fs / 2))
    font = validate_subtitle_font(font)
    text_color = clean_hex_color(text_color, "FFFFFF")
    highlight_color = clean_hex_color(highlight_color, "FFD700")
    bg_style = validate_bg_style(bg_style)
    # ASS usa l'ordine BGR (&HBBGGRR&); il pannello/CSS usa RGB (RRGGBB): converti
    # cosi' i colori nel video coincidono esattamente con l'anteprima.
    text_ass = text_color[4:6] + text_color[2:4] + text_color[0:2]
    hl_ass = highlight_color[4:6] + highlight_color[2:4] + highlight_color[0:2]
    # Stile base condiviso: la parola evidenziata si distingue SOLO per il colore,
    # mai per dimensione (nessun override \fs / \fscx).
    # Colori ASS completi a 8 cifre AABBGGRR (alpha: 00 = opaco, FF = trasparente).
    # IMPORTANTE: con BorderStyle=3 (box opaco) libass riempie il box con OutlineColour,
    # NON con BackColour (che colora solo l'ombra). Quindi per gli sfondi box il colore
    # va in outline_ass. "none"/"glow" usano BorderStyle=1 e restano identici a prima.
    outline_ass = "00000000"  # contorno testo nero opaco (none/glow)
    back_ass = "FF000000"     # ombra trasparente (spenta)
    border_style = 1
    outline = 2
    shadow = 0

    if bg_style == "black":
        # Box nero semi-trasparente: alpha 80 = opacita' 0.5 (come .bg-black rgba(0,0,0,.5)).
        outline_ass = "80000000"
        border_style = 3
        outline = 2
        shadow = 0
    elif bg_style == "white":
        # Box bianco semi-trasparente: alpha 73 = opacita' ~0.55 (come .bg-white rgba(255,255,255,.55)).
        outline_ass = "73FFFFFF"
        border_style = 3
        outline = 2
        shadow = 0
    elif bg_style == "glow":
        # Le parole base hanno solo un'ombra scura (come .bg-glow .sub-base);
        # il bagliore colorato e' applicato inline SOLO alla parola evidenziata.
        outline_ass = "00000000"
        back_ass = "FF000000"
        border_style = 1
        outline = 2
        shadow = 1

    header = f"[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV\nStyle: Default,{font},{fs},&H00{text_ass},&H00{hl_ass},&H{outline_ass},&H{back_ass},-1,0,{border_style},{outline},{shadow},2,10,10,{mv}\n\n[Events]\nFormat: Start, End, Style, Text\n"
    # Override inline per la parola evidenziata: SOLO colore (e bagliore colorato
    # se glow), poi \r per ripristinare lo stile Default sulle parole successive.
    if bg_style == "glow":
        hl_open = f"{{\\c&H00{hl_ass}&\\3c&H00{hl_ass}&\\bord3\\blur6\\shad0}}"
    else:
        hl_open = f"{{\\c&H00{hl_ass}&}}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        if any(len(w["text"].split())>2 for w in words):
            for w in words:
                text = w['text'].replace('{', '\\{').replace('}', '\\}')
                # Modalita' frase: nessuna parola "corrente" -> colore testo normale.
                f.write(f"Dialogue: {format_ass_time(w['start'])},{format_ass_time(w['end'])},Default,{text}\n")
        else:
            for chunk in chunk_words(words):
                for i, aw in enumerate(chunk):
                    parts = []
                    for j, w in enumerate(chunk):
                        text = w['text'].replace('{', '\\{').replace('}', '\\}')
                        if j == i:
                            parts.append(f"{hl_open}{text}{{\\r}}")
                        else:
                            parts.append(text)
                    f.write(f"Dialogue: {format_ass_time(aw['start'])},{format_ass_time(aw['end'])},Default,{' '.join(parts)}\n")

def format_ass_time(sec):
    h,m,s = int(sec//3600), int((sec%3600)//60), int(sec%60)
    return f"{h}:{m:02d}:{s:02d}.{int((sec-int(sec))*100):02d}"

def build_video(input_path: Path, audio_path: Optional[Path], output_path: Path, platform_cfg: Dict, keep_segments: List[Dict], words: Optional[List[Dict]], max_sec: int, auto_zoom: bool, subtitle_font: str = "Arial", subtitle_highlight_color: str = "#FFD700", subtitle_text_color: str = "#FFFFFF", subtitle_bg_style: str = "none", subtitle_position: str = "bottom") -> None:
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
            generate_ass_file(words, ass_path, is_portrait, font=subtitle_font, highlight_color=subtitle_highlight_color, text_color=subtitle_text_color, bg_style=subtitle_bg_style, position=subtitle_position)
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

def run_job(job_id, video_path, audio_path, video_filename, user_prompt, selected_platforms, auto_zoom, cut_silences: bool, subtitle_font: str = "Arial", subtitle_highlight_color: str = "#FFD700", subtitle_text_color: str = "#FFFFFF", subtitle_bg_style: str = "none", subtitle_position: str = "bottom"):
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
                    build_video(video_path, audio_path, out, cfg, keep_segments, pw, cfg["max_sec"], auto_zoom, subtitle_font, subtitle_highlight_color, subtitle_text_color, subtitle_bg_style, subtitle_position)
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

HTML = r"""<!DOCTYPE html><html lang="it"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>ClipFlow</title><link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500&family=Montserrat:wght@400;500;700&family=Oswald:wght@400;700&display=swap" rel="stylesheet"><style>*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0B0E14;--surface:#121722;--surface-2:#171E2B;--accent:#1E9FE6;--accent-soft:rgba(30,159,230,.12);--accent-border:rgba(30,159,230,.4);--text:#E6EAF0;--muted:#8893A6;--border:rgba(255,255,255,.07);--success:#2ECC71;--error:#FF5A65;--r:14px;--r-lg:20px;--r-pill:999px}
html,body{min-height:100%}
body{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);font-weight:400;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:760px;margin:0 auto;padding:22px 16px 64px}
.hd{display:flex;flex-direction:column;gap:12px;padding:6px 4px 0}
.brand{display:flex;align-items:center;gap:12px}
.brand .mark{width:40px;height:40px;flex:none}
.brand .mark svg{width:100%;height:100%;display:block}
.brand .name{font-size:23px;font-weight:500;letter-spacing:-.02em}
.tagline{color:var(--muted);font-size:15px}
.status{display:flex;gap:16px;flex-wrap:wrap;margin-top:2px}
.status-item{display:inline-flex;align-items:center;gap:7px;font-size:13px;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--success);box-shadow:0 0 0 3px rgba(46,204,113,.15)}
.dot.off{background:var(--error);box-shadow:0 0 0 3px rgba(255,90,101,.15)}
.flow{display:flex;align-items:center;gap:8px;margin:24px 0 6px}
.flow .f{flex:1;display:flex;flex-direction:column;align-items:center;gap:8px;text-align:center}
.flow .f .n{width:30px;height:30px;border-radius:50%;display:grid;place-items:center;font-size:13px;font-weight:500;background:var(--surface-2);color:var(--muted);border:1px solid var(--border);transition:.25s}
.flow .f.cur .n{background:var(--accent);color:#04111c;border-color:transparent}
.flow .f span{font-size:12px;color:var(--muted);transition:.25s}
.flow .f.cur span{color:var(--text)}
.flow .bar{height:1px;flex:0 0 24px;background:var(--border)}
.wrap:has(#videoThumb.has-file) .flow .f:nth-of-type(2) .n{background:var(--accent);color:#04111c;border-color:transparent}
.wrap:has(#videoThumb.has-file) .flow .f:nth-of-type(2) span{color:var(--text)}
.wrap:has(#genBtn.processing) .flow .f:nth-of-type(3) .n{background:var(--accent);color:#04111c;border-color:transparent}
.wrap:has(#genBtn.processing) .flow .f:nth-of-type(3) span{color:var(--text)}
.sec{margin-top:18px}
.sec>.lab{font-size:13px;font-weight:500;color:var(--muted);letter-spacing:.04em;text-transform:uppercase;margin:0 4px 10px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);padding:18px}
.drop{border:1.5px dashed rgba(255,255,255,.14);border-radius:var(--r);padding:26px 18px;text-align:center;cursor:pointer;transition:.2s;background:var(--surface-2)}
.drop:hover{border-color:var(--accent-border);background:var(--accent-soft)}
.drop.drag{border-color:var(--accent);background:var(--accent-soft)}
.drop.has-file{border-color:var(--accent-border);border-style:solid}
.drop .ic{font-size:25px;display:block;margin-bottom:8px;opacity:.85}
.drop p{font-size:14px;color:var(--text)}
.drop small{display:block;margin-top:4px;color:var(--muted);font-size:12px}
.file-info{margin-top:10px;font-size:13px;color:var(--accent)}
input[type=file]{display:none}
.video-thumb{margin-top:14px;display:none;grid-template-columns:auto 1fr auto;gap:14px;align-items:center;padding:14px;border-radius:var(--r);background:var(--surface);border:1px solid var(--border);cursor:pointer;transition:.2s}
.video-thumb.has-file{display:grid}
.video-thumb:hover{border-color:rgba(255,255,255,.16)}
.thumb-icon{width:46px;height:46px;border-radius:12px;background:var(--accent-soft);display:grid;place-items:center;font-size:22px}
.thumb-details{min-width:0;display:flex;flex-direction:column;gap:3px;text-align:left}
.thumb-title{font-size:12px;color:var(--muted)}
#thumbText{font-size:13px;color:var(--muted);display:flex;flex-direction:column;gap:2px;overflow:hidden}
#thumbText strong{font-weight:500;color:var(--text);font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#thumbText .thumb-action{color:var(--accent);font-size:12px}
.check{width:26px;height:26px;border-radius:50%;background:var(--success);display:grid;place-items:center;color:#04140a;font-size:14px;flex:none}
.video-preview{display:block;width:100%;height:100%;object-fit:contain;background:#000}
.preview-wrap{position:relative;margin:12px auto 0;display:none;line-height:0;max-width:100%;border-radius:var(--r);overflow:hidden;background:#000}
.sub-pos-overlay{position:absolute;inset:0;border-radius:var(--r);overflow:hidden;pointer-events:none}
.sub-pos-hint{position:absolute;top:6px;left:8px;font-size:10px;line-height:1.2;text-transform:uppercase;letter-spacing:.05em;color:rgba(255,255,255,.75);background:rgba(0,0,0,.45);padding:3px 7px;border-radius:var(--r-pill);pointer-events:none}
.sub-pos-marker{position:absolute;left:0;right:0;height:22px;transform:translateY(-50%);display:flex;align-items:center;cursor:pointer;pointer-events:auto}
.sub-pos-line{flex:1;height:2px;background:rgba(255,255,255,.4);box-shadow:0 0 0 1px rgba(0,0,0,.35)}
.sub-pos-tag{font-size:10px;font-weight:600;line-height:1;color:#fff;background:rgba(0,0,0,.5);padding:3px 7px;border-radius:var(--r-pill);margin-left:8px;white-space:nowrap}
.sub-pos-marker:hover .sub-pos-line{background:rgba(255,255,255,.7)}
.sub-pos-marker.active .sub-pos-line{height:3px;background:var(--accent);box-shadow:0 0 8px var(--accent-border)}
.sub-pos-marker.active .sub-pos-tag{background:var(--accent);color:#fff}
.opt-audio{margin-top:12px}
.opt-audio>summary{cursor:pointer;color:var(--muted);font-size:13px;list-style:none;display:flex;align-items:center;gap:6px;padding:6px 2px}
.opt-audio>summary::-webkit-details-marker{display:none}
.opt-audio>summary:before{content:'+';font-size:15px;color:var(--accent)}
.opt-audio[open]>summary:before{content:'\2212'}
.audio-preview{width:100%;margin-top:10px;display:none}
.platforms{display:grid;grid-template-columns:repeat(auto-fill,minmax(94px,1fr));gap:10px}
.plat{position:relative;display:flex;flex-direction:column;align-items:center;gap:9px;padding:14px 8px;border-radius:var(--r);background:var(--surface-2);border:1.5px solid var(--border);cursor:pointer;transition:.18s;user-select:none}
.plat:hover{border-color:rgba(255,255,255,.18)}
.plat input{position:absolute;opacity:0;pointer-events:none}
.plat-logo{width:38px;height:38px;border-radius:11px;background:#fff;display:grid;place-items:center;padding:7px;box-shadow:0 1px 4px rgba(0,0,0,.25)}
.plat-logo svg{width:100%;height:100%;display:block}
.plat-logo.mic{color:#0B0E14}
.plat-name{font-size:12px;color:var(--muted);font-weight:400;text-align:center}
.plat:has(input:checked),.plat.selected{border-color:var(--accent);background:var(--accent-soft);box-shadow:0 0 0 1px var(--accent) inset,0 10px 26px -12px var(--accent)}
.plat:has(input:checked) .plat-name,.plat.selected .plat-name{color:var(--text);font-weight:500}
.plat:has(input:checked):after,.plat.selected:after{content:'';position:absolute;top:8px;right:8px;width:8px;height:8px;border-radius:50%;background:var(--accent);box-shadow:0 0 8px var(--accent)}
.options{display:flex;flex-direction:column;gap:2px}
.toggle{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:13px 2px;cursor:pointer}
.toggle+.toggle{border-top:1px solid var(--border)}
.t-txt{display:flex;flex-direction:column;gap:2px}
.t-txt b{font-weight:500;font-size:14px}
.t-txt small{color:var(--muted);font-size:12px}
.toggle input{position:absolute;opacity:0;width:0;height:0}
.sw{width:42px;height:24px;border-radius:var(--r-pill);background:var(--surface-2);border:1px solid var(--border);position:relative;transition:.2s;flex:none}
.sw:after{content:'';position:absolute;top:2px;left:2px;width:18px;height:18px;border-radius:50%;background:var(--muted);transition:.2s}
.toggle input:checked+.sw{background:var(--accent);border-color:transparent}
.toggle input:checked+.sw:after{background:#fff;transform:translateX(18px)}
.card .card{background:transparent;border:none;padding:0;margin:0 0 6px}
.card h2{font-size:13px;font-weight:500;color:var(--muted);letter-spacing:.04em;text-transform:uppercase;margin-bottom:12px}
.subtitle-style{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;margin-top:4px}
.sub-controls{display:flex;flex-direction:column;gap:12px}
.subtitle-style label{display:flex;flex-direction:column;gap:5px}
.subtitle-style label>span{font-weight:500;color:var(--text);font-size:12px}
.subtitle-style select,.subtitle-style input[type=color]{background:var(--surface-2);border:1px solid var(--border);color:var(--text);border-radius:10px;padding:8px 9px;font:inherit;font-size:13px}
.subtitle-style input[type=color]{padding:2px;height:38px;width:54px;cursor:pointer}
.sub-preview{display:flex;flex-direction:column;gap:8px}
.sub-preview-lab{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-weight:500}
.sub-stage{min-height:120px;border-radius:var(--r);display:grid;place-items:center;padding:18px;text-align:center;background:linear-gradient(135deg,#42495a,#5b6376 60%,#6b7384);box-shadow:inset 0 0 0 1px var(--border);overflow:hidden}
.sub-line{display:inline-block;font-size:27px;font-weight:700;line-height:1.2;font-family:Impact,sans-serif;margin:0}
.sub-word,.sub-base{transition:color .08s}
.bg-none .sub-line{text-shadow:0 1px 2px rgba(0,0,0,.85),0 0 3px rgba(0,0,0,.7)}
.bg-black .sub-line{background:rgba(0,0,0,.5);padding:6px 14px;border-radius:8px}
.bg-white .sub-line{background:rgba(255,255,255,.55);padding:6px 14px;border-radius:8px}
.bg-glow .sub-word{text-shadow:0 0 6px currentColor,0 0 14px currentColor,0 0 22px currentColor}
.bg-glow .sub-base{text-shadow:0 1px 2px rgba(0,0,0,.7)}
@media(max-width:560px){.subtitle-style{grid-template-columns:1fr}}
.gen-wrap{position:relative;margin-top:22px}
.gen{position:relative;overflow:hidden;width:100%;padding:16px;border:none;border-radius:var(--r);background:var(--accent);color:#04111c;font:inherit;font-size:16px;font-weight:500;cursor:pointer;transition:background .2s,filter .2s,transform .2s;box-shadow:0 12px 30px -12px var(--accent)}
.gen:hover:not(:disabled):not(.processing){filter:brightness(1.08);transform:translateY(-1px)}
.gen:disabled:not(.processing):not(.done){background:var(--surface-2);color:var(--muted);box-shadow:none;cursor:not-allowed}
.gen-fill{position:absolute;left:0;top:0;bottom:0;width:0;background:var(--accent);border-radius:inherit;transition:width .45s ease;z-index:0;pointer-events:none}
.gen-label{position:relative;z-index:1;display:inline-flex;align-items:center;justify-content:center;gap:10px}
.gen.processing{background:#2C3A47;color:#fff;cursor:default;box-shadow:none}
.gen.done{background:#2C3A47;color:#04111c;cursor:pointer;box-shadow:0 12px 30px -12px var(--accent)}
.gen.done .gen-fill{width:100%!important}
.gen-done-overlay{display:none;position:absolute;inset:0;align-items:center;justify-content:center;gap:16px;z-index:2;pointer-events:none;color:#04111c;font-size:16px;font-weight:500}
.gen.done+.gen-done-overlay{display:flex}
.gen.done .gen-label{opacity:0}
.gen-check{font-size:4.6em;line-height:0;color:#FF2800;-webkit-text-stroke:2px rgba(255,255,255,.95);paint-order:stroke fill;text-shadow:0 2px 8px rgba(0,0,0,.6),0 0 14px rgba(0,0,0,.4);position:relative;top:-2px}
.results{display:none;margin-top:22px}
.results.active{display:block}
.transcript-section{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-lg);padding:16px;margin-bottom:14px}
.transcript-section h3{font-size:13px;font-weight:500;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);margin-bottom:10px}
.transcript-box{color:var(--muted);font-size:14px;line-height:1.7;max-height:160px;overflow-y:auto}
.result-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.result-card{display:flex;gap:12px;align-items:center;background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:14px}
.result-thumb{width:46px;height:46px;border-radius:12px;display:grid;place-items:center;font-size:20px;background:var(--accent-soft);flex:none}
.result-content{flex:1;min-width:0;display:flex;flex-direction:column;gap:8px}
.result-header{display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap}
.result-header span{font-weight:500;font-size:14px}
.badge{padding:4px 10px;border-radius:8px;font-size:12px;font-weight:500;background:rgba(46,204,113,.14);color:var(--success)}
.badge.err{background:rgba(255,90,101,.14);color:var(--error)}
.dl-btn{display:inline-flex;align-items:center;justify-content:center;padding:9px 14px;border-radius:10px;background:var(--accent);color:#04111c;font-weight:500;text-decoration:none;font-size:13px;transition:.2s}
.dl-btn:hover{filter:brightness(1.08)}
.error-text{color:var(--error);font-size:13px}
@media(max-width:520px){.wrap{padding:18px 12px 48px}.stepper{grid-template-columns:repeat(2,1fr)}.flow .f span{font-size:11px}}</style></head><body><div class="wrap">
<header class="hd"><div class="brand"><span class="mark"><svg viewBox="0 0 32 32" fill="none"><rect x="3" y="3" width="26" height="26" rx="9" fill="#1E9FE6"/><path d="M13 11l8 5-8 5z" fill="#fff"/></svg></span><span class="name">ClipFlow</span></div><div class="tagline">Un video. Tutti i social. Un tap.</div><div class="status"><span class="status-item"><span class="dot" id="dotFfmpeg"></span><span id="sFfmpeg">Verifica ffmpeg…</span></span><span class="status-item"><span class="dot" id="dotWhisper"></span><span id="sWhisper">Verifica Whisper…</span></span></div></header>
<div class="flow"><div class="f cur"><span class="n">1</span><span>Carica</span></div><div class="bar"></div><div class="f"><span class="n">2</span><span>Configura</span></div><div class="bar"></div><div class="f"><span class="n">3</span><span>Genera</span></div></div>
<section class="sec"><div class="lab">Il tuo video</div><div class="card"><div class="drop" id="dropVideo"><span class="ic">⬆️</span><p>Trascina il video qui o tocca per caricarlo</p><small>MP4 · MOV · WEBM</small><div class="file-info" id="dropVideoTxt"></div><input type="file" id="fileVideo" accept="video/*"><div class="video-thumb" id="videoThumb"><div class="thumb-icon">🎬</div><div class="thumb-details"><div class="thumb-title">Video caricato</div><div id="thumbText"></div></div><div class="check">✓</div></div><div id="previewWrap" class="preview-wrap"><video id="previewVideo" class="video-preview" controls playsinline></video><div class="sub-pos-overlay" id="subPosOverlay"><span class="sub-pos-hint">Posizione sottotitoli</span><div class="sub-pos-marker" data-pos="top" style="top:10%"><span class="sub-pos-line"></span><span class="sub-pos-tag">In alto</span></div><div class="sub-pos-marker" data-pos="one_quarter" style="top:25%"><span class="sub-pos-line"></span><span class="sub-pos-tag">A 1/4</span></div><div class="sub-pos-marker" data-pos="middle" style="top:50%"><span class="sub-pos-line"></span><span class="sub-pos-tag">A metà</span></div><div class="sub-pos-marker" data-pos="three_quarter" style="top:75%"><span class="sub-pos-line"></span><span class="sub-pos-tag">A 3/4</span></div><div class="sub-pos-marker active" data-pos="bottom" style="top:85%"><span class="sub-pos-line"></span><span class="sub-pos-tag">In basso</span></div></div></div></div><details class="opt-audio"><summary>Aggiungi una traccia audio (opzionale)</summary><div class="drop" id="dropAudio" style="margin-top:8px;padding:18px"><span class="ic">🎧</span><p>Trascina l'audio o tocca per caricarlo</p><small>MP3 · WAV</small><div class="file-info" id="dropAudioTxt"></div><input type="file" id="fileAudio" accept="audio/*"><audio id="previewAudio" class="audio-preview" controls></audio></div></details></div></section>
<section class="sec"><div class="lab">Dove pubblicare</div><div class="card"><div class="platforms"><label class="plat"><input type="checkbox" name="p" value="tiktok" checked><span class="plat-logo"><svg viewBox="0 0 256 290" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" preserveAspectRatio="xMidYMid"> <g> <path d="M189.720224,104.421475 C208.398189,117.766281 231.279538,125.618095 255.992548,125.618095 L255.992548,78.0872726 C251.315611,78.0882654 246.650588,77.6008156 242.074913,76.6318726 L242.074913,114.045382 C217.363889,114.045382 194.485518,106.193568 175.80259,92.8497541 L175.80259,189.846306 C175.80259,238.368905 136.447224,277.701437 87.902784,277.701437 C69.7897057,277.701437 52.9543216,272.228299 38.9691786,262.841664 C54.9309256,279.153859 77.1908018,289.273158 101.81744,289.273158 C150.364858,289.273158 189.72221,249.940626 189.72221,201.416041 L189.72221,104.421475 L189.720224,104.421475 Z M206.889179,56.4687254 C197.343701,46.0456391 191.076347,32.5757434 189.720224,17.6842019 L189.720224,11.5707278 L176.531282,11.5707278 C179.851103,30.497877 191.174632,46.6681056 206.889179,56.4687254 L206.889179,56.4687254 Z M69.6735517,225.606854 C64.3403943,218.617757 61.4583846,210.068027 61.4712906,201.277053 C61.4712906,179.084685 79.472186,161.090739 101.680438,161.090739 C105.819294,161.089747 109.933331,161.723134 113.877603,162.974023 L113.877603,114.380938 C109.268175,113.749536 104.616057,113.481488 99.9659254,113.579773 L99.9659254,151.402303 C96.0186741,150.151413 91.9026521,149.516041 87.7628035,149.520012 C65.5545513,149.520012 47.5546487,167.511972 47.5546487,189.707318 C47.5546487,205.401018 56.552118,218.98806 69.6735517,225.606854 Z" fill="#FF004F"></path> <path d="M175.80259,92.8487613 C194.485518,106.192575 217.363889,114.044389 242.074913,114.044389 L242.074913,76.6308799 C228.281375,73.6942679 216.070311,66.4897401 206.889179,56.4687254 C191.173639,46.6671128 179.851103,30.4968842 176.531282,11.5707278 L141.8876,11.5707278 L141.8876,201.414056 C141.809172,223.545865 123.839052,241.466346 101.678453,241.466346 C88.6195635,241.466346 77.0180599,235.24466 69.6705734,225.606854 C56.5501325,218.98806 47.5526631,205.400025 47.5526631,189.708311 C47.5526631,167.51495 65.5525657,149.521004 87.760818,149.521004 C92.0158278,149.521004 96.1169583,150.183182 99.9639399,151.403295 L99.9639399,113.580765 C52.272289,114.565593 13.9166419,153.513923 13.9166419,201.415048 C13.9166419,225.326893 23.4680767,247.004014 38.9701714,262.842657 C52.9553144,272.228299 69.7906985,277.70243 87.9037768,277.70243 C136.449209,277.70243 175.803582,238.367912 175.803582,189.846306 L175.803582,92.8487613 L175.80259,92.8487613 Z" fill="#000000"></path> <path d="M242.074913,76.6308799 L242.074913,66.5145593 C229.636505,66.5334219 217.442318,63.0517795 206.889179,56.4677326 C216.231139,66.6902795 228.532545,73.7389425 242.074913,76.6308799 Z M176.531282,11.5707278 C176.214589,9.76190185 175.971361,7.9411627 175.80259,6.11347418 L175.80259,0 L127.968973,0 L127.968973,189.845313 C127.89253,211.974144 109.923403,229.894625 87.760818,229.894625 C81.2542071,229.894625 75.1109499,228.350869 69.6705734,225.607847 C77.0180599,235.24466 88.6195635,241.465353 101.678453,241.465353 C123.837066,241.465353 141.810164,223.546857 141.8876,201.415048 L141.8876,11.5707278 L176.531282,11.5707278 Z M99.9659254,113.580765 L99.9659254,102.811203 C95.9690357,102.265179 91.9393845,101.991175 87.9047695,101.99315 C39.3553659,101.99315 0,141.326686 0,189.845313 C0,220.263769 15.4673478,247.071522 38.9711641,262.840672 C23.4690694,247.003021 13.9176347,225.324907 13.9176347,201.414056 C13.9176347,153.513923 52.272289,114.565593 99.9659254,113.580765 Z" fill="#00F2EA"></path> </g> </svg></span><span class="plat-name">TikTok</span></label><label class="plat"><input type="checkbox" name="p" value="instagram"><span class="plat-logo"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-label="Instagram" role="img" viewBox="0 0 512 512"><path d="m0 0H512V512H0" id="ig-b"/><use fill="url(#ig-a)" xlink:href="#ig-b"/><use fill="url(#ig-c)" xlink:href="#ig-b"/><radialGradient id="ig-a" cx=".4" cy="1" r="1"><stop offset=".1" stop-color="#fd5"/><stop offset=".5" stop-color="#ff543e"/><stop offset="1" stop-color="#c837ab"/></radialGradient><linearGradient id="ig-c" x2=".2" y2="1"><stop offset=".1" stop-color="#3771c8"/><stop offset=".5" stop-color="#60f" stop-opacity="0"/></linearGradient><g fill="none" stroke="#fff" stroke-width="30"><rect width="308" height="308" x="102" y="102" rx="81"/><circle cx="256" cy="256" r="72"/><circle cx="347" cy="165" r="6"/></g></svg></span><span class="plat-name">Instagram</span></label><label class="plat"><input type="checkbox" name="p" value="youtube"><span class="plat-logo"><svg viewBox="0 0 256 180" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" preserveAspectRatio="xMidYMid"> <g> <path d="M250.346231,28.0746923 C247.358133,17.0320558 238.732098,8.40602109 227.689461,5.41792308 C207.823743,0 127.868333,0 127.868333,0 C127.868333,0 47.9129229,0.164179487 28.0472049,5.58210256 C17.0045684,8.57020058 8.37853373,17.1962353 5.39043571,28.2388718 C-0.618533519,63.5374615 -2.94988224,117.322662 5.5546152,151.209308 C8.54271322,162.251944 17.1687479,170.877979 28.2113844,173.866077 C48.0771024,179.284 128.032513,179.284 128.032513,179.284 C128.032513,179.284 207.987923,179.284 227.853641,173.866077 C238.896277,170.877979 247.522312,162.251944 250.51041,151.209308 C256.847738,115.861464 258.801474,62.1091 250.346231,28.0746923 Z" fill="#FF0000"></path> <polygon fill="#FFFFFF" points="102.420513 128.06 168.749025 89.642 102.420513 51.224"></polygon> </g> </svg></span><span class="plat-name">YouTube</span></label><label class="plat"><input type="checkbox" name="p" value="twitter"><span class="plat-logo"><svg viewBox="0 0 251 256" version="1.1" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid"> <g> <path d="M149.078767,108.398529 L242.331303,0 L220.233437,0 L139.262272,94.1209195 L74.5908396,0 L0,0 L97.7958952,142.3275 L0,256 L22.0991185,256 L107.606755,156.605109 L175.904525,256 L250.495364,256 L149.07334,108.398529 L149.078767,108.398529 Z M118.810995,143.581438 L108.902233,129.408828 L30.0617399,16.6358981 L64.0046968,16.6358981 L127.629893,107.647252 L137.538655,121.819862 L220.243874,240.120681 L186.300917,240.120681 L118.810995,143.586865 L118.810995,143.581438 Z" fill="#000000"></path> </g> </svg></span><span class="plat-name">X</span></label><label class="plat"><input type="checkbox" name="p" value="linkedin"><span class="plat-logo"><svg viewBox="0 0 256 256" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" preserveAspectRatio="xMidYMid"> <g> <path d="M218.123122,218.127392 L180.191928,218.127392 L180.191928,158.724263 C180.191928,144.559023 179.939053,126.323993 160.463756,126.323993 C140.707926,126.323993 137.685284,141.757585 137.685284,157.692986 L137.685284,218.123441 L99.7540894,218.123441 L99.7540894,95.9665207 L136.168036,95.9665207 L136.168036,112.660562 L136.677736,112.660562 C144.102746,99.9650027 157.908637,92.3824528 172.605689,92.9280076 C211.050535,92.9280076 218.138927,118.216023 218.138927,151.114151 L218.123122,218.127392 Z M56.9550587,79.2685282 C44.7981969,79.2707099 34.9413443,69.4171797 34.9391618,57.260052 C34.93698,45.1029244 44.7902948,35.2458562 56.9471566,35.2436736 C69.1040185,35.2414916 78.9608713,45.0950217 78.963054,57.2521493 C78.9641017,63.090208 76.6459976,68.6895714 72.5186979,72.8184433 C68.3913982,76.9473153 62.7929898,79.26748 56.9550587,79.2685282 M75.9206558,218.127392 L37.94995,218.127392 L37.94995,95.9665207 L75.9206558,95.9665207 L75.9206558,218.127392 Z M237.033403,0.0182577091 L18.8895249,0.0182577091 C8.57959469,-0.0980923971 0.124827038,8.16056231 -0.001,18.4706066 L-0.001,237.524091 C0.120519052,247.839103 8.57460631,256.105934 18.8895249,255.9977 L237.033403,255.9977 C247.368728,256.125818 255.855922,247.859464 255.999,237.524091 L255.999,18.4548016 C255.851624,8.12438979 247.363742,-0.133792868 237.033403,0.000790807055" fill="#0A66C2"></path> </g> </svg></span><span class="plat-name">LinkedIn</span></label><label class="plat"><input type="checkbox" name="p" value="youtube_long"><span class="plat-logo"><svg viewBox="0 0 256 180" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" preserveAspectRatio="xMidYMid"> <g> <path d="M250.346231,28.0746923 C247.358133,17.0320558 238.732098,8.40602109 227.689461,5.41792308 C207.823743,0 127.868333,0 127.868333,0 C127.868333,0 47.9129229,0.164179487 28.0472049,5.58210256 C17.0045684,8.57020058 8.37853373,17.1962353 5.39043571,28.2388718 C-0.618533519,63.5374615 -2.94988224,117.322662 5.5546152,151.209308 C8.54271322,162.251944 17.1687479,170.877979 28.2113844,173.866077 C48.0771024,179.284 128.032513,179.284 128.032513,179.284 C128.032513,179.284 207.987923,179.284 227.853641,173.866077 C238.896277,170.877979 247.522312,162.251944 250.51041,151.209308 C256.847738,115.861464 258.801474,62.1091 250.346231,28.0746923 Z" fill="#FF0000"></path> <polygon fill="#FFFFFF" points="102.420513 128.06 168.749025 89.642 102.420513 51.224"></polygon> </g> </svg></span><span class="plat-name">YouTube Long</span></label><label class="plat"><input type="checkbox" name="p" value="podcast"><span class="plat-logo mic"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="2" width="6" height="11" rx="3"/><path d="M5 10a7 7 0 0 0 14 0"/><line x1="12" y1="17" x2="12" y2="22"/><line x1="8.5" y1="22" x2="15.5" y2="22"/></svg></span><span class="plat-name">Podcast</span></label></div></div></section>
<section class="sec"><div class="lab">Elaborazioni</div><div class="card"><div class="options"><label class="toggle"><span class="t-txt"><b>Taglia silenzi</b><small>Rimuove automaticamente le pause</small></span><input type="checkbox" id="autoCutSilences" checked><span class="sw"></span></label><label class="toggle"><span class="t-txt"><b>Auto-zoom</b><small>Zoom dinamico durante il parlato</small></span><input type="checkbox" id="autoZoom"><span class="sw"></span></label></div></div></section>
<div class="gen-wrap"><button class="gen" id="genBtn" disabled><span class="gen-fill" id="genFill"></span><span class="gen-label" id="genLabel"><span id="genText">Genera clip</span></span></button><div class="gen-done-overlay" id="genDoneOverlay"><span class="gen-check">✓</span><span>Fatto</span></div></div>
<div class="results" id="results"><section class="transcript-section"><h3>Trascrizione</h3><div class="transcript-box" id="transcriptBox">Nessun audio.</div></section><section class="result-cards" id="resultCards"></section></div>
</div><script>
let selVideo=null,selAudio=null,pollInterval=null,currentJobId=null;
const videoThumb=document.getElementById("videoThumb");
const previewVideo=document.getElementById("previewVideo");
const thumbText=document.getElementById("thumbText");

function setGenFill(p){const f=document.getElementById("genFill");if(f)f.style.width=Math.max(0,Math.min(100,p))+"%";}
function setGenState(state){const b=document.getElementById("genBtn"),t=document.getElementById("genText");if(!b)return;b.classList.remove("processing","done");if(state==="processing"){b.classList.add("processing");if(t)t.textContent="In preparazione";setGenFill(0);}else if(state==="done"){b.classList.add("done");if(t)t.textContent="Fatto";setGenFill(100);}else{if(t)t.textContent="Genera clip";setGenFill(0);}}

function fitPreview(){const v=previewVideo,wrap=document.getElementById("previewWrap");if(!v.videoWidth||!v.videoHeight||wrap.style.display==="none")return;const parent=wrap.parentElement,cs=getComputedStyle(parent);const padX=parseFloat(cs.paddingLeft)+parseFloat(cs.paddingRight);const availW=Math.max(0,parent.clientWidth-padX);const maxH=Math.min(window.innerHeight*0.7,460);const ar=v.videoWidth/v.videoHeight;let dispW=availW,dispH=availW/ar;if(dispH>maxH){dispH=maxH;dispW=maxH*ar;}wrap.style.width=dispW+"px";wrap.style.height=dispH+"px";}
function handleDatasetVideo(file){if(!file)return;selVideo=file;const size=(file.size/1024/1024).toFixed(1)+" MB";thumbText.innerHTML=`<strong>${file.name}</strong><span>${size}</span>`;videoThumb.classList.add("has-file");document.getElementById("dropVideoTxt").innerHTML="Anteprima del video";previewVideo.src=URL.createObjectURL(file);previewVideo.style.display="block";document.getElementById("previewWrap").style.display="block";checkReady();}
previewVideo.addEventListener("loadedmetadata",fitPreview);
window.addEventListener("resize",fitPreview);

document.getElementById("dropVideo").addEventListener("click",()=>document.getElementById("fileVideo").click());
document.getElementById("dropVideo").addEventListener("dragover",e=>{e.preventDefault();e.currentTarget.classList.add("drag");});
document.getElementById("dropVideo").addEventListener("dragleave",e=>{e.currentTarget.classList.remove("drag");});
document.getElementById("dropVideo").addEventListener("drop",e=>{e.preventDefault();e.currentTarget.classList.remove("drag");if(e.dataTransfer.files[0])handleDatasetVideo(e.dataTransfer.files[0]);});
document.getElementById("fileVideo").addEventListener("change",e=>{if(e.target.files[0])handleDatasetVideo(e.target.files[0]);});

document.getElementById("videoThumb").addEventListener("click",()=>{if(!selVideo){document.getElementById("fileVideo").click();return;}const w=document.getElementById("previewWrap");w.style.display=w.style.display==="block"?"none":"block";if(w.style.display==="block")fitPreview();});
let subPosition="bottom";
document.querySelectorAll(".sub-pos-marker").forEach(m=>m.addEventListener("click",e=>{e.stopPropagation();subPosition=m.dataset.pos;document.querySelectorAll(".sub-pos-marker").forEach(x=>x.classList.toggle("active",x===m));}));

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
        section.innerHTML = `<h2>🎨 Stile Sottotitoli</h2><div class="subtitle-style"><div class="sub-controls"><label><span>Font</span><select id="subtitleFont" name="subtitle_font"><option>Arial</option><option>Impact</option><option>Montserrat</option><option>Oswald</option><option>Bebas Neue</option></select></label><label><span>Colore evidenziata</span><input type="color" id="subtitleHighlight" name="subtitle_highlight_color" value="#FFD700"></label><label><span>Colore testo</span><input type="color" id="subtitleTextColor" name="subtitle_text_color" value="#FFFFFF"></label><label><span>Sfondo</span><select id="subtitleBgStyle" name="subtitle_bg_style"><option value="none">Nessuno</option><option value="black">Nero semi-trasparente</option><option value="white">Bianco semi-trasparente</option><option value="glow">Alone sfocato</option></select></label></div><div class="sub-preview"><span class="sub-preview-lab">Anteprima dal vivo</span><div class="sub-stage bg-none" id="subStage"><p class="sub-line" id="subLine"><span class="sub-base" id="subBase">Clip</span><span class="sub-word" id="subWord">Flow</span></p></div></div></div>`;
        if(optionsEl && optionsEl.parentNode) optionsEl.parentNode.insertBefore(section, optionsEl);
        const subLine=section.querySelector('#subLine'),subBase=section.querySelector('#subBase'),subWord=section.querySelector('#subWord'),subStage=section.querySelector('#subStage');
        const fontEl=section.querySelector('#subtitleFont'),hlEl=section.querySelector('#subtitleHighlight'),txtEl=section.querySelector('#subtitleTextColor'),bgEl=section.querySelector('#subtitleBgStyle');
        const FONT_STACK={'Arial':'Arial, sans-serif','Impact':'Impact, sans-serif','Montserrat':"'Montserrat', sans-serif",'Oswald':"'Oswald', sans-serif",'Bebas Neue':"'Bebas Neue', sans-serif"};
        function applySubPreview(){subLine.style.fontFamily=FONT_STACK[fontEl.value]||'Impact, sans-serif';subBase.style.color=txtEl.value;subWord.style.color=hlEl.value;subStage.className='sub-stage bg-'+bgEl.value;}
        [fontEl,hlEl,txtEl,bgEl].forEach(el=>{el.addEventListener('input',applySubPreview);el.addEventListener('change',applySubPreview);});
        applySubPreview();
    }catch(e){console.warn('subtitle UI insert failed', e)}
})();

function onVideo(f){handleDatasetVideo(f);}
function onAudio(f){if(!f)return;selAudio=f;document.getElementById("previewAudio").src=URL.createObjectURL(f);document.getElementById("previewAudio").style.display="block";document.getElementById("dropAudioTxt").innerHTML="✅ "+f.name;document.getElementById("dropAudio").classList.add("has-file");}
function checkReady(){document.getElementById("genBtn").disabled=!selVideo;}
function togglePlat(cb){cb.closest(".plat").classList.toggle("selected",cb.checked);} 
async function checkStatus(){try{const j=await(await fetch("/api/status")).json();setDot("dotFfmpeg","sFfmpeg",j.ffmpeg,"ffmpeg: "+(j.ffmpeg?"✓":"Mancante"));setDot("dotWhisper","sWhisper",j.whisper,"Whisper: "+(j.whisper?"✓":"No"));}catch(e){}}
function setDot(d,l,ok,t){document.getElementById(d).classList.toggle("off",!ok);document.getElementById(l).textContent=t;}
async function startJob(){if(pollInterval)clearInterval(pollInterval);pollInterval=null;currentJobId=null;const rawPlatforms=Array.from(document.querySelectorAll("input[name=p]:checked")).map(c=>c.value);const platforms=[];rawPlatforms.forEach(p=>p.split(',').forEach(v=>platforms.push(v.trim())));if(!selVideo||!platforms.length){alert("⚠️ Seleziona almeno una piattaforma");return;}const autoZoom=document.getElementById("autoZoom").checked;const autoCutSilences=document.getElementById("autoCutSilences").checked;document.getElementById("genBtn").disabled=true;setGenState("processing");document.getElementById("results").classList.remove("active");document.getElementById("resultCards").innerHTML="";const formData=new FormData();formData.append("video",selVideo);if(selAudio)formData.append("audio",selAudio);formData.append("auto_zoom",autoZoom);formData.append("cut_silences",autoCutSilences);platforms.forEach(p=>formData.append("platforms",p));const subFont=(document.getElementById("subtitleFont")&&document.getElementById("subtitleFont").value)||"Arial";const subHighlight=(document.getElementById("subtitleHighlight")&&document.getElementById("subtitleHighlight").value)||"#FFD700";const subText=(document.getElementById("subtitleTextColor")&&document.getElementById("subtitleTextColor").value)||"#FFFFFF";const subBg=(document.getElementById("subtitleBgStyle")&&document.getElementById("subtitleBgStyle").value)||"none";formData.append("subtitle_font",subFont);formData.append("subtitle_highlight_color",subHighlight);formData.append("subtitle_text_color",subText);formData.append("subtitle_bg_style",subBg);formData.append("subtitle_position",subPosition);try{const res=await fetch("/api/job/start",{method:"POST",body:formData});if(!res.ok){const e=await res.json().catch(()=>({}));throw new Error(e.detail||"Errore "+res.status);}const j=await res.json();currentJobId=j.job_id;pollInterval=setInterval(()=>pollJob(currentJobId),1500);}catch(e){alert("❌ "+e.message);resetJob();}}
async function pollJob(id){if(!currentJobId||id!==currentJobId)return;try{const res=await fetch("/api/job/status?job_id="+id);if(!res.ok)return;const j=await res.json();setGenFill(j.progress||0);if(j.status==="done"){clearInterval(pollInterval);pollInterval=null;setGenState("done");document.getElementById("genBtn").disabled=!selVideo;showResults(j.results);}else if(j.status==="error"){clearInterval(pollInterval);pollInterval=null;alert("❌ "+j.message);resetJob();}}catch(e){}}
function resetJob(){if(pollInterval)clearInterval(pollInterval);pollInterval=null;setGenState("idle");document.getElementById("genBtn").disabled=!selVideo;}
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
async def start_job(background_tasks: BackgroundTasks, video: UploadFile = File(...), audio: Optional[UploadFile] = File(None), auto_zoom: str = Form("false"), cut_silences: str = Form("true"), platforms: List[str] = Form(...), subtitle_font: str = Form("Arial"), subtitle_highlight_color: str = Form("#FFD700"), subtitle_text_color: str = Form("#FFFFFF"), subtitle_bg_style: str = Form("none"), subtitle_position: str = Form("bottom")):
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
    background_tasks.add_task(run_job, jid, vp, ap, video.filename, "", platforms, auto_zoom.lower()=="true", cut_silences.lower()=="true", subtitle_font, subtitle_highlight_color, subtitle_text_color, subtitle_bg_style, subtitle_position)
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
