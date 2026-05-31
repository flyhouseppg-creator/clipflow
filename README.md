# Content Agent Pro

Backend Python per generare clip social ottimizzate con `ffmpeg`, `FastAPI` e `faster-whisper`.

## Come eseguire

```bash
python content_agent_pro.py
```

## Deploy su Railway

- Railway usa `HOST=0.0.0.0` e `PORT` dal runtime
- `requirements.txt`, `Dockerfile`, `Procfile` e `runtime.txt` sono già presenti
- il `Dockerfile` installa anche `ffmpeg`

## File principali

- `content_agent_pro.py` - applicazione FastAPI
- `Dockerfile` - immagine container pronta per Railway
- `requirements.txt` - dipendenze Python
- `Procfile` - comando di avvio
- `runtime.txt` - versione Python 3.11
