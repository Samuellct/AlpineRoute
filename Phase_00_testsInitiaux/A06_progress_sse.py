# A06 - test SSE (Server-Sent Events) pour progress bar

import asyncio
import json
import time
import uuid
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse


# -- stockage des jobs avec dict
jobs: dict = {}


@asynccontextmanager
async def lifespan(app):
    yield
    jobs.clear()


app = FastAPI(title="AlpineRoute SSE Test", version="0.1", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- simulation d'un calcul de route avec etapes
def fake_computation(job_id: str):
    """Simule les etapes du pipeline (load DEM, cost, pathfind, etc)."""
    steps = [
        ("load_dem", "Chargement DEM...", 15),
        ("compute_cost", "Calcul surface de cout...", 30),
        ("pathfinding", "Dijkstra en cours...", 60),
        ("export", "Export GeoJSON/GPX...", 85),
        ("done", "Termine !", 100),
    ]

    jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "step": "init",
        "message": "Demarrage...",
    }

    for step_id, msg, pct in steps:
        # simule du travail (0.5-2s par etape)
        delay = 0.5 + (pct - jobs[job_id]["progress"]) * 0.04
        time.sleep(delay)

        jobs[job_id].update({
            "progress": pct,
            "step": step_id,
            "message": msg,
        })

    jobs[job_id]["status"] = "completed"
    jobs[job_id]["result"] = {
        "distance_km": 4.2,
        "dplus_m": 1126,
        "time_h": 3.5,
    }


@app.post("/calculate-async")
async def calculate_async():
    """Lance un calcul en background, retourne un job_id."""
    job_id = str(uuid.uuid4())[:8]
    t = threading.Thread(target=fake_computation, args=(job_id,), daemon=True)
    t.start()
    return {"job_id": job_id}


@app.get("/progress/{job_id}")
async def progress_sse(job_id: str):
    """Endpoint SSE qui streame la progression du job."""

    async def event_generator():
        last_pct = -1
        while True:
            job = jobs.get(job_id)
            if job is None:
                yield f"event: error\ndata: {json.dumps({'error': 'job not found'})}\n\n"
                break

            pct = job["progress"]
            if pct != last_pct:
                data = {
                    "progress": pct,
                    "step": job["step"],
                    "message": job["message"],
                    "status": job["status"],
                }
                # si termine, ajoute le resultat
                if job["status"] == "completed" and "result" in job:
                    data["result"] = job["result"]

                yield f"data: {json.dumps(data)}\n\n"
                last_pct = pct

                if job["status"] == "completed":
                    break

            await asyncio.sleep(0.2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # pour nginx
        },
    )


# -- page HTML de test embarquee
TEST_HTML = """<!DOCTYPE html>
<html>
<head>
<title>SSE Progress Test</title>
<style>
  body { font-family: monospace; max-width: 600px; margin: 50px auto; background: #1a1a2e; color: #eee; }
  .bar-container { background: #333; border-radius: 8px; overflow: hidden; height: 30px; margin: 20px 0; }
  .bar { background: linear-gradient(90deg, #00e5ff, #00b8d4); height: 100%; transition: width 0.3s; display: flex; align-items: center; padding-left: 10px; font-weight: bold; }
  button { background: #00e5ff; color: #000; border: none; padding: 10px 24px; border-radius: 6px; cursor: pointer; font-size: 14px; font-family: monospace; }
  button:hover { background: #00b8d4; }
  #log { background: #0d0d1a; padding: 12px; border-radius: 6px; max-height: 300px; overflow-y: auto; font-size: 12px; }
  .log-line { margin: 2px 0; }
  .result { background: #1e4620; padding: 12px; border-radius: 6px; margin-top: 10px; }
</style>
</head>
<body>
<h2>AlpineRoute - SSE Progress Test</h2>
<button onclick="startCalc()">Lancer un calcul</button>
<div class="bar-container"><div class="bar" id="bar" style="width:0%">0%</div></div>
<p id="step">En attente...</p>
<div id="log"></div>
<div id="result" class="result" style="display:none"></div>

<script>
let eventSource = null;

function log(msg) {
  const div = document.getElementById('log');
  const line = document.createElement('div');
  line.className = 'log-line';
  line.textContent = new Date().toISOString().substr(11,12) + ' ' + msg;
  div.appendChild(line);
  div.scrollTop = div.scrollHeight;
}

async function startCalc() {
  if (eventSource) { eventSource.close(); }
  document.getElementById('result').style.display = 'none';
  document.getElementById('bar').style.width = '0%';
  document.getElementById('bar').textContent = '0%';

  log('POST /calculate-async...');
  const resp = await fetch('/calculate-async', {method: 'POST'});
  const data = await resp.json();
  const jobId = data.job_id;
  log('job_id = ' + jobId);

  eventSource = new EventSource('/progress/' + jobId);

  eventSource.onmessage = function(e) {
    const d = JSON.parse(e.data);
    log('[SSE] ' + d.step + ' ' + d.progress + '% - ' + d.message);

    document.getElementById('bar').style.width = d.progress + '%';
    document.getElementById('bar').textContent = d.progress + '%';
    document.getElementById('step').textContent = d.message;

    if (d.status === 'completed') {
      eventSource.close();
      log('DONE');
      if (d.result) {
        const r = d.result;
        document.getElementById('result').style.display = 'block';
        document.getElementById('result').innerHTML =
          '<b>Resultat:</b> ' + r.distance_km + ' km, D+ ' + r.dplus_m + 'm, ~' + r.time_h + 'h';
      }
    }
  };

  eventSource.onerror = function() {
    log('[SSE] connexion fermee');
    eventSource.close();
  };
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def test_page():
    return TEST_HTML


if __name__ == "__main__":
    import uvicorn
    print("SSE Progress Test")
    print("  -> http://127.0.0.1:8001")
    print("  -> Ouvrir dans un navigateur, cliquer 'Lancer un calcul'")
    uvicorn.run(app, host="127.0.0.1", port=8001)
