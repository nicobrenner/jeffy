"""Jeffy pretrained decision server.

Jeff-compatible API with pretrained capability catalog.

Run:
    python -m jeffy.server
    # or
    uvicorn jeffy.server:app --port 8400
"""

import json as _json_module
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import asyncio
import base64
import logging

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

from .engine import Engine

import os

app = FastAPI(
    title="Jeffy",
    description="Pretrained decision engine with reusable embeddings and tiny classifiers.",
    version="0.1.0a11",
)

_engine: Engine | None = None

# --- Visitor analytics ---

_analytics_log: list[dict] = []
_analytics_summary: dict = defaultdict(int)


class AnalyticsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - t0) * 1000, 1)

        ip = request.headers.get("cf-connecting-ip") or request.headers.get("x-forwarded-for", "").split(",")[0].strip() or request.client.host
        path = request.url.path
        ua = request.headers.get("user-agent", "")
        country = request.headers.get("cf-ipcountry", "")

        is_local = ip in ("127.0.0.1", "::1", "localhost")

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "ip": ip,
            "method": request.method,
            "path": path,
            "status": response.status_code,
            "duration_ms": duration_ms,
            "ua": ua[:200],
            "country": country,
            "local": is_local,
        }
        _analytics_log.append(entry)
        if len(_analytics_log) > 10000:
            _analytics_log.pop(0)

        _analytics_summary[f"{request.method} {path}"] += 1
        if not is_local:
            _analytics_summary["unique_ips"] = len({e["ip"] for e in _analytics_log if not e["local"]})
            _analytics_summary["external_requests"] += 1

        return response


app.add_middleware(AnalyticsMiddleware)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        device = os.environ.get("JEFFY_DEVICE", "cpu")
        _engine = Engine(device=device)  # uses default_pack_dir()
        _engine.load()
    return _engine


# --- Request/Response models ---

class PredictRequest(BaseModel):
    text: str | None = Field(None, description="Input text to classify")
    features: list[float] | None = Field(None, description="Input feature vector (for feature-based classifiers)")
    task: str = Field(..., description="Pretrained capability task ID")

class PredictResponse(BaseModel):
    label: str
    label_id: str
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    embedding_ms: float
    classifier_ms: float
    capability: dict

class BatchPredictRequest(BaseModel):
    texts: list[str]
    task: str

class CapabilityInfo(BaseModel):
    task_id: str
    name: str
    description: str
    n_classes: int
    task_type: str
    labels: dict[str, str]
    encoder: str
    train_examples: int
    train_accuracy: float | None
    test_accuracy: float | None
    license: str

class HealthResponse(BaseModel):
    status: str
    capabilities: int
    encoder: str


# --- Endpoints ---

@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ready",
        "capabilities": len(engine.capabilities),
        "encoder": engine.encoder_name,
        "encoder_load_time_s": round(engine.encoder_load_time, 2),
    }


@app.get("/v1/capabilities")
def list_capabilities():
    engine = get_engine()
    result = []
    for cap in engine.capabilities.values():
        result.append({
            "task_id": cap.task_id,
            "name": cap.name,
            "description": cap.description,
            "n_classes": cap.n_classes,
            "task_type": cap.task_type,
            "labels": cap.labels,
            "encoder": cap.encoder,
            "train_examples": cap.train_examples,
            "train_accuracy": cap.train_accuracy,
            "test_accuracy": cap.test_accuracy,
            "license": cap.license,
            **({"n_features": cap.n_features, "feature_layout": cap.feature_layout}
               if cap.is_feature_based else {}),
        })
    return {"capabilities": result, "count": len(result)}


@app.get("/v1/capabilities/{task_id}")
def get_capability(task_id: str):
    engine = get_engine()
    if task_id not in engine.capabilities:
        raise HTTPException(404, f"Capability '{task_id}' not found")
    cap = engine.capabilities[task_id]
    return {
        "task_id": cap.task_id,
        "name": cap.name,
        "description": cap.description,
        "n_classes": cap.n_classes,
        "task_type": cap.task_type,
        "labels": cap.labels,
        "encoder": cap.encoder,
        "train_examples": cap.train_examples,
        "train_accuracy": cap.train_accuracy,
        "test_accuracy": cap.test_accuracy,
        "license": cap.license,
        **({"n_features": cap.n_features, "feature_layout": cap.feature_layout}
           if cap.is_feature_based else {}),
        "example_request": (
            {"features": [0.0] * (cap.n_features or 1), "task": task_id}
            if cap.is_feature_based else
            {"text": _example_text(task_id), "task": task_id}
        ),
    }


@app.post("/v1/predict")
def predict(req: PredictRequest):
    engine = get_engine()
    if req.features is not None:
        result = engine.predict_features(req.task, req.features)
    elif req.text is not None:
        result = engine.predict(req.task, req.text)
    else:
        raise HTTPException(400, "Provide either 'text' or 'features'")
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@app.post("/v1/predict/batch")
def predict_batch(req: BatchPredictRequest):
    engine = get_engine()
    results = engine.predict_batch(req.task, req.texts)
    if results and "error" in results[0]:
        raise HTTPException(400, results[0]["error"])
    return {"predictions": results, "count": len(results)}


class SystemOneQuestion(BaseModel):
    type: str  # "choice" or "noul"
    instructions: str | None = None
    criteria: dict[str, str | None] | list[str | None] | None = None

class SystemOneRequest(BaseModel):
    """Jeff-compatible /v1/systemone request.

    Requires an explicit `capability` field to bind each question to a
    pretrained head. No ambiguous matching by label names or class count.

    The `label_map` field maps request option IDs (criteria keys) to
    the head's internal label names, preserving arbitrary caller IDs.
    """
    model: str = "jeffy"
    state: str | dict | list = ""
    questions: dict[str, SystemOneQuestion] = {}
    capability: str = Field(..., description="Pretrained task ID (e.g. 'banking77')")
    label_map: dict[str, str] | None = Field(
        None,
        description="Optional mapping from request criteria keys to head label names. "
                    "Required when criteria keys differ from the head's labels.",
    )

@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """Jeff-compatible decision endpoint with explicit capability binding.

    Requires the caller to specify which pretrained head handles the request.
    Rejects requests where no head is specified or the head doesn't exist.
    """
    engine = get_engine()

    # Validate capability exists
    if req.capability not in engine.capabilities:
        raise HTTPException(400,
            f"Capability '{req.capability}' not found. "
            f"Available: {sorted(engine.capabilities.keys())}")

    cap = engine.capabilities[req.capability]

    def _predict():
        if cap.is_feature_based:
            if not isinstance(req.state, list):
                raise HTTPException(400,
                    f"Capability '{req.capability}' is feature-based. "
                    f"'state' must be a list of {cap.n_features} numeric features.")
            return engine.predict_features(req.capability, req.state)
        state_text = req.state if isinstance(req.state, str) else _json_module.dumps(req.state)
        return engine.predict(req.capability, state_text)

    # Build reverse label map: head_label -> request_key
    head_to_request = {}
    if req.label_map:
        for request_key, head_label in req.label_map.items():
            if head_label not in cap.labels.values():
                raise HTTPException(400,
                    f"label_map value '{head_label}' not in head labels: "
                    f"{sorted(cap.labels.values())}")
            head_to_request[head_label] = request_key

    answers = {}
    for qname, question in req.questions.items():
        if question.type == "choice":
            if not isinstance(question.criteria, dict):
                answers[qname] = {
                    "type": "error",
                    "detail": "choice questions require criteria as a dict of options",
                }
                continue

            # Validate criteria match head labels (via label_map or directly)
            criteria_keys = set(question.criteria.keys())
            head_labels = set(cap.labels.values())

            if req.label_map:
                # All criteria keys must appear in label_map
                unmapped = criteria_keys - set(req.label_map.keys())
                if unmapped:
                    answers[qname] = {
                        "type": "error",
                        "detail": f"Criteria keys {sorted(unmapped)} not in label_map",
                    }
                    continue
            else:
                # Without label_map, criteria keys must exactly match head labels
                if criteria_keys != head_labels:
                    answers[qname] = {
                        "type": "error",
                        "detail": (
                            f"Criteria keys {sorted(criteria_keys)} don't match "
                            f"head labels {sorted(head_labels)}. "
                            "Provide a label_map to translate, or use all head labels."
                        ),
                    }
                    continue

            result = _predict()

            # Map probabilities to request option IDs
            probs = {}
            for head_label, prob in result["probabilities"].items():
                request_key = head_to_request.get(head_label, head_label)
                if request_key in criteria_keys:
                    probs[request_key] = prob

            # Normalize
            total = sum(probs.values()) or 1.0
            probs = {k: round(v / total, 4) for k, v in probs.items()}

            # Map predicted label to request key
            pred_request_key = head_to_request.get(result["label"], result["label"])

            answers[qname] = {
                "type": "choice",
                "choice": pred_request_key,
                "probabilities": probs,
                "confidence": result["confidence"],
                "capability": result["capability"],
            }

        elif question.type == "noul":
            if cap.n_classes != 2:
                answers[qname] = {
                    "type": "error",
                    "detail": f"noul requires a 2-class head; '{req.capability}' has {cap.n_classes} classes",
                }
                continue

            # Determine positive/negative from criteria or label_map
            result = _predict()
            all_labels = sorted(cap.labels.values())

            if question.criteria and isinstance(question.criteria, dict):
                true_label = question.criteria.get("true")
                false_label = question.criteria.get("false")
                if true_label and true_label in cap.labels.values():
                    pos_label = true_label
                elif req.label_map and "true" in req.label_map:
                    pos_label = req.label_map["true"]
                else:
                    answers[qname] = {
                        "type": "error",
                        "detail": (
                            "noul requires criteria.true to specify the positive class, "
                            f"matching one of {all_labels}. Or provide label_map."
                        ),
                    }
                    continue
            else:
                answers[qname] = {
                    "type": "error",
                    "detail": (
                        "noul requires criteria with 'true' and 'false' keys "
                        f"mapping to head labels {all_labels}"
                    ),
                }
                continue

            noul_prob = result["probabilities"].get(pos_label, 0.5)

            answers[qname] = {
                "type": "noul",
                "noul": round(noul_prob, 4),
                "capability": result["capability"],
            }

        else:
            answers[qname] = {
                "type": "error",
                "detail": f"Question type '{question.type}' not supported. Use 'choice' or 'noul'.",
            }

    return {
        "model": "jeffy",
        "answers": answers,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }

@app.websocket("/v1/doom/stream")
async def doom_stream(ws: WebSocket):
    await ws.accept()
    try:
        from .doom_runner import DoomSession
        engine = get_engine()
        session = DoomSession(engine=engine, task_id="doom_fire")
        session.start()
        target_fps = 20
        frame_interval = 1.0 / target_fps

        while True:
            t0 = time.perf_counter()
            result = session.tick()
            if result is None:
                break

            jpeg_bytes, meta = result
            await ws.send_text(_json_module.dumps(meta))
            await ws.send_bytes(jpeg_bytes)

            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Doom stream error: {e}")
    finally:
        if 'session' in locals():
            session.close()


@app.get("/v1/analytics")
def analytics(last: int = 50, external_only: bool = False):
    entries = _analytics_log
    if external_only:
        entries = [e for e in entries if not e["local"]]
    recent = entries[-last:]
    unique_external = len({e["ip"] for e in _analytics_log if not e["local"]})
    return {
        "total_requests": len(_analytics_log),
        "external_requests": _analytics_summary.get("external_requests", 0),
        "unique_external_ips": unique_external,
        "endpoints": {k: v for k, v in sorted(_analytics_summary.items()) if k not in ("unique_ips", "external_requests")},
        "recent": list(reversed(recent)),
    }


@app.get("/", response_class=HTMLResponse)
def playground():
    return PLAYGROUND_HTML


def _example_text(task_id: str) -> str:
    examples = {
        "banking77": "I've been charged twice for the same transaction, can I get a refund?",
        "ag_news": "The Federal Reserve raised interest rates by 0.25% today.",
        "dbpedia": "Harvard University is a private Ivy League research university in Cambridge, Massachusetts.",
        "sst2": "This movie was absolutely terrible, a waste of time.",
        "emotion": "I just got accepted into my dream school! I can't believe it!",
        "imdb": "A beautifully crafted film with stunning performances throughout.",
        "sms_spam": "WINNER!! You have been selected for a 900 prize reward! Call now!",
        "snli": "A man is playing guitar on a street corner. [SEP] A musician performs outdoors.",
        "tweet_eval_sentiment": "Best day ever! Finally got my dream job! #blessed",
        "tweet_eval_emotion": "I am so frustrated with this company's customer service.",
        "tweet_eval_offensive": "Great work on the project team, really proud of everyone.",
    }
    return examples.get(task_id, "Enter text here")


PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jeffy · Playground</title>
<style>
:root{color-scheme:light;--bg:#0c0c0c;--panel:#161616;--line:#2a2a2a;--text:#e8e8e8;--muted:#888;--accent:#ff4444;--accent2:#44ff44;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 var(--mono)}
header{max-width:1200px;margin:auto;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line)}
main{max-width:1200px;margin:auto;padding:20px 24px}
.brand a{font-size:20px;font-weight:700;letter-spacing:-0.5px;color:#fff;text-decoration:none}
.brand a:hover{color:var(--accent)}
.brand span{font-weight:400;color:var(--muted);font-size:13px;margin-left:10px}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--line);margin-bottom:20px}
.tab{padding:10px 20px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;font:13px var(--mono);transition:color .15s}
.tab:hover{color:var(--text)}
.tab.active{color:#fff;border-bottom-color:var(--accent)}
.tab-content{display:none}
.tab-content.active{display:block}
.status{font-size:11px;color:var(--muted)}.status::before{content:"●";color:#555;margin-right:5px}
.status[data-state="ready"]::before{color:#22c55e}
.doom-layout{display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:start}
.doom-left{display:flex;flex-direction:column;gap:12px;width:480px}
.game-container{position:relative;background:#000;border-radius:8px;overflow:hidden;aspect-ratio:4/3}
.game-container img{width:100%;height:100%;object-fit:contain;display:block}
.hud{position:absolute;bottom:0;left:0;right:0;padding:10px 14px;background:linear-gradient(transparent,rgba(0,0,0,.85));display:flex;justify-content:space-between;align-items:flex-end}
.hud-stat{text-align:center;font-size:11px;color:#ccc;line-height:1.3}
.hud-stat .val{font-size:22px;font-weight:700;color:#fff;display:block}
.decision-overlay{position:absolute;top:12px;left:50%;transform:translateX(-50%);padding:6px 16px;border-radius:6px;font-size:15px;font-weight:700;letter-spacing:1px;text-transform:uppercase;pointer-events:none;transition:all .1s}
.decision-overlay.fire{background:rgba(255,60,60,.9);color:#fff}
.decision-overlay.turn_left,.decision-overlay.turn_right{background:rgba(40,40,40,.8);color:#aaa}
.sidebar{display:flex;flex-direction:column;gap:16px}
.info-card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.info-card h3{margin:0 0 10px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.stat-row{display:flex;justify-content:space-between;font-size:12px;padding:3px 0;border-bottom:1px solid var(--line)}
.stat-row:last-child{border-bottom:0}
.stat-row .label{color:var(--muted)}
.prob-bar{height:6px;border-radius:3px;background:#222;margin-top:6px;overflow:hidden}
.prob-fill{height:100%;border-radius:3px;transition:width .15s}
.prob-fill.fire{background:var(--accent)}
.prob-fill.hold{background:var(--accent2)}
.restart-btn{padding:6px 14px;border:1px solid var(--line);background:transparent;color:var(--muted);border-radius:6px;font:11px var(--mono);cursor:pointer}
.restart-btn:hover{border-color:var(--accent);color:var(--text)}
.log{max-height:180px;overflow-y:auto;font-size:10px;color:var(--muted);line-height:1.6;padding:0;margin:0;list-style:none}
.log li.fire{color:var(--accent)}
.log li.hold{color:var(--accent2)}
.text-layout{display:grid;grid-template-columns:1.05fr 1fr;gap:20px}
.panel{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--panel)}
.panel-head{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}
.panel-body{padding:16px}
h2{font-size:14px;margin:0;color:#fff}
label{display:block;color:var(--muted);font-size:11px;margin-bottom:5px}
textarea,select{width:100%;background:#1a1a1a;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:10px;font:12px/1.5 var(--mono)}
textarea{resize:vertical;min-height:100px}
textarea:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px}
select{padding:8px}
.field{margin-top:14px}
button{border:1px solid var(--line);background:transparent;color:var(--text);border-radius:6px;padding:7px 12px;cursor:pointer;font:11px var(--mono)}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.5;cursor:wait}
.run{background:var(--accent);color:#fff;border:0;font-weight:650;padding:10px 18px}
.actions{display:flex;align-items:center;justify-content:flex-end;margin-top:14px;gap:10px}
.result-value{font-size:22px;font-weight:650;letter-spacing:-0.5px;margin:6px 0 10px;color:#fff}
.bar-row{display:grid;grid-template-columns:minmax(80px,1.2fr) 2fr 50px;gap:8px;align-items:center;font-size:11px;margin:6px 0}
.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}
.track{height:7px;background:#222;border-radius:6px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:6px}
.percentage{text-align:right;color:var(--muted)}
.badge{font-size:10px;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:2px 6px}
.meta{font-size:10px;color:var(--muted);margin-top:10px;line-height:1.7}
.empty{padding:40px 16px;text-align:center;color:var(--muted);font-size:13px}
.error{color:#ff6b6b;background:#1a0000;border:1px solid #4a0000;border-radius:6px;padding:10px;font-size:11px}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}
summary{color:var(--muted);font-size:11px;cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:10px/1.6 var(--mono);max-height:300px;overflow:auto;color:var(--muted)}
.hidden{display:none!important}
footer{max-width:1200px;margin:auto;padding:16px 24px;font-size:10px;color:#555;display:flex;justify-content:space-between}
a{color:var(--muted)}
@media(max-width:800px){.doom-layout,.text-layout{grid-template-columns:1fr}.doom-left{width:auto}}
</style>
</head>
<body>
<header>
<div class="brand"><a href="https://github.com/nicobrenner/jeffy" target="_blank">Jeffy</a><span>Playground</span></div>
<div style="display:flex;align-items:center;gap:14px">
<a href="https://github.com/nicobrenner/jeffy" target="_blank" style="font-size:11px;color:var(--muted)">GitHub ↗</a>
<div id="status" class="status">Connecting</div>
</div>
</header>
<main>
<div class="tabs">
<div class="tab active" data-tab="doom">Doom Demo</div>
<div class="tab" data-tab="text">Text Classifier</div>
</div>
<div id="tab-doom" class="tab-content active">
<div class="doom-layout">
<div class="doom-left">
<div class="game-container" id="game-container">
<img id="game-frame" src="" alt="Doom frame">
<div id="decision-overlay" class="decision-overlay hold">HOLD</div>
<div class="hud">
<div class="hud-stat"><span class="val" id="hud-health">100</span>HP</div>
<div class="hud-stat"><span class="val" id="hud-ammo">26</span>Ammo</div>
<div class="hud-stat"><span class="val" id="hud-kills">0</span>Kills</div>
<div class="hud-stat"><span class="val" id="hud-enemies">0</span>Enemies</div>
<div class="hud-stat"><span class="val" id="hud-episode">1</span>Episode</div>
</div>
</div>
<div class="info-card">
<div style="display:flex;justify-content:space-between;align-items:center">
<h3 style="margin:0">Decision Log</h3>
<div style="display:flex;gap:8px;align-items:center">
<span id="stream-status" style="font-size:10px;color:var(--muted)"></span>
<button id="restart-btn" class="restart-btn">Restart</button>
</div>
</div>
<ul id="decision-log" class="log" style="margin-top:8px"></ul>
</div>
</div>
<div class="sidebar">
<div class="info-card">
<h3>Classifier Decision</h3>
<div style="display:flex;justify-content:space-between;align-items:baseline">
<span id="decision-label" style="font-size:20px;font-weight:700;color:var(--accent)">—</span>
<span id="decision-conf" style="font-size:13px;color:var(--muted)">—</span>
</div>
<div id="prob-bars" style="margin-top:8px"></div>
</div>
<div class="info-card">
<h3>Stats</h3>
<div class="stat-row"><span class="label">Classify latency</span><span id="stat-ms">—</span></div>
<div class="stat-row"><span class="label">Total kills</span><span id="stat-total-kills">0</span></div>
<div class="stat-row"><span class="label">Step</span><span id="stat-step">0</span></div>
<div class="stat-row"><span class="label">FPS</span><span id="stat-fps">—</span></div>
</div>
<div class="info-card">
<h3>How it works</h3>
<div style="font-size:11px;color:var(--muted);line-height:1.6">
Jeffy extracts 24 game-state features (enemy positions, health, ammo, action history) and runs them through a logistic regression classifier to decide: <span style="color:var(--accent)">FIRE</span>, <span style="color:#44aaff">TURN LEFT</span>, or <span style="color:#ffaa44">TURN RIGHT</span>. No neural network, no GPU.
</div>
</div>
</div>
</div>
</div>
<div id="tab-text" class="tab-content">
<div class="text-layout">
<section class="panel">
<div class="panel-head"><h2>Input</h2></div>
<form id="form" class="panel-body">
<label for="task">Pretrained capability</label>
<select id="task"></select>
<div id="task-info" class="meta"></div>
<div class="field">
<label for="text">Text</label>
<textarea id="text" rows="5" spellcheck="false"></textarea>
</div>
<div class="actions">
<span id="latency" style="font-size:11px;color:var(--muted)"></span>
<button class="run" id="run" type="submit">Classify →</button>
</div>
<div id="curl-section" style="margin-top:14px;border-top:1px solid var(--line);padding-top:10px">
<div style="display:flex;align-items:center;justify-content:space-between">
<span style="color:var(--muted);font-size:11px">or use curl</span>
<button id="copy-curl" style="font-size:10px;padding:4px 8px">Copy</button>
</div>
<pre id="curl-cmd" style="background:#1a1a1a;border-radius:5px;padding:8px;margin-top:5px;font-size:10px"></pre>
</div>
</form>
</section>
<section class="panel">
<div class="panel-head"><h2>Result</h2><span id="result-badge" class="badge hidden"></span></div>
<div class="panel-body">
<div id="empty" class="empty">Select a capability and enter text to classify.</div>
<div id="error" class="error hidden"></div>
<div id="result" class="hidden">
<div id="result-label" class="result-value"></div>
<div id="bars"></div>
<div id="meta" class="meta"></div>
</div>
<details id="raw-section" class="hidden">
<summary>Response JSON</summary>
<pre id="raw"></pre>
</details>
</div>
</section>
</div>
</div>
</main>
<footer>
<span><a href="/docs" target="_blank">API docs</a></span>
<span><a href="https://github.com/nicobrenner/jeffy" target="_blank">GitHub</a></span>
</footer>
<script>
const $=id=>document.getElementById(id);
let caps={},ws=null,frameCount=0,fpsStart=0;

// Tabs
document.querySelectorAll(".tab").forEach(t=>{
  t.onclick=()=>{
    document.querySelectorAll(".tab").forEach(x=>x.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(x=>x.classList.remove("active"));
    t.classList.add("active");
    $("tab-"+t.dataset.tab).classList.add("active");
  };
});

// Health check
fetch("/health").then(r=>r.json()).then(d=>{
  $("status").dataset.state=d.status;
  $("status").textContent=d.capabilities+" capabilities ready";
}).catch(()=>{$("status").textContent="Server unavailable"});

// --- Doom Stream ---
function startDoom(){
  if(ws&&ws.readyState===WebSocket.OPEN)ws.close();
  const proto=location.protocol==="https:"?"wss:":"ws:";
  ws=new WebSocket(proto+"//"+location.host+"/v1/doom/stream");
  $("stream-status").textContent="Connecting...";
  $("decision-log").innerHTML="";
  frameCount=0;fpsStart=performance.now();

  ws.onopen=()=>{$("stream-status").textContent="Live";};
  ws.binaryType="blob";
  let pendingMeta=null;
  ws.onmessage=e=>{
    if(typeof e.data==="string"){
      pendingMeta=JSON.parse(e.data);
      return;
    }
    if(!pendingMeta)return;
    const d=pendingMeta;pendingMeta=null;
    const url=URL.createObjectURL(e.data);
    const img=$("game-frame");
    const old=img.src;
    img.src=url;
    if(old.startsWith("blob:"))URL.revokeObjectURL(old);
    const ov=$("decision-overlay");
    ov.textContent=d.decision.toUpperCase();
    ov.className="decision-overlay "+d.decision;
    $("hud-health").textContent=d.health;
    $("hud-ammo").textContent=d.ammo;
    $("hud-kills").textContent=d.kills;
    $("hud-enemies").textContent=d.enemies_visible;
    $("hud-episode").textContent=d.episode;
    const dLabel=d.decision.replace("_"," ").toUpperCase();
    $("decision-label").textContent=dLabel;
    $("decision-label").style.color=d.decision==="fire"?"var(--accent)":"var(--accent2)";
    $("decision-conf").textContent=(d.confidence*100).toFixed(1)+"%";
    const pb=$("prob-bars");pb.innerHTML="";
    const colors={fire:"#ff4444",turn_left:"#44aaff",turn_right:"#ffaa44"};
    Object.entries(d.probabilities).forEach(([lbl,p])=>{
      const pct=Math.round(p*100);
      pb.innerHTML+=`<div style="display:flex;align-items:center;gap:6px;font-size:10px;margin:3px 0;color:var(--muted)"><span style="width:70px;text-align:right">${lbl.replace("_"," ")}</span><div style="flex:1;height:5px;background:#222;border-radius:3px;overflow:hidden"><div style="width:${pct}%;height:100%;background:${colors[lbl]||'#888'};border-radius:3px"></div></div><span style="width:30px">${pct}%</span></div>`;
    });
    $("stat-ms").textContent=d.classify_ms+"ms";
    $("stat-total-kills").textContent=d.total_kills;
    $("stat-step").textContent=d.step;
    frameCount++;
    const elapsed=(performance.now()-fpsStart)/1000;
    if(elapsed>1){$("stat-fps").textContent=Math.round(frameCount/elapsed);frameCount=0;fpsStart=performance.now();}
    const log=$("decision-log");
    const li=document.createElement("li");
    li.className=d.decision;
    li.textContent="["+d.step+"] "+d.decision.toUpperCase()+" "+(d.confidence*100).toFixed(0)+"% | "+d.enemies_visible+" enemies | "+d.action;
    log.prepend(li);
    while(log.children.length>50)log.lastChild.remove();
  };
  ws.onclose=()=>{
    $("stream-status").textContent="Disconnected";
    ws=null;
  };
  ws.onerror=()=>{$("stream-status").textContent="Connection error";};
}
$("restart-btn").onclick=()=>startDoom();
startDoom();

// --- Text Classifier ---
fetch("/v1/capabilities").then(r=>r.json()).then(d=>{
  const sel=$("task");
  d.capabilities.filter(c=>c.encoder!=="features").forEach(c=>{
    caps[c.task_id]=c;
    const o=document.createElement("option");
    o.value=c.task_id;
    o.textContent=c.task_id+" — "+c.name;
    sel.appendChild(o);
  });
  if(d.capabilities.length){updateTaskInfo()}
});

$("task").onchange=()=>{updateTaskInfo()};
$("text").oninput=updateCurl;

function updateCurl(){
  const payload=JSON.stringify({text:$("text").value||"your text here",task:$("task").value});
  const escaped=payload.replace(/'/g,"'\\''");
  $("curl-cmd").textContent="curl -s -X POST "+location.origin+"/v1/predict \\\n  -H 'Content-Type: application/json' \\\n  -d '"+escaped+"' | python3 -m json.tool";
}

function updateTaskInfo(){
  const c=caps[$("task").value];
  if(!c)return;
  $("task-info").innerHTML=
    c.n_classes+" classes · "+c.train_examples+" training examples"+
    (c.train_accuracy?" · train acc "+Math.round(c.train_accuracy*1000)/10+"%":"")+
    "<br>Encoder: "+c.encoder+"<br>License: "+c.license;
  fetch("/v1/capabilities/"+c.task_id).then(r=>r.json()).then(d=>{
    if(d.example_request&&d.example_request.text)$("text").value=d.example_request.text;
    updateCurl();
  });
}

$("form").onsubmit=async e=>{
  e.preventDefault();
  $("run").disabled=true;$("run").textContent="Classifying…";
  $("error").classList.add("hidden");$("empty").classList.add("hidden");
  $("result").classList.add("hidden");$("raw-section").classList.add("hidden");
  const t0=performance.now();
  try{
    const r=await fetch("/v1/predict",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:$("text").value,task:$("task").value})});
    const d=await r.json();
    if(!r.ok)throw Error(d.detail||JSON.stringify(d));
    $("result-label").textContent=d.label;
    $("result-badge").textContent=d.capability.task_id;
    $("result-badge").classList.remove("hidden");
    const bars=$("bars");bars.innerHTML="";
    Object.entries(d.probabilities).forEach(([label,prob])=>{
      const row=document.createElement("div");row.className="bar-row";
      row.innerHTML='<span class="bar-label" title="'+label+'">'+label+'</span>'+
        '<div class="track"><div class="fill" style="width:'+Math.round(prob*100)+'%"></div></div>'+
        '<span class="percentage">'+(prob*100).toFixed(1)+'%</span>';
      bars.appendChild(row);
    });
    $("meta").innerHTML="Confidence: "+(d.confidence*100).toFixed(1)+"%"+
      " · Embedding: "+d.embedding_ms+"ms · Classifier: "+d.classifier_ms+"ms"+
      " · Total: "+d.latency_ms+"ms"+
      "<br>Encoder: "+d.capability.encoder+
      " · Head: "+d.capability.n_classes+" classes, "+d.capability.train_examples+" train examples";
    $("result").classList.remove("hidden");
    $("raw").textContent=JSON.stringify(d,null,2);
    $("raw-section").classList.remove("hidden");
    $("latency").textContent=Math.round(performance.now()-t0)+"ms round-trip";
  }catch(err){
    $("error").textContent=err.message;$("error").classList.remove("hidden");
  }finally{
    $("run").disabled=false;$("run").textContent="Classify →";
  }
};
$("copy-curl").onclick=()=>{
  const cmd=$("curl-cmd").textContent;
  navigator.clipboard.writeText(cmd).then(()=>{
    $("copy-curl").textContent="Copied!";
    setTimeout(()=>$("copy-curl").textContent="Copy",1500);
  }).catch(()=>{
    const s=window.getSelection(),r=document.createRange();
    r.selectNodeContents($("curl-cmd"));s.removeAllRanges();s.addRange(r);
  });
};
</script>
</body>
</html>"""


def main():
    import uvicorn
    port = int(os.environ.get("JEFFY_PORT", "8400"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
