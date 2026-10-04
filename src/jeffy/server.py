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
    version="0.1.0a12",
)

_engine: Engine | None = None

# --- Visitor analytics (persisted to disk) ---

_ANALYTICS_FILE = Path(os.environ.get("JEFFY_ANALYTICS_FILE", "~/.jeffy-analytics.jsonl")).expanduser()
_analytics_log: list[dict] = []
_analytics_summary: dict = defaultdict(int)


def _load_analytics():
    if not _ANALYTICS_FILE.exists():
        return
    try:
        for line in _ANALYTICS_FILE.read_text().splitlines():
            if line.strip():
                entry = _json_module.loads(line)
                _analytics_log.append(entry)
                _analytics_summary[f"{entry['method']} {entry['path']}"] += 1
                if not entry.get("local", False):
                    _analytics_summary["external_requests"] += 1
        _analytics_summary["unique_ips"] = len({e["ip"] for e in _analytics_log if not e.get("local", False)})
    except Exception:
        pass

_load_analytics()


def _append_analytics(entry: dict):
    try:
        with _ANALYTICS_FILE.open("a") as f:
            f.write(_json_module.dumps(entry) + "\n")
    except Exception:
        pass


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
        _append_analytics(entry)
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


@app.websocket("/v1/poker/stream")
async def poker_stream(ws: WebSocket):
    await ws.accept()
    try:
        from .poker_runner import PokerSession
        engine = get_engine()
        session = PokerSession(engine=engine, task_id="poker_decision")
        session.start()
        tick_interval = 0.15

        while True:
            t0 = time.perf_counter()
            result = session.tick()
            if result is None:
                break

            await ws.send_text(_json_module.dumps(result))

            elapsed = time.perf_counter() - t0
            sleep_time = tick_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning(f"Poker stream error: {e}")
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
        "inbox_router": "Can you send me the Q4 projections?",
    }
    return examples.get(task_id, "Enter text here")


PLAYGROUND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jeffy · Classifier Catalog</title>
<style>
:root{color-scheme:light;--bg:#0c0c0c;--panel:#161616;--line:#2a2a2a;--text:#e8e8e8;--muted:#888;--accent:#ff4444;--accent2:#44ff44;--blue:#44aaff;--orange:#ffaa44;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.5 var(--mono)}
header{max-width:1200px;margin:auto;padding:16px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line)}
main{max-width:1200px;margin:auto;padding:20px 24px}
a{color:var(--muted)}
.brand a{font-size:20px;font-weight:700;letter-spacing:-0.5px;color:#fff;text-decoration:none}
.brand a:hover{color:var(--accent)}
.brand span{font-weight:400;color:var(--muted);font-size:13px;margin-left:10px}
.status{font-size:11px;color:var(--muted)}.status::before{content:"\\25cf";color:#555;margin-right:5px}
.status[data-state="ready"]::before{color:#22c55e}
.view{display:none}.view.active{display:block}

.catalog-head{margin-bottom:20px}
.catalog-head h2{font-size:15px;color:#fff;margin:0 0 4px}
.catalog-head p{margin:0;font-size:12px;color:var(--muted)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px;cursor:pointer;transition:border-color .15s}
.card:hover{border-color:var(--accent)}
.card.feat{border-left:3px solid var(--orange)}
.card-top{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px}
.card-id{font-size:13px;font-weight:700;color:#fff}
.card-acc{font-size:12px;font-weight:600;color:var(--accent2)}
.card-name{font-size:11px;color:var(--muted);margin-bottom:10px;line-height:1.4}
.card-tags{display:flex;gap:6px;flex-wrap:wrap}
.tag{font-size:10px;color:var(--muted);border:1px solid var(--line);border-radius:4px;padding:1px 6px}
.card-bar{height:4px;border-radius:2px;background:#222;margin-top:10px;overflow:hidden}
.card-bar-fill{height:100%;border-radius:2px;background:var(--accent2)}
.demo-badge{font-size:9px;color:var(--accent);border:1px solid var(--accent);border-radius:3px;padding:1px 5px;text-transform:uppercase;letter-spacing:.5px}

.back{font-size:12px;color:var(--muted);cursor:pointer;margin-bottom:16px;display:inline-block}
.back:hover{color:var(--text)}
.d-header{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:20px;margin-bottom:16px}
.d-title{font-size:18px;font-weight:700;color:#fff;margin:0 0 4px}
.d-desc{font-size:12px;color:var(--muted);margin:0 0 14px}
.d-stats{display:flex;gap:20px;flex-wrap:wrap}
.d-stat{text-align:center}
.d-stat .v{font-size:20px;font-weight:700;color:#fff;display:block}
.d-stat .l{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.d-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.d-section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:16px}
.d-section h3{margin:0 0 10px;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:1px}
.d-section.full{grid-column:1/-1}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:10px;color:var(--text);background:#222;border-radius:4px;padding:2px 8px}
textarea{width:100%;background:#1a1a1a;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:10px;font:12px/1.5 var(--mono);resize:vertical;min-height:80px}
textarea:focus{outline:2px solid var(--accent);outline-offset:1px}
button{border:1px solid var(--line);background:transparent;color:var(--text);border-radius:6px;padding:7px 12px;cursor:pointer;font:11px var(--mono)}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.5;cursor:wait}
.run-btn{background:var(--accent);color:#fff;border:0;font-weight:650;padding:10px 18px;margin-top:10px}
.try-row{display:flex;align-items:center;justify-content:space-between;margin-top:10px}
.result-label{font-size:22px;font-weight:650;letter-spacing:-.5px;margin:10px 0;color:#fff}
.bar-row{display:grid;grid-template-columns:minmax(80px,1.2fr) 2fr 50px;gap:8px;align-items:center;font-size:11px;margin:6px 0}
.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--muted)}
.track{height:7px;background:#222;border-radius:6px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:6px}
.pct{text-align:right;color:var(--muted)}
.rmeta{font-size:10px;color:var(--muted);margin-top:10px;line-height:1.7}
.err-msg{color:#ff6b6b;background:#1a0000;border:1px solid #4a0000;border-radius:6px;padding:10px;font-size:11px;margin-top:10px;display:none}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:10px/1.6 var(--mono);max-height:300px;overflow:auto;color:var(--muted);background:#1a1a1a;border-radius:5px;padding:10px;margin:8px 0 0}
.codeblk{position:relative}
.cpbtn{position:absolute;top:6px;right:6px;font-size:10px;padding:3px 8px}
details{margin-top:14px;border-top:1px solid var(--line);padding-top:10px}
summary{color:var(--muted);font-size:11px;cursor:pointer}

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
.restart-btn{padding:6px 14px;border:1px solid var(--line);background:transparent;color:var(--muted);border-radius:6px;font:11px var(--mono);cursor:pointer}
.restart-btn:hover{border-color:var(--accent);color:var(--text)}
.log{max-height:180px;overflow-y:auto;font-size:10px;color:var(--muted);line-height:1.6;padding:0;margin:0;list-style:none}
.log li.fire{color:var(--accent)}
.log li.turn_left{color:var(--blue)}
.log li.turn_right{color:var(--orange)}
footer{max-width:1200px;margin:auto;padding:16px 24px;font-size:10px;color:#555;display:flex;justify-content:space-between;gap:12px}

.ib-layout{display:grid;grid-template-columns:1fr 1fr;gap:14px;height:460px}
.ib-panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;display:flex;flex-direction:column;overflow:hidden;transition:opacity .35s}
.ib-panel[hidden]{display:none!important}
#ib-trainPanel,#ib-inboxPanel{grid-column:1;grid-row:1}
#ib-logPanel{grid-column:2;grid-row:1}
.ib-panel-head{display:flex;align-items:center;justify-content:space-between;padding:11px 14px;border-bottom:1px solid var(--line);flex-shrink:0}
.ib-panel-head h3{font-size:13px;font-weight:600;margin:0;color:#fff}
.ib-panel-head .meta{font-size:10px;color:var(--muted);font-variant-numeric:tabular-nums}
.ib-panel-body{flex:1;overflow-y:auto;padding:10px 12px;display:flex;flex-direction:column;gap:10px}
.ib-train-row{display:flex;align-items:center;gap:8px;padding:6px 8px;font-size:12px;line-height:1.4;border-bottom:1px solid rgba(255,255,255,.04)}
.ib-train-text{flex:1;min-width:0;color:var(--text)}
.ib-train-arrow{color:var(--muted);font-size:10px;flex-shrink:0}
.ib-train-label{font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;padding:2px 7px;border-radius:3px;flex-shrink:0}
.ib-train-label.work{background:#1e3358;color:#6ba3d8}
.ib-train-label.family{background:#3a2218;color:#d47e5c}
.ib-train-label.promo{background:#3a2e10;color:#cda24a}
.ib-train-label.notif{background:#1a3528;color:#5eaa7e}
.ib-train-more{padding:8px;font-size:11px;color:var(--muted);text-align:center}
.ib-section-label{font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:0 0 5px;display:flex;align-items:center;gap:5px}
.ib-cat-dot{width:6px;height:6px;border-radius:50%;display:inline-block}
.ib-cat-dot.work{background:#6ba3d8}.ib-cat-dot.family{background:#d47e5c}.ib-cat-dot.promo{background:#cda24a}.ib-cat-dot.notif{background:#5eaa7e}
.ib-cat-n{font-variant-numeric:tabular-nums;opacity:.5}
.ib-categories{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.ib-bucket{display:flex;flex-direction:column;gap:4px;min-height:2px}
.ib-incoming{display:flex;flex-direction:column;gap:4px}
.ib-msg{padding:6px 10px;border-radius:5px;font-size:11px;line-height:1.4;border:1px solid var(--line);background:var(--panel);color:var(--text);display:flex;align-items:center;gap:7px}
.ib-msg.pending{visibility:hidden}.ib-msg.show{visibility:visible}
.ib-msg-text{flex:1;min-width:0}
.ib-msg-lat{font:9px var(--mono);color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap;flex-shrink:0}
.ib-msg.sorted{border-color:transparent}
.ib-msg.sorted.work{background:#1e3358}.ib-msg.sorted.family{background:#3a2218}.ib-msg.sorted.promo{background:#3a2e10}.ib-msg.sorted.notif{background:#1a3528}
.ib-msg.leaving{opacity:0;transform:translateX(-14px) scale(.97);transition:opacity .2s,transform .2s}
.ib-msg.flash{transition:box-shadow .1s}.ib-msg.flash-fade{transition:box-shadow .35s}
.ib-typing{display:inline-flex;gap:3px;padding:2px 3px;flex-shrink:0}
.ib-typing span{width:3px;height:3px;border-radius:50%;background:var(--muted);animation:ib-bounce 1.4s infinite ease-in-out}
.ib-typing span:nth-child(2){animation-delay:.16s}
.ib-typing span:nth-child(3){animation-delay:.32s}
@keyframes ib-bounce{0%,60%,100%{opacity:.15;transform:translateY(0)}30%{opacity:.8;transform:translateY(-3px)}}
.ib-ghost{position:absolute;z-index:15;box-shadow:0 2px 8px rgba(0,0,0,.3)}
.ib-signal{position:absolute;height:2px;z-index:20;pointer-events:none;background:var(--lc)}
.ib-signal::after{content:"";position:absolute;top:-4px;border-top:5px solid transparent;border-bottom:5px solid transparent}
.ib-signal.to-right::after{right:-8px;border-left:8px solid var(--lc)}
.ib-signal.to-left::after{left:-8px;border-right:8px solid var(--lc)}
.ib-log-chrome{display:flex;align-items:center;gap:6px;padding:9px 12px;border-bottom:1px solid rgba(255,255,255,.05);flex-shrink:0}
.ib-dots{display:flex;gap:5px}.ib-dots span{width:8px;height:8px;border-radius:50%}
.ib-dots span:nth-child(1){background:#E5534B}.ib-dots span:nth-child(2){background:#D4A03C}.ib-dots span:nth-child(3){background:#4CAF6A}
.ib-log-chrome .title{font:10px var(--mono);color:#444;margin-left:3px}
.ib-log-body{flex:1;overflow-y:auto;padding:8px 12px;font:10px/1.7 var(--mono)}
.ib-log-body::-webkit-scrollbar{width:3px}.ib-log-body::-webkit-scrollbar-thumb{background:#333;border-radius:2px}
.ib-log-line{color:var(--muted);white-space:pre-wrap;word-break:break-all;opacity:0;transform:translateY(-5px);animation:ib-fadeDown .18s forwards}
@keyframes ib-fadeDown{to{opacity:1;transform:translateY(0)}}
.ib-log-line.spacer{height:5px;animation:none}
.ib-log-line .method{color:#6BA3C7;font-weight:500}
.ib-log-line .bright{color:#CDD1DC}
.ib-log-line .ok{color:#5FB882;font-weight:500}
.ib-log-line .dim{color:#3E4250}
.ib-log-line .lbl-work{color:#6ba3d8}.ib-log-line .lbl-family{color:#d47e5c}.ib-log-line .lbl-promo{color:#cda24a}.ib-log-line .lbl-notif{color:#5eaa7e}

.pk-layout{display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:start}
.pk-table{position:relative;width:520px;height:380px;background:radial-gradient(ellipse at center,#1a5c2a 0%,#0e3d1a 70%,#0a2e13 100%);border-radius:180px;border:8px solid #2a1a0a;box-shadow:0 0 40px rgba(0,0,0,.5),inset 0 0 60px rgba(0,0,0,.3)}
.pk-pot{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);text-align:center;color:#dda;font-size:11px;font-weight:600}
.pk-pot .amt{font-size:20px;color:#ffd700;display:block;text-shadow:0 1px 3px rgba(0,0,0,.5)}
.pk-community{position:absolute;top:38%;left:50%;transform:translate(-50%,-50%);display:flex;gap:4px}
.pk-card{width:38px;height:54px;background:#fff;border-radius:4px;display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:700;box-shadow:0 1px 4px rgba(0,0,0,.3);color:#1a1a1a;line-height:1}
.pk-card.red{color:#cc2222}
.pk-card.back{background:linear-gradient(135deg,#1a3a8a,#2a4aaa);color:transparent}
.pk-card.empty{background:rgba(255,255,255,.08);box-shadow:none;border:1px dashed rgba(255,255,255,.15)}
.pk-seat{position:absolute;display:flex;flex-direction:column;align-items:center;gap:3px;width:110px}
.pk-seat[data-pos="0"]{top:-20px;left:50%;transform:translateX(-50%)}
.pk-seat[data-pos="1"]{right:-30px;top:50%;transform:translateY(-50%)}
.pk-seat[data-pos="2"]{bottom:-20px;left:50%;transform:translateX(-50%)}
.pk-seat[data-pos="3"]{left:-30px;top:50%;transform:translateY(-50%)}
.pk-name{font-size:10px;font-weight:600;color:#ccc;background:rgba(0,0,0,.5);padding:1px 8px;border-radius:3px}
.pk-name .chips{font-weight:400;color:#aaa;margin-left:4px}
.pk-hole{display:flex;gap:2px}
.pk-hole .pk-card{width:32px;height:45px;font-size:11px}
.pk-action-badge{font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;padding:2px 8px;border-radius:3px;background:rgba(0,0,0,.6);color:var(--muted);min-height:16px}
.pk-action-badge.fold{color:#888}.pk-action-badge.check{color:#6baa6b}.pk-action-badge.call{color:#6ba3d8}.pk-action-badge.raise{color:#d4a04a}.pk-action-badge.all_in{color:#ff4444}
.pk-seat.folded{opacity:.4}
.pk-seat.active-turn .pk-name{box-shadow:0 0 8px rgba(255,215,0,.6)}
.pk-dealer-chip{position:absolute;width:16px;height:16px;border-radius:50%;background:#fff;color:#000;font-size:8px;font-weight:800;display:flex;align-items:center;justify-content:center;box-shadow:0 1px 3px rgba(0,0,0,.4)}
.pk-sidebar{display:flex;flex-direction:column;gap:14px}
.pk-hand-label{font-size:11px;color:var(--muted);margin-top:2px}
.pk-decision-row{display:flex;align-items:center;gap:8px;padding:4px 0;font-size:11px;border-bottom:1px solid var(--line)}
.pk-decision-row .pname{width:45px;color:var(--muted)}.pk-decision-row .pact{font-weight:600}
.pk-decision-row .pact.fold{color:#888}.pk-decision-row .pact.check{color:#6baa6b}.pk-decision-row .pact.call{color:#6ba3d8}.pk-decision-row .pact.raise{color:#d4a04a}
.pk-history-row{font-size:10px;color:var(--muted);padding:2px 0;border-bottom:1px solid rgba(255,255,255,.03)}
.pk-round-badge{font-size:9px;text-transform:uppercase;letter-spacing:.5px;padding:1px 6px;border-radius:3px;background:rgba(255,255,255,.06);color:var(--muted);display:inline-block}

@media(max-width:800px){.grid{grid-template-columns:1fr}.d-grid{grid-template-columns:1fr}.doom-layout{grid-template-columns:1fr}.doom-left{width:auto}.ib-layout{grid-template-columns:1fr;height:auto;min-height:70vh}.ib-categories{grid-template-columns:1fr}.ib-signal,.ib-ghost{display:none!important}.pk-layout{grid-template-columns:1fr}.pk-table{width:100%;height:300px}}
</style>
</head>
<body>
<header>
<div class="brand"><a href="https://jeffyclassify.com" target="_blank">Jeffy</a><span>Classifier Catalog</span></div>
<div style="display:flex;align-items:center;gap:14px">
<a href="https://github.com/nicobrenner/jeffy" target="_blank" style="font-size:11px;color:var(--muted)">GitHub &#8599;</a>
<div id="status" class="status">Connecting</div>
</div>
</header>
<main>

<!-- Catalog -->
<div id="v-catalog" class="view active">
<div class="catalog-head">
<h2>Pretrained Classifiers</h2>
<p id="grid-count">16 classifiers ready to use. Click a model to try it.</p>
</div>
<div id="grid" class="grid"></div>
</div>

<!-- Model Detail -->
<div id="v-detail" class="view">
<span class="back" onclick="go('')">&#8592; Back to catalog</span>
<div id="dh" class="d-header"></div>
<div class="d-grid">
<div class="d-section"><h3>Labels</h3><div id="d-labels" class="chips"></div></div>
<div class="d-section"><h3>Info</h3><div id="d-info"></div></div>
<div class="d-section"><h3>Try it</h3><div id="d-try"></div></div>
<div class="d-section"><h3>Usage</h3><div id="d-usage"></div></div>
</div>
</div>

<!-- Doom -->
<div id="v-doom" class="view">
<span class="back" onclick="go('')">&#8592; Back to catalog</span>
<div class="d-header">
<div class="d-title">doom_fire</div>
<div class="d-desc">Real-time Doom gameplay classifier. Extracts 24 game-state features and decides: FIRE, TURN LEFT, or TURN RIGHT. Logistic regression, no neural network, no GPU.</div>
<div class="d-stats">
<div class="d-stat"><span class="v">3</span><span class="l">Classes</span></div>
<div class="d-stat"><span class="v">24</span><span class="l">Features</span></div>
<div class="d-stat"><span class="v">&lt;1ms</span><span class="l">Latency</span></div>
<div class="d-stat"><span class="v">CPU</span><span class="l">Runtime</span></div>
</div>
</div>
<div class="doom-layout" style="margin-top:16px">
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
<span id="decision-label" style="font-size:20px;font-weight:700;color:var(--accent)">&mdash;</span>
<span id="decision-conf" style="font-size:13px;color:var(--muted)">&mdash;</span>
</div>
<div id="prob-bars" style="margin-top:8px"></div>
</div>
<div class="info-card">
<h3>Stats</h3>
<div class="stat-row"><span class="label">Classify latency</span><span id="stat-ms">&mdash;</span></div>
<div class="stat-row"><span class="label">Total kills</span><span id="stat-total-kills">0</span></div>
<div class="stat-row"><span class="label">Step</span><span id="stat-step">0</span></div>
<div class="stat-row"><span class="label">FPS</span><span id="stat-fps">&mdash;</span></div>
</div>
<div class="info-card">
<h3>How it works</h3>
<div style="font-size:11px;color:var(--muted);line-height:1.6">
Jeffy extracts 24 game-state features (enemy positions, health, ammo, action history) and runs them through a logistic regression classifier to decide: <span style="color:var(--accent)">FIRE</span>, <span style="color:var(--blue)">TURN LEFT</span>, or <span style="color:var(--orange)">TURN RIGHT</span>. No neural network, no GPU.
</div>
</div>
</div>
</div>
</div>

<!-- Inbox Demo -->
<div id="v-inbox" class="view">
<span class="back" onclick="go('')">&#8592; Back to catalog</span>
<div class="d-header">
<div class="d-title">Inbox Classifier Demo</div>
<div class="d-desc">Train an inbox router from 24 labeled examples, then watch it classify new messages in real time. CPU only, no API keys.</div>
<div class="d-stats">
<div class="d-stat"><span class="v">4</span><span class="l">Classes</span></div>
<div class="d-stat"><span class="v">24</span><span class="l">Train examples</span></div>
<div class="d-stat"><span class="v">~50ms</span><span class="l">Latency</span></div>
<div class="d-stat"><span class="v">CPU</span><span class="l">Runtime</span></div>
</div>
</div>
<div class="ib-layout" id="ib-demo" style="margin-top:16px;position:relative">
<div class="ib-panel" id="ib-trainPanel">
<div class="ib-panel-head"><h3>Training Data</h3><span class="meta">24 labeled examples</span></div>
<div class="ib-panel-body"><div id="ib-trainList"></div></div>
</div>
<div class="ib-panel" id="ib-inboxPanel" hidden>
<div class="ib-panel-head"><h3>Inbox</h3><span class="meta" id="ib-counter">0 / 8 sorted</span></div>
<div class="ib-panel-body">
<div>
<div class="ib-section-label">Incoming</div>
<div class="ib-incoming" id="ib-incoming"></div>
</div>
<div class="ib-categories">
<div><div class="ib-section-label"><span class="ib-cat-dot work"></span>Work <span class="ib-cat-n" id="ib-n-work">0</span></div><div class="ib-bucket" id="ib-b-work"></div></div>
<div><div class="ib-section-label"><span class="ib-cat-dot family"></span>Family <span class="ib-cat-n" id="ib-n-family">0</span></div><div class="ib-bucket" id="ib-b-family"></div></div>
<div><div class="ib-section-label"><span class="ib-cat-dot promo"></span>Promo <span class="ib-cat-n" id="ib-n-promo">0</span></div><div class="ib-bucket" id="ib-b-promo"></div></div>
<div><div class="ib-section-label"><span class="ib-cat-dot notif"></span>Notifications <span class="ib-cat-n" id="ib-n-notif">0</span></div><div class="ib-bucket" id="ib-b-notif"></div></div>
</div>
</div>
</div>
<div class="ib-panel" id="ib-logPanel">
<div class="ib-log-chrome"><div class="ib-dots"><span></span><span></span><span></span></div><span class="title">terminal</span></div>
<div class="ib-log-body" id="ib-logBody"></div>
</div>
</div>
<div style="margin-top:14px;display:flex;align-items:center;gap:14px;flex-wrap:wrap">
<code style="font:11px var(--mono);color:var(--muted)">pip install jeffy-classify</code>
<button id="ib-replayBtn" class="restart-btn" hidden>Replay</button>
</div>
<div class="d-grid" style="margin-top:20px">
<div class="d-section"><h3>Try it</h3>
<textarea id="ib-try-text" rows="2" spellcheck="false" placeholder="Enter an email subject or message...">Can you send me the Q4 projections?</textarea>
<div class="try-row"><span id="ib-try-lat" style="font-size:11px;color:var(--muted)"></span>
<button class="run-btn" id="ib-try-run">Classify &#8594;</button></div>
<div id="ib-try-err" class="err-msg"></div>
<div id="ib-try-res" style="display:none"><div id="ib-try-lbl" class="result-label"></div><div id="ib-try-bars"></div><div id="ib-try-meta" class="rmeta"></div></div>
</div>
<div class="d-section"><h3>Usage</h3><div id="ib-usage"></div></div>
</div>
</div>

<!-- Poker Demo -->
<div id="v-poker" class="view">
<span class="back" onclick="go('')">&#8592; Back to catalog</span>
<div class="d-header">
<div class="d-title">Poker AI Demo</div>
<div class="d-desc">Four AI players play Texas Hold'em. Each decision is a Jeffy classifier running on 18 game-state features. No neural network, no GPU.</div>
<div class="d-stats">
<div class="d-stat"><span class="v">4</span><span class="l">Players</span></div>
<div class="d-stat"><span class="v">4</span><span class="l">Actions</span></div>
<div class="d-stat"><span class="v">18</span><span class="l">Features</span></div>
<div class="d-stat"><span class="v">&lt;1ms</span><span class="l">Classify</span></div>
</div>
</div>
<div class="pk-layout" style="margin-top:16px">
<div>
<div class="pk-table" id="pk-table">
<div class="pk-community" id="pk-community"></div>
<div class="pk-pot" id="pk-pot"><span class="amt">0</span>pot</div>
<div class="pk-seat" data-pos="0" id="pk-s0">
<div class="pk-name">Alice <span class="chips" id="pk-stack0">1000</span></div>
<div class="pk-hole" id="pk-hole0"></div>
<div class="pk-action-badge" id="pk-act0"></div>
</div>
<div class="pk-seat" data-pos="1" id="pk-s1">
<div class="pk-action-badge" id="pk-act1"></div>
<div class="pk-hole" id="pk-hole1"></div>
<div class="pk-name">Bob <span class="chips" id="pk-stack1">1000</span></div>
</div>
<div class="pk-seat" data-pos="2" id="pk-s2">
<div class="pk-action-badge" id="pk-act2"></div>
<div class="pk-hole" id="pk-hole2"></div>
<div class="pk-name">Carol <span class="chips" id="pk-stack2">1000</span></div>
</div>
<div class="pk-seat" data-pos="3" id="pk-s3">
<div class="pk-name">Dave <span class="chips" id="pk-stack3">1000</span></div>
<div class="pk-hole" id="pk-hole3"></div>
<div class="pk-action-badge" id="pk-act3"></div>
</div>
</div>
<div style="margin-top:12px;display:flex;align-items:center;gap:12px">
<span class="pk-round-badge" id="pk-round">preflop</span>
<span style="font-size:11px;color:var(--muted)" id="pk-hand-num">Hand #1</span>
<span style="font-size:10px;color:var(--muted)" id="pk-status"></span>
<button class="restart-btn" id="pk-restart">Restart</button>
</div>
</div>
<div class="pk-sidebar">
<div class="info-card">
<h3>Current Hand</h3>
<div id="pk-decisions" style="max-height:200px;overflow-y:auto"></div>
</div>
<div class="info-card">
<h3>Hand History</h3>
<div id="pk-history" style="max-height:180px;overflow-y:auto"></div>
</div>
<div class="info-card">
<h3>How it works</h3>
<div style="font-size:11px;color:var(--muted);line-height:1.6">
Each player is a logistic regression classifier trained on 21K simulated poker decisions. Features include hand strength, pot odds, position, stack ratios, and betting patterns. The classifier outputs: <span style="color:#888">FOLD</span>, <span style="color:#6baa6b">CHECK</span>, <span style="color:#6ba3d8">CALL</span>, or <span style="color:#d4a04a">RAISE</span>.
</div>
</div>
</div>
</div>
</div>

</main>
<footer>
<span><a href="https://jeffyclassify.com" target="_blank">jeffyclassify.com</a></span>
<span><a href="/docs" target="_blank">API docs</a></span>
<span>pip install jeffy-classify</span>
<span><a href="https://github.com/nicobrenner/jeffy" target="_blank">GitHub</a></span>
</footer>
<script>
const $=id=>document.getElementById(id);
let allCaps=[],capMap={},ws=null,frameCount=0,fpsStart=0,doomActive=false;

fetch("/health").then(r=>r.json()).then(d=>{
  $("status").dataset.state=d.status;
  $("status").textContent=d.capabilities+" classifiers ready";
}).catch(()=>{$("status").textContent="Server unavailable"});

fetch("/v1/capabilities").then(r=>r.json()).then(d=>{
  allCaps=d.capabilities;
  allCaps.forEach(c=>capMap[c.task_id]=c);
  buildGrid();
  route();
});

function buildGrid(){
  const g=$("grid");g.innerHTML="";
  const sorted=[...allCaps].sort((a,b)=>{
    if(a.task_id==="doom_fire")return -1;
    if(b.task_id==="doom_fire")return 1;
    return(b.test_accuracy||0)-(a.test_accuracy||0);
  });
  sorted.forEach(c=>{
    const isFeat=c.encoder==="features";
    const acc=c.test_accuracy?Math.round(c.test_accuracy*1000)/10:null;
    const d=document.createElement("div");
    d.className="card"+(isFeat?" feat":"");
    d.onclick=()=>go("model/"+c.task_id);
    d.innerHTML=
      '<div class="card-top"><span class="card-id">'+c.task_id+'</span>'+
      (c.task_id==="doom_fire"?'<span class="demo-badge">Live Demo</span>':
       acc!==null?'<span class="card-acc">'+acc+'%</span>':'')+
      '</div>'+
      '<div class="card-name">'+c.name+'</div>'+
      '<div class="card-tags">'+
        '<span class="tag">'+c.n_classes+' classes</span>'+
        '<span class="tag">'+(isFeat?(c.n_features||"?")+' features':'text')+'</span>'+
        '<span class="tag">'+c.license+'</span>'+
      '</div>'+
      (acc!==null?'<div class="card-bar"><div class="card-bar-fill" style="width:'+acc+'%"></div></div>':'');
    g.appendChild(d);
    if(c.task_id==="doom_fire"){
      const ib=document.createElement("div");
      ib.className="card";
      ib.onclick=()=>go("inbox");
      ib.innerHTML=
        '<div class="card-top"><span class="card-id">inbox_router</span><span class="demo-badge">Live Demo</span></div>'+
        '<div class="card-name">Train and classify emails into Work, Family, Promo, Notifications</div>'+
        '<div class="card-tags"><span class="tag">4 classes</span><span class="tag">text</span><span class="tag">custom</span></div>'+
        '<div class="card-bar"><div class="card-bar-fill" style="width:92%"></div></div>';
      g.appendChild(ib);
      var pk=document.createElement("div");
      pk.className="card feat";
      pk.onclick=function(){go("poker");};
      pk.innerHTML=
        '<div class="card-top"><span class="card-id">poker_decision</span><span class="demo-badge">Live Demo</span></div>'+
        '<div class="card-name">4 AI players play Texas Hold\\u2019em using game-state classifiers</div>'+
        '<div class="card-tags"><span class="tag">4 classes</span><span class="tag">18 features</span><span class="tag">MIT</span></div>'+
        '<div class="card-bar"><div class="card-bar-fill" style="width:88%"></div></div>';
      g.appendChild(pk);
    }
  });
}

function showDetail(tid){
  const c=capMap[tid];if(!c)return;
  const isFeat=c.encoder==="features";
  const acc=c.test_accuracy?Math.round(c.test_accuracy*1000)/10:null;

  $("dh").innerHTML=
    '<div class="d-title">'+c.task_id+'</div>'+
    '<div class="d-desc">'+c.name+'</div>'+
    '<div class="d-stats">'+
      (acc!==null?'<div class="d-stat"><span class="v">'+acc+'%</span><span class="l">Accuracy</span></div>':'')+
      '<div class="d-stat"><span class="v">'+c.n_classes+'</span><span class="l">Classes</span></div>'+
      (c.train_examples?'<div class="d-stat"><span class="v">'+c.train_examples.toLocaleString()+'</span><span class="l">Train examples</span></div>':'')+
      '<div class="d-stat"><span class="v">'+(isFeat?"features":"bge-large")+'</span><span class="l">Encoder</span></div>'+
    '</div>';

  const lb=$("d-labels");lb.innerHTML="";
  if(c.labels)Object.values(c.labels).forEach(l=>{lb.innerHTML+='<span class="chip">'+l+'</span>';});

  $("d-info").innerHTML=
    '<div style="font-size:12px;line-height:2">'+
    '<div><span style="color:var(--muted)">License:</span> '+c.license+'</div>'+
    '<div><span style="color:var(--muted)">Encoder:</span> '+c.encoder+'</div>'+
    '<div><span style="color:var(--muted)">Type:</span> '+(isFeat?"Feature-based (numeric vectors)":"Text classifier")+'</div>'+
    (c.train_accuracy?'<div><span style="color:var(--muted)">Train accuracy:</span> '+Math.round(c.train_accuracy*1000)/10+'%</div>':'')+
    '</div>';

  if(isFeat){
    $("d-try").innerHTML=
      '<div style="font-size:12px;color:var(--muted);line-height:1.6">'+
      'This classifier operates on numeric feature vectors'+(c.n_features?' ('+c.n_features+' features)':'')+', not text.'+
      (tid==="doom_fire"?'<br><br><a href="#doom" style="color:var(--accent)" onclick="go(&apos;doom&apos;);return false">Watch the live Doom demo &#8594;</a>':'')+
      '</div>';
  } else {
    $("d-try").innerHTML=
      '<textarea id="try-text" rows="3" spellcheck="false" placeholder="Enter text to classify..."></textarea>'+
      '<div class="try-row"><span id="try-lat" style="font-size:11px;color:var(--muted)"></span>'+
      '<button class="run-btn" id="try-run">Classify &#8594;</button></div>'+
      '<div id="try-err" class="err-msg"></div>'+
      '<div id="try-res" style="display:none"><div id="try-lbl" class="result-label"></div><div id="try-bars"></div><div id="try-meta" class="rmeta"></div></div>'+
      '<details id="try-raw-s" style="display:none"><summary>Response JSON</summary><pre id="try-raw"></pre></details>';
    $("try-run").onclick=()=>runPredict(tid);
    fetch("/v1/capabilities/"+tid).then(r=>r.json()).then(d=>{
      const ta=$("try-text");
      if(ta&&d.example_request&&d.example_request.text)ta.value=d.example_request.text;
    });
  }

  var curlData=isFeat
    ?JSON.stringify({features:[0.0],task:tid})
    :JSON.stringify({text:"your text here",task:tid});
  var NL=String.fromCharCode(10),BS=String.fromCharCode(92),DQ=String.fromCharCode(34);
  var curl="curl -s -X POST "+location.origin+"/v1/predict "+BS+NL+"  -H "+DQ+"Content-Type: application/json"+DQ+" "+BS+NL+"  -d "+JSON.stringify(curlData)+" | python3 -m json.tool";
  var py=["from jeffy.engine import Engine","",
    "engine = Engine()","engine.load()",
    isFeat?"result = engine.predict_features("+DQ+tid+DQ+", [0.0] * "+(c.n_features||24)+")":"result = engine.predict("+DQ+tid+DQ+", "+DQ+"your text here"+DQ+")",
    "print(result["+DQ+"label"+DQ+"])"].join(NL);
  $("d-usage").innerHTML=
    '<div style="font-size:11px;color:var(--muted);margin-bottom:6px">curl</div>'+
    '<div class="codeblk"><pre>'+curl+'</pre><button class="cpbtn" onclick="cpCode(this)">Copy</button></div>'+
    '<div style="font-size:11px;color:var(--muted);margin:12px 0 6px">Python</div>'+
    '<div class="codeblk"><pre>'+py+'</pre><button class="cpbtn" onclick="cpCode(this)">Copy</button></div>'+
    '<div style="font-size:11px;color:var(--muted);margin-top:12px">pip install jeffy-classify</div>';
}

async function runPredict(tid){
  const ta=$("try-text");if(!ta)return;
  const btn=$("try-run");
  btn.disabled=true;btn.textContent="Classifying\\u2026";
  const errEl=$("try-err");errEl.style.display="none";
  const resEl=$("try-res");resEl.style.display="none";
  const rawS=$("try-raw-s");rawS.style.display="none";
  const t0=performance.now();
  try{
    const r=await fetch("/v1/predict",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({text:ta.value,task:tid})});
    const d=await r.json();
    if(!r.ok)throw Error(d.detail||JSON.stringify(d));
    $("try-lbl").textContent=d.label;
    const bars=$("try-bars");bars.innerHTML="";
    Object.entries(d.probabilities).forEach(([label,prob])=>{
      const row=document.createElement("div");row.className="bar-row";
      row.innerHTML='<span class="bar-label" title="'+label+'">'+label+'</span>'+
        '<div class="track"><div class="fill" style="width:'+Math.round(prob*100)+'%"></div></div>'+
        '<span class="pct">'+(prob*100).toFixed(1)+'%</span>';
      bars.appendChild(row);
    });
    $("try-meta").innerHTML="Confidence: "+(d.confidence*100).toFixed(1)+"%"+
      " \\u00b7 Embedding: "+d.embedding_ms+"ms \\u00b7 Classifier: "+d.classifier_ms+"ms"+
      " \\u00b7 Total: "+d.latency_ms+"ms";
    resEl.style.display="block";
    $("try-raw").textContent=JSON.stringify(d,null,2);
    rawS.style.display="block";
    $("try-lat").textContent=Math.round(performance.now()-t0)+"ms round-trip";
  }catch(err){
    errEl.textContent=err.message;errEl.style.display="block";
  }finally{
    btn.disabled=false;btn.textContent="Classify \\u2192";
  }
}

function cpCode(btn){
  const pre=btn.parentElement.querySelector("pre");
  navigator.clipboard.writeText(pre.textContent).then(()=>{
    btn.textContent="Copied!";setTimeout(()=>btn.textContent="Copy",1500);
  }).catch(()=>{
    const s=window.getSelection(),r=document.createRange();
    r.selectNodeContents(pre);s.removeAllRanges();s.addRange(r);
  });
}

// --- Doom Stream ---
function startDoom(){
  if(ws&&ws.readyState===WebSocket.OPEN)ws.close();
  const proto=location.protocol==="https:"?"wss:":"ws:";
  ws=new WebSocket(proto+"//"+location.host+"/v1/doom/stream");
  $("stream-status").textContent="Connecting...";
  $("decision-log").innerHTML="";
  frameCount=0;fpsStart=performance.now();
  doomActive=true;

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
      pb.innerHTML+='<div style="display:flex;align-items:center;gap:6px;font-size:10px;margin:3px 0;color:var(--muted)"><span style="width:70px;text-align:right">'+lbl.replace("_"," ")+'</span><div style="flex:1;height:5px;background:#222;border-radius:3px;overflow:hidden"><div style="width:'+pct+'%;height:100%;background:'+(colors[lbl]||"#888")+';border-radius:3px"></div></div><span style="width:30px">'+pct+'%</span></div>';
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
    ws=null;doomActive=false;
  };
  ws.onerror=()=>{$("stream-status").textContent="Connection error";};
}

function stopDoom(){
  if(ws&&ws.readyState===WebSocket.OPEN)ws.close();
  doomActive=false;
}

// --- Inbox Demo ---
var ibTRAIN=[
  {text:"Please review the Q3 budget draft",cat:"work"},
  {text:"Mom\\u2019s birthday is next Saturday",cat:"family"},
  {text:"Flash sale \\u2014 70% off all items",cat:"promo"},
  {text:"Your flight has been rescheduled",cat:"notif"},
  {text:"Team standup moved to 10am",cat:"work"},
  {text:"Uncle Joe is coming to visit",cat:"family"},
];
var ibMSGS=[
  {text:"Can you send me the Q4 projections?",cat:"work",conf:.59,ms:48},
  {text:"Aunt Clara is hosting Thanksgiving",cat:"family",conf:.78,ms:52},
  {text:"Buy 2 get 1 free \\u2014 today only!",cat:"promo",conf:.76,ms:45},
  {text:"Your order #7832 has shipped",cat:"notif",conf:.59,ms:51},
  {text:"Sprint planning at 3pm in room B",cat:"work",conf:.53,ms:49},
  {text:"Dad wants to know about the cookout",cat:"family",conf:.64,ms:47},
  {text:"Limited time: upgrade for $5/mo",cat:"promo",conf:.43,ms:53},
  {text:"Your credit card statement is ready",cat:"notif",conf:.75,ms:50},
];
var ibStopped=false,ibSorted=0,ibActive=false;
var ibWait=ms=>new Promise(r=>setTimeout(r,ms));
var ibColors={work:"#6ba3d8",family:"#d47e5c",promo:"#cda24a",notif:"#5eaa7e"};

function ibLog(html){
  var el=document.createElement("div");
  el.className="ib-log-line";el.innerHTML=html;
  $("ib-logBody").prepend(el);
}
function ibLogSp(){
  var el=document.createElement("div");
  el.className="ib-log-line spacer";
  $("ib-logBody").prepend(el);
}

async function ibShowTraining(){
  ibLog('<span class="dim">$ jeffy-train --task inbox_router --input examples.csv</span>');
  await ibWait(600);
  for(var i=0;i<ibTRAIN.length;i++){
    if(ibStopped)return;
    var ex=ibTRAIN[i];
    var row=document.createElement("div");
    row.className="ib-train-row";row.style.opacity="0";
    row.innerHTML='<span class="ib-train-text">\\u201c'+ex.text+'\\u201d</span><span class="ib-train-arrow">\\u2192</span><span class="ib-train-label '+ex.cat+'">'+ex.cat+'</span>';
    $("ib-trainList").appendChild(row);
    try{await row.animate([{opacity:0,transform:"translateX(-10px)"},{opacity:1,transform:"translateX(0)"}],{duration:260,fill:"forwards",easing:"ease-out"}).finished;}catch(e){}
    await ibWait(220);
  }
  var more=document.createElement("div");
  more.className="ib-train-more";more.textContent="\\u22ef +18 more labeled examples";
  more.style.opacity="0";$("ib-trainList").appendChild(more);
  try{await more.animate([{opacity:0},{opacity:1}],{duration:300,fill:"forwards"}).finished;}catch(e){}
  await ibWait(500);
  if(ibStopped)return;
  ibLog('<span class="dim">[info]</span> Encoding 24 examples\\u2026');
  await ibWait(500);
  ibLog('<span class="dim">[info]</span> Fitting classifier (4 classes)\\u2026');
  await ibWait(600);
  ibLog('<span class="dim">[info]</span> CV accuracy: <span class="ok">92%</span> (3-fold)');
  await ibWait(350);
  ibLog('<span class="dim">[info]</span> Saved <span class="bright">inbox_router</span> (1.8s)');
  ibLogSp();
  await ibWait(900);
}

async function ibSwitchToInbox(){
  $("ib-trainPanel").style.opacity="0";
  await ibWait(350);
  $("ib-trainPanel").hidden=true;
  $("ib-inboxPanel").hidden=false;
  $("ib-inboxPanel").style.opacity="0";
  requestAnimationFrame(function(){$("ib-inboxPanel").style.opacity="1";});
  $("ib-inboxPanel").style.transition="opacity .35s";
  await ibWait(350);
  ibLog('<span class="dim">$ jeffy-serve --task inbox_router</span>');
  await ibWait(400);
  ibLog('<span class="dim">[info]</span> <span class="bright">inbox_router</span> ready (24 examples, 4 classes)');
  await ibWait(300);
  ibLog('<span class="dim">[info]</span> Listening on <span class="bright">http://localhost:8400</span>');
  ibLogSp();
  await ibWait(800);
}

async function ibArriveMsg(msg){
  var card=document.createElement("div");
  card.className="ib-msg pending";
  card.innerHTML='<span class="ib-msg-text">'+msg.text+'</span><span class="ib-typing"><span></span><span></span><span></span></span>';
  $("ib-incoming").appendChild(card);

  var demo=$("ib-demo"),dr=demo.getBoundingClientRect(),cr=card.getBoundingClientRect();
  var ip=$("ib-inboxPanel").getBoundingClientRect();
  var ghost=document.createElement("div");
  ghost.className="ib-msg ib-ghost";ghost.innerHTML=card.innerHTML;
  ghost.style.width=cr.width+"px";ghost.style.top=(cr.top-dr.top)+"px";
  var targetX=cr.left-dr.left,startX=ip.left-dr.left-cr.width-20;
  ghost.style.left=startX+"px";
  demo.appendChild(ghost);
  try{await ghost.animate([{transform:"translateX(0)",opacity:.8},{transform:"translateX("+(targetX-startX)+"px)",opacity:1}],{duration:480,easing:"cubic-bezier(.15,.75,.3,1)",fill:"forwards"}).finished;}catch(e){}
  card.classList.remove("pending");card.classList.add("show");
  ghost.remove();
  return card;
}

async function ibDrawLine(card,color,direction){
  var demo=$("ib-demo"),dr=demo.getBoundingClientRect(),cr=card.getBoundingClientRect();
  var lp=$("ib-logPanel").getBoundingClientRect(),ip=$("ib-inboxPanel").getBoundingClientRect();
  var y=cr.top+cr.height/2-dr.top,x=ip.right-dr.left+1,w=lp.left-ip.right-2;
  if(w<2)return;
  var line=document.createElement("div");
  line.className="ib-signal "+(direction==="req"?"to-right":"to-left");
  line.style.setProperty("--lc",color);
  line.style.left=x+"px";line.style.top=(y-1)+"px";line.style.width=w+"px";
  demo.appendChild(line);
  var clip=direction==="req"?["inset(-10px 100% -10px 0)","inset(-10px -10px -10px 0)"]:["inset(-10px 0 -10px 100%)","inset(-10px 0 -10px -10px)"];
  try{await line.animate([{clipPath:clip[0]},{clipPath:clip[1]}],{duration:340,easing:"ease-out",fill:"forwards"}).finished;}catch(e){}
  await ibWait(180);
  try{await line.animate([{opacity:1},{opacity:0}],{duration:250,fill:"forwards"}).finished;}catch(e){}
  line.remove();
}

async function ibProcessMsg(msg){
  if(ibStopped)return;
  var card=await ibArriveMsg(msg);
  await ibWait(400);
  if(ibStopped)return;
  await ibDrawLine(card,"#6BA3C7","req");
  if(ibStopped)return;
  var trunc=msg.text.length>34?msg.text.slice(0,34)+"\\u2026":msg.text;
  ibLog('<span class="method">POST</span> <span class="bright">/v1/predict</span>  <span class="dim">"'+trunc+'"</span>');
  await ibWait(250+msg.ms*2.5);
  if(ibStopped)return;
  ibLog('<span class="ok">\\u2190 200</span>  <span class="lbl-'+msg.cat+'">'+msg.cat+'</span>  <span class="dim">'+msg.ms+'ms</span>');
  ibLogSp();
  var cc=ibColors[msg.cat];
  await ibDrawLine(card,cc,"res");
  if(ibStopped)return;
  card.style.boxShadow="0 0 0 2px "+cc;
  var dots=card.querySelector(".ib-typing");if(dots)dots.remove();
  card.classList.add(msg.cat);
  var lat=document.createElement("span");lat.className="ib-msg-lat";lat.textContent=msg.ms+"ms";
  card.appendChild(lat);
  await ibWait(320);
  card.style.boxShadow="0 0 0 2px transparent";card.style.transition="box-shadow .35s";
  await ibWait(280);
  if(ibStopped)return;
  card.classList.add("leaving");
  await ibWait(220);
  card.remove();
  var sc=document.createElement("div");
  sc.className="ib-msg sorted "+msg.cat;
  sc.style.opacity="0";sc.style.transform="translateY(5px)";
  sc.innerHTML='<span class="ib-msg-text">'+msg.text+'</span><span class="ib-msg-lat">'+msg.ms+'ms</span>';
  $("ib-b-"+msg.cat).appendChild(sc);
  requestAnimationFrame(function(){sc.style.transition="opacity .25s, transform .25s";sc.style.opacity="1";sc.style.transform="translateY(0)";});
  $("ib-n-"+msg.cat).textContent=$("ib-b-"+msg.cat).children.length;
  ibSorted++;
  $("ib-counter").textContent=ibSorted+" / 8 sorted";
  await ibWait(300);
}

async function ibRun(){
  ibStopped=false;ibSorted=0;ibActive=true;
  $("ib-counter").textContent="0 / 8 sorted";
  $("ib-replayBtn").hidden=true;
  $("ib-incoming").innerHTML="";$("ib-logBody").innerHTML="";$("ib-trainList").innerHTML="";
  $("ib-demo").querySelectorAll(".ib-ghost,.ib-signal").forEach(function(e){e.remove();});
  ["work","family","promo","notif"].forEach(function(c){$("ib-b-"+c).innerHTML="";$("ib-n-"+c).textContent="0";});
  $("ib-trainPanel").hidden=false;$("ib-trainPanel").style.opacity="1";
  $("ib-inboxPanel").hidden=true;
  await ibShowTraining();
  if(ibStopped)return;
  await ibSwitchToInbox();
  if(ibStopped)return;
  for(var i=0;i<ibMSGS.length;i++){
    await ibProcessMsg(ibMSGS[i]);
    if(ibStopped)return;
  }
  await ibWait(500);
  ibLog('<span class="dim">\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500</span>');
  var total=ibMSGS.reduce(function(s,m){return s+m.ms;},0);
  ibLog('<span class="bright">8 sorted \\u00b7 avg '+Math.round(total/8)+'ms \\u00b7 no API keys</span>');
  await ibWait(400);
  $("ib-replayBtn").hidden=false;
  ibActive=false;
}

function ibStop(){ibStopped=true;ibActive=false;}

$("ib-replayBtn").addEventListener("click",function(){ibStopped=true;setTimeout(ibRun,100);});

// Inbox Try-it + Usage
$("ib-try-run").onclick=function(){
  var ta=$("ib-try-text");if(!ta||!ta.value.trim())return;
  var btn=$("ib-try-run");
  btn.disabled=true;btn.textContent="Classifying…";
  $("ib-try-err").style.display="none";
  $("ib-try-res").style.display="none";
  var t0=performance.now();
  fetch("/v1/predict",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text:ta.value,task:"inbox_router"})})
  .then(function(r){return r.json().then(function(d){if(!r.ok)throw Error(d.detail||JSON.stringify(d));return d;});})
  .then(function(d){
    $("ib-try-lbl").textContent=d.label;
    var bars=$("ib-try-bars");bars.innerHTML="";
    Object.entries(d.probabilities).forEach(function(e){
      var label=e[0],prob=e[1];
      var row=document.createElement("div");row.className="bar-row";
      row.innerHTML='<span class="bar-label" title="'+label+'">'+label+'</span>'+
        '<div class="track"><div class="fill" style="width:'+Math.round(prob*100)+'%"></div></div>'+
        '<span class="pct">'+(prob*100).toFixed(1)+'%</span>';
      bars.appendChild(row);
    });
    $("ib-try-meta").innerHTML="Confidence: "+(d.confidence*100).toFixed(1)+"%"+
      " · Embedding: "+d.embedding_ms+"ms · Classifier: "+d.classifier_ms+"ms"+
      " · Total: "+d.latency_ms+"ms";
    $("ib-try-res").style.display="block";
    $("ib-try-lat").textContent=Math.round(performance.now()-t0)+"ms round-trip";
  })
  .catch(function(err){$("ib-try-err").textContent=err.message;$("ib-try-err").style.display="block";})
  .finally(function(){btn.disabled=false;btn.textContent="Classify →";});
};

(function(){
  var NL=String.fromCharCode(10),BS=String.fromCharCode(92),DQ=String.fromCharCode(34);
  var curlData=JSON.stringify({text:"Can you send me the Q4 projections?",task:"inbox_router"});
  var curl="curl -s -X POST "+location.origin+"/v1/predict "+BS+NL+"  -H "+DQ+"Content-Type: application/json"+DQ+" "+BS+NL+"  -d "+JSON.stringify(curlData)+" | python3 -m json.tool";
  var py=["from jeffy.engine import Engine","","engine = Engine()","engine.load()","result = engine.predict("+DQ+"inbox_router"+DQ+", "+DQ+"Can you send me the Q4 projections?"+DQ+")","print(result["+DQ+"label"+DQ+"])"].join(NL);
  $("ib-usage").innerHTML=
    '<div style="font-size:11px;color:var(--muted);margin-bottom:6px">curl</div>'+
    '<div class="codeblk"><pre>'+curl+'</pre><button class="cpbtn" onclick="cpCode(this)">Copy</button></div>'+
    '<div style="font-size:11px;color:var(--muted);margin:12px 0 6px">Python</div>'+
    '<div class="codeblk"><pre>'+py+'</pre><button class="cpbtn" onclick="cpCode(this)">Copy</button></div>'+
    '<div style="font-size:11px;color:var(--muted);margin-top:12px">pip install jeffy-classify</div>';
})();

// --- Poker Demo ---
var pkWs=null,pkActive=false;
var pkSuitColor={"s":"","h":"red","d":"red","c":""};

function renderCard(code){
  if(!code)return '<div class="pk-card empty"></div>';
  var rank=code[0]==="T"?"10":code[0];
  var suit=code[1];
  var sym={"s":"\\u2660","h":"\\u2665","d":"\\u2666","c":"\\u2663"}[suit]||"";
  var cls=pkSuitColor[suit]||"";
  return '<div class="pk-card '+cls+'">'+rank+sym+'</div>';
}

function pkUpdate(d){
  var cc=$("pk-community");
  var html="";
  for(var i=0;i<5;i++){
    if(d.community&&i<d.community.length){
      html+=renderCard(d.community[i]);
    } else {
      html+='<div class="pk-card empty"></div>';
    }
  }
  cc.innerHTML=html;
  $("pk-pot").innerHTML='<span class="amt">'+d.pot+'</span>pot';
  for(var i=0;i<4;i++){
    var p=d.players[i];
    var seat=$("pk-s"+i);
    seat.className="pk-seat"+(p.folded?" folded":"");
    seat.setAttribute("data-pos",i);
    $("pk-stack"+i).textContent=p.stack;
    var holeHtml="";
    if(p.hole&&p.hole.length===2&&!p.folded){
      holeHtml=renderCard(p.hole[0])+renderCard(p.hole[1]);
    } else if(p.folded){
      holeHtml='<div class="pk-card back"></div><div class="pk-card back"></div>';
    }
    $("pk-hole"+i).innerHTML=holeHtml;
    var actEl=$("pk-act"+i);
    actEl.textContent=p.last_action?p.last_action.toUpperCase():"";
    actEl.className="pk-action-badge "+(p.last_action||"");
  }
  $("pk-round").textContent=d.round;
  $("pk-hand-num").textContent="Hand #"+d.hand_number;
  if(d.detail&&d.event==="action"){
    var dd=d.detail;
    var row=document.createElement("div");
    row.className="pk-decision-row";
    row.innerHTML='<span class="pname">'+dd.name+'</span><span class="pact '+dd.action+'">'+dd.action.toUpperCase()+'</span><span style="color:var(--muted);font-size:10px">'+(dd.confidence*100).toFixed(0)+'%</span>';
    $("pk-decisions").prepend(row);
    while($("pk-decisions").children.length>20)$("pk-decisions").lastChild.remove();
  }
  if(d.event==="showdown"&&d.detail){
    var sd=d.detail;
    var row=document.createElement("div");
    row.className="pk-decision-row";
    row.style.borderColor="var(--accent2)";
    row.innerHTML='<span style="color:var(--accent2);font-weight:700">\\u2605 '+sd.winner_name+' wins '+sd.pot+'</span><span style="color:var(--muted);font-size:10px">'+sd.win_reason+'</span>';
    $("pk-decisions").prepend(row);
  }
  if(d.event==="new_hand"){$("pk-decisions").innerHTML="";}
  if(d.hand_history){
    var hh=$("pk-history");hh.innerHTML="";
    for(var i=d.hand_history.length-1;i>=0;i--){
      var h=d.hand_history[i];
      var r=document.createElement("div");
      r.className="pk-history-row";
      r.textContent="#"+h.hand+" "+h.winner+" wins "+h.pot+" ("+h.reason+")";
      hh.appendChild(r);
    }
  }
}

function startPoker(){
  if(pkWs&&pkWs.readyState===WebSocket.OPEN)pkWs.close();
  var proto=location.protocol==="https:"?"wss:":"ws:";
  pkWs=new WebSocket(proto+"//"+location.host+"/v1/poker/stream");
  $("pk-status").textContent="Connecting...";
  $("pk-decisions").innerHTML="";
  $("pk-history").innerHTML="";
  pkActive=true;
  pkWs.onopen=function(){$("pk-status").textContent="Live";};
  pkWs.onmessage=function(e){pkUpdate(JSON.parse(e.data));};
  pkWs.onclose=function(){$("pk-status").textContent="Disconnected";pkWs=null;pkActive=false;};
  pkWs.onerror=function(){$("pk-status").textContent="Error";};
}

function stopPoker(){
  if(pkWs&&pkWs.readyState===WebSocket.OPEN)pkWs.close();
  pkActive=false;
}

$("pk-restart").onclick=function(){startPoker();};

// --- Routing ---
function go(h){location.hash=h;}

function route(){
  var h=location.hash.replace(/^#/,"");
  document.querySelectorAll(".view").forEach(function(v){v.classList.remove("active");});
  if(h==="doom"){
    $("v-doom").classList.add("active");
    if(!doomActive)startDoom();
    ibStop();stopPoker();
  } else if(h==="inbox"){
    $("v-inbox").classList.add("active");
    if(!ibActive)ibRun();
    stopDoom();stopPoker();
  } else if(h==="poker"){
    $("v-poker").classList.add("active");
    if(!pkActive)startPoker();
    stopDoom();ibStop();
  } else if(h.startsWith("model/")){
    var tid=h.split("/")[1];
    if(tid==="doom_fire"){go("doom");return;}
    if(tid==="poker_decision"){go("poker");return;}
    $("v-detail").classList.add("active");
    showDetail(tid);
    stopDoom();ibStop();stopPoker();
  } else {
    $("v-catalog").classList.add("active");
    stopDoom();ibStop();stopPoker();
  }
}

window.addEventListener("hashchange",route);
$("restart-btn").onclick=function(){startDoom();};
</script>
</body>
</html>"""


def main():
    import uvicorn
    port = int(os.environ.get("JEFFY_PORT", "8400"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
