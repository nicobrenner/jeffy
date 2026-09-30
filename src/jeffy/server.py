"""Jeffy pretrained decision server.

Jeff-compatible API with pretrained capability catalog.

Run:
    python -m jeffy.server
    # or
    uvicorn jeffy.server:app --port 8400
"""

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .engine import Engine

import os

app = FastAPI(
    title="Jeffy",
    description="Pretrained decision engine with reusable embeddings and tiny classifiers.",
    version="0.1.0a3",
)

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        device = os.environ.get("JEFFY_DEVICE", "cpu")
        _engine = Engine(device=device)  # uses default_pack_dir()
        _engine.load()
    return _engine


# --- Request/Response models ---

class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to classify")
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
        "example_request": {
            "text": _example_text(task_id),
            "task": task_id,
        },
    }


@app.post("/v1/predict")
def predict(req: PredictRequest):
    engine = get_engine()
    result = engine.predict(req.task, req.text)
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
    import json as _json

    engine = get_engine()
    state_text = req.state if isinstance(req.state, str) else _json.dumps(req.state)

    # Validate capability exists
    if req.capability not in engine.capabilities:
        raise HTTPException(400,
            f"Capability '{req.capability}' not found. "
            f"Available: {sorted(engine.capabilities.keys())}")

    cap = engine.capabilities[req.capability]

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

            result = engine.predict(req.capability, state_text)

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
            result = engine.predict(req.capability, state_text)
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
:root{color-scheme:light;--bg:#fff;--panel:#fff;--line:#e1e7e3;--text:#000;--muted:#555;--accent:#2563eb;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 var(--mono)}
header,main,footer{max-width:1200px;margin:auto;padding:20px 32px}
header{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line)}
.brand{font-size:22px;font-weight:700;letter-spacing:-0.7px}
.brand span{font-weight:400;color:var(--muted);font-size:14px;margin-left:12px}
.status{font-size:12px;color:var(--muted)}.status::before{content:"●";color:#99a7ac;margin-right:6px}
.status[data-state="ready"]::before{color:#22c55e}
.layout{display:grid;grid-template-columns:1.05fr 1fr;gap:24px}
.panel{border:1px solid var(--line);border-radius:12px;overflow:hidden}
.panel-head{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between}
.panel-body{padding:20px}
h2{font-size:15px;margin:0}
label{display:block;color:var(--muted);font-size:12px;margin-bottom:6px}
textarea,select{width:100%;background:#fff;color:var(--text);border:1px solid #d5ded8;border-radius:7px;padding:10px;font:13px/1.5 var(--mono)}
textarea{resize:vertical;min-height:120px}
textarea:focus,select:focus{outline:2px solid var(--accent);outline-offset:1px}
select{padding:8px}
.field{margin-top:16px}
button{border:1px solid var(--line);background:transparent;color:var(--text);border-radius:7px;padding:8px 14px;cursor:pointer;font:12px var(--mono)}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.5;cursor:wait}
.run{background:#dbeafe;color:#1e40af;border:0;font-weight:650;padding:12px 20px;min-width:140px}
.actions{display:flex;align-items:center;justify-content:flex-end;margin-top:16px;gap:12px}
.result-value{font-size:24px;font-weight:650;letter-spacing:-0.5px;margin:8px 0 12px}
.bar-row{display:grid;grid-template-columns:minmax(80px,1.2fr) 2fr 50px;gap:10px;align-items:center;font-size:11px;margin:8px 0}
.bar-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.track{height:9px;background:#edf3ee;border-radius:8px;overflow:hidden}
.fill{height:100%;background:var(--accent);border-radius:8px}
.percentage{text-align:right;color:var(--muted)}
.badge{font-size:11px;color:var(--muted);border:1px solid var(--line);border-radius:5px;padding:3px 7px}
.meta{font-size:11px;color:var(--muted);margin-top:12px;line-height:1.8}
.empty{padding:50px 18px;text-align:center;color:var(--muted)}
.error{color:#991b1b;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;font-size:12px}
details{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}
summary{color:var(--muted);font-size:12px;cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:11px/1.7 var(--mono);max-height:400px;overflow:auto}
.hidden{display:none!important}
footer{font-size:11px;color:var(--muted);display:flex;justify-content:space-between}
a{color:var(--muted)}
@media(max-width:800px){.layout{grid-template-columns:1fr}}
</style>
</head>
<body>
<header>
<div class="brand">Jeffy<span>Playground</span></div>
<div id="status" class="status">Connecting</div>
</header>
<main>
<div class="layout">
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
</main>
<footer>
<span><a href="/docs" target="_blank">API docs ↗</a></span>
<span><a href="/v1/capabilities" target="_blank">Capabilities ↗</a></span>
</footer>
<script>
const $=id=>document.getElementById(id);
let caps={};

fetch("/health").then(r=>r.json()).then(d=>{
  $("status").dataset.state=d.status;
  $("status").textContent=d.capabilities+" capabilities ready";
}).catch(()=>{$("status").textContent="Server unavailable"});

fetch("/v1/capabilities").then(r=>r.json()).then(d=>{
  const sel=$("task");
  d.capabilities.forEach(c=>{
    caps[c.task_id]=c;
    const o=document.createElement("option");
    o.value=c.task_id;
    o.textContent=c.task_id+" — "+c.name;
    sel.appendChild(o);
  });
  if(d.capabilities.length)updateTaskInfo();
});

$("task").onchange=updateTaskInfo;

function updateTaskInfo(){
  const c=caps[$("task").value];
  if(!c)return;
  $("task-info").innerHTML=
    c.n_classes+" classes · "+c.train_examples+" training examples"+
    (c.train_accuracy?" · train acc "+Math.round(c.train_accuracy*1000)/10+"%":"")+
    "<br>Encoder: "+c.encoder+"<br>License: "+c.license;
  // Load example text
  fetch("/v1/capabilities/"+c.task_id).then(r=>r.json()).then(d=>{
    if(d.example_request)$("text").value=d.example_request.text;
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
</script>
</body>
</html>"""


def main():
    import uvicorn
    port = int(os.environ.get("JEFFY_PORT", "8400"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
