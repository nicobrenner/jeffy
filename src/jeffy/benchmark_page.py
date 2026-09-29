"""Generate benchmark HTML page from evaluation results.

Usage:
    python -m jeffy.benchmark_page
"""

import json
from pathlib import Path


def generate_html(results_path: str = "data/eval_results/benchmark.json",
                  jeff_path: str = "data/eval_results/jeff_comparison.json",
                  output_path: str = "data/eval_results/benchmark.html"):
    with open(results_path) as f:
        data = json.load(f)

    # Load Jeff results if available
    jeff_data = {}
    jeff_note = "not evaluated"
    jeff_path = Path(jeff_path)
    if jeff_path.exists():
        with open(jeff_path) as f:
            jeff_raw = json.load(f)
        for r in jeff_raw.get("results", []):
            jeff_data[r["task_id"]] = r
        jeff_note = jeff_raw.get("note", "")

    results = sorted(data["results"], key=lambda r: -r["accuracy"])
    hw = data.get("hardware", {})

    rows = ""
    for r in results:
        bl_maj = r["baselines"]["majority"]["accuracy"] if r.get("baselines") else None
        bl_tfidf = r["baselines"]["tfidf_lr"]["accuracy"] if r.get("baselines") else None
        lat = r["latency"]["total_p50_ms"] if r.get("latency") else None
        emb_lat = r["latency"]["embed_p50_ms"] if r.get("latency") else None
        clf_lat = r["latency"]["clf_p50_ms"] if r.get("latency") else None
        notes = "; ".join(r.get("notes", []))
        ci = f'[{r["accuracy_ci_lo"]:.3f}, {r["accuracy_ci_hi"]:.3f}]'

        # Clinc detail
        extra = ""
        if r.get("clinc_detail"):
            cd = r["clinc_detail"]
            extra = f' (in-scope {cd["in_scope_acc"]:.1%}, OOS {cd["oos_acc"]:.1%})'

        artifact_kb = {"banking77": 334, "clinc_oos": 632, "massive_intent": 271,
                       "ag_news": 41, "dbpedia": 81, "emotion": 49, "imdb": 29,
                       "sms_spam": 29, "snli": 37, "sst2": 29,
                       "tweet_eval_emotion": 41, "tweet_eval_offensive": 29,
                       "tweet_eval_sentiment": 37}.get(r["task_id"], "?")

        rows += f"""<tr>
<td><strong>{r['task_id']}</strong><br><small>{r['name']}</small></td>
<td>{r['n_classes']}</td>
<td>{r['test_examples']}</td>
<td>{r['eval_split']}</td>
<td><strong>{r['accuracy']:.1%}</strong><br><small>{ci}</small></td>
<td>{r['macro_f1']:.1%}</td>
<td>{f'{bl_maj:.1%}' if bl_maj is not None else '—'}</td>
<td>{f'{bl_tfidf:.1%}' if bl_tfidf is not None else '—'}</td>
<td>{f'{jeff_data[r["task_id"]]["jeff"]["accuracy_all"]:.1%}' if r['task_id'] in jeff_data and 'jeff' in jeff_data[r['task_id']] else (f'{jeff_data[r["task_id"]].get("jeff_accuracy",0):.1%}' if r['task_id'] in jeff_data and 'jeff_accuracy' in jeff_data[r['task_id']] else 'not evaluated')}</td>
<td>{lat:.0f}ms<br><small>emb {emb_lat:.0f} + clf {clf_lat:.2f}</small></td>
<td>{artifact_kb} KB</td>
</tr>\n"""

    avg_acc = sum(r["accuracy"] for r in results) / len(results)
    avg_f1 = sum(r["macro_f1"] for r in results) / len(results)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jeffy Benchmark</title>
<style>
:root{{--mono:ui-monospace,SFMono-Regular,Consolas,monospace;--accent:#2563eb}}
*{{box-sizing:border-box}}
body{{margin:0;font:14px/1.6 var(--mono);color:#111;background:#fff;padding:32px}}
.container{{max-width:1400px;margin:auto}}
h1{{font-size:28px;font-weight:700;margin:0 0 4px}}
.sub{{color:#555;font-size:13px;margin-bottom:24px}}
h2{{font-size:18px;margin:32px 0 12px;border-bottom:1px solid #e5e5e5;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{padding:8px 10px;text-align:left;border-bottom:1px solid #eee}}
th{{background:#f8f8f8;font-weight:600;position:sticky;top:0}}
tr:hover{{background:#f0f7ff}}
small{{color:#777}}
strong{{font-weight:600}}
.meta{{font-size:12px;color:#555;margin:16px 0;line-height:1.8}}
.meta dt{{font-weight:600;display:inline}}
.meta dd{{display:inline;margin:0 16px 0 0}}
.note{{background:#fffbeb;border:1px solid #fde68a;border-radius:6px;padding:12px;font-size:12px;margin:16px 0}}
.arch{{background:#f0f7ff;border:1px solid #bfdbfe;border-radius:6px;padding:16px;font-size:13px;margin:16px 0}}
code{{background:#f5f5f5;padding:2px 5px;border-radius:3px;font-size:12px}}
a{{color:var(--accent)}}
</style>
</head>
<body>
<div class="container">
<h1>Jeffy Benchmark</h1>
<p class="sub">Frozen pretrained embeddings + task-specific logistic heads. Evaluated {data['evaluation_date']}.</p>

<div class="arch">
<strong>Architecture:</strong> BAAI/bge-large-en-v1.5 (1024d frozen encoder, ~1.2 GB) → StandardScaler → LogisticRegression (C=0.01, newton-cg)<br>
<strong>Each head:</strong> 1 scaler + 1 linear classifier. No fine-tuning, no MLP, no generation.<br>
<strong>Training regime:</strong> Each head trained on up to 10,000 examples from the dataset's training split (seed=42 for subsampling).<br>
<strong>Comparison context:</strong> Jeffy heads are task-trained; Jeff and similar zero-shot models are not. This is a deployment comparison, not an equal-supervision experiment.<br>
<strong>Baselines:</strong> TF-IDF+LR tuned via 3-fold CV over C in {{0.01, 0.1, 1, 10}}, analyzer in {{word, char_wb}}, ngram_range in {{(1,1), (1,2)}}, max_features in {{10000, 30000}}. Vectorizer fit inside each fold.
</div>

<h2>Results</h2>
<table>
<thead>
<tr>
<th>Task</th><th>Classes</th><th>Test N</th><th>Split</th>
<th>Accuracy</th><th>Macro F1</th>
<th>Majority</th><th>TF-IDF+LR</th><th>Jeff</th>
<th>Latency (p50)</th><th>Head Size</th>
</tr>
</thead>
<tbody>
{rows}
<tr style="font-weight:600;border-top:2px solid #333">
<td>Average (13 tasks)</td><td></td><td></td><td></td>
<td>{avg_acc:.1%}</td><td>{avg_f1:.1%}</td>
<td></td><td></td><td></td><td></td><td></td>
</tr>
</tbody>
</table>

<div class="note">
<strong>Jeff comparison:</strong> Not yet evaluated locally. Jeff is a zero-shot decision model (no per-task training).
A fair comparison requires running Jeff on identical evaluation examples with the same label meanings.
Jeffy's advantage comes from task-specific training; Jeff's advantage is generalization without training data.
</div>

<h2>Task-Specific Notes</h2>
<dl class="meta">
<dt>SST-2:</dt><dd>Evaluated on validation split; official test labels are not public.</dd>
<dt>SMS Spam:</dt><dd>Random train/test split (test_size=0.2, seed=42); no standard benchmark split.</dd>
<dt>SNLI:</dt><dd>Input encoded as "premise [SEP] hypothesis". Label -1 (unlabeled) filtered.</dd>
<dt>CLINC-OOS:</dt><dd>151 classes including out-of-scope. In-scope accuracy {[r for r in results if r['task_id']=='clinc_oos'][0].get('clinc_detail',{}).get('in_scope_acc','?'):.1%}, OOS detection {[r for r in results if r['task_id']=='clinc_oos'][0].get('clinc_detail',{}).get('oos_acc','?'):.1%}.</dd>
<dt>MASSIVE:</dt><dd>English subset only (config='en'). String labels used directly.</dd>
</dl>

<h2>Infrastructure</h2>
<dl class="meta">
<dt>Hardware:</dt><dd>{hw.get('platform','?')}, {hw.get('cpu_count','?')} cores, {hw.get('ram_gb','?')} GB RAM</dd>
<dt>Encoder:</dt><dd>BAAI/bge-large-en-v1.5, ~1.2 GB on disk, {data.get('encoder_load_time_s','?')}s load time</dd>
<dt>Total pack:</dt><dd>{data.get('pack_total_bytes',0)/1024:.0f} KB (13 heads, no encoder)</dd>
<dt>Threads:</dt><dd>Recorded: device={hw.get('device','?')}, torch_threads={hw.get('torch_threads','?')}, OMP_NUM_THREADS={hw.get('omp_threads','?')}</dd>
<dt>Batch size:</dt><dd>1 (single-example inference measured)</dd>
<dt>Caching:</dt><dd>No embedding cache; each request embeds fresh</dd>
<dt>Truncation:</dt><dd>Default sentence-transformers truncation (512 tokens)</dd>
</dl>

<h2>Reproduce</h2>
<pre><code># Install
pip install sentence-transformers scikit-learn fastapi uvicorn

# Build model pack from source datasets
python -m jeffy.build_pack --out data/model_pack

# Run evaluation
python -m jeffy.evaluate --baselines --latency

# Start server with playground
python -m jeffy.server
# Open http://localhost:8400
</code></pre>

<h2>API</h2>
<pre><code># Predict with a pretrained head
curl -X POST http://localhost:8400/v1/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"text": "I was charged twice", "task": "banking77"}}'

# List capabilities
curl http://localhost:8400/v1/capabilities

# Jeff-compatible endpoint (matches criteria to pretrained heads)
curl -X POST http://localhost:8400/v1/systemone \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"jeffy","state":"I was charged twice",
       "questions":{{"intent":{{"type":"choice","criteria":{{
         "transaction_charged_twice":null,"request_refund":null,
         "cancel_transfer":null}}}}}}}}'
</code></pre>

</div>
</body>
</html>"""

    Path(output_path).write_text(html)
    print(f"Benchmark page: {output_path}")


if __name__ == "__main__":
    generate_html()
