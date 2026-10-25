"""Jeffy post experiment runner.

Workflow:
1. Queue up post/comment ideas with metadata
2. Before posting: snapshot baseline metrics
3. Post (manually) and mark as posted
4. Monitor for 2 hours
5. Generate comparison report
6. Over time: build a dataset of what works

Usage:
    python post_experiments.py plan          # show queued experiments
    python post_experiments.py baseline <id> # capture baseline for experiment
    python post_experiments.py posted <id>   # mark as posted, start monitoring
    python post_experiments.py report <id>   # generate report
    python post_experiments.py report-all    # compare all completed experiments
"""

import json
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
EXPERIMENTS_FILE = DATA_DIR / "experiments.json"
METRICS_DIR = DATA_DIR / "metrics"
METRICS_DIR.mkdir(exist_ok=True)


def load_experiments() -> list[dict]:
    if EXPERIMENTS_FILE.exists():
        return json.loads(EXPERIMENTS_FILE.read_text())
    return []


def save_experiments(experiments: list[dict]):
    EXPERIMENTS_FILE.write_text(json.dumps(experiments, indent=2, default=str))


def add_experiment(exp: dict) -> int:
    experiments = load_experiments()
    exp["id"] = len(experiments) + 1
    exp["status"] = "queued"
    exp["created_at"] = datetime.now().isoformat()
    exp["baseline_captured_at"] = None
    exp["posted_at"] = None
    exp["monitoring_until"] = None
    experiments.append(exp)
    save_experiments(experiments)
    return exp["id"]


def capture_baseline(exp_id: int, metrics: dict):
    """Capture pre-post metrics snapshot."""
    experiments = load_experiments()
    exp = next(e for e in experiments if e["id"] == exp_id)
    exp["baseline_captured_at"] = datetime.now().isoformat()
    exp["baseline_metrics"] = metrics
    exp["status"] = "baseline_captured"
    save_experiments(experiments)
    print(f"Baseline captured for experiment #{exp_id}")


def mark_posted(exp_id: int):
    """Mark experiment as posted, start 2-hour monitoring window."""
    experiments = load_experiments()
    exp = next(e for e in experiments if e["id"] == exp_id)
    now = datetime.now()
    exp["posted_at"] = now.isoformat()
    exp["monitoring_until"] = (now + timedelta(hours=2)).isoformat()
    exp["status"] = "monitoring"
    save_experiments(experiments)
    print(f"Experiment #{exp_id} marked as posted at {now.strftime('%H:%M')}")
    print(f"Monitoring until {(now + timedelta(hours=2)).strftime('%H:%M')}")


def capture_post_metrics(exp_id: int, metrics: dict):
    """Capture post-period metrics snapshot."""
    experiments = load_experiments()
    exp = next(e for e in experiments if e["id"] == exp_id)
    exp["post_metrics"] = metrics
    exp["status"] = "completed"
    exp["completed_at"] = datetime.now().isoformat()
    save_experiments(experiments)
    print(f"Post-metrics captured for experiment #{exp_id}")


def generate_report(exp_id: int) -> dict:
    """Generate comparison report for one experiment."""
    experiments = load_experiments()
    exp = next(e for e in experiments if e["id"] == exp_id)

    if "baseline_metrics" not in exp or "post_metrics" not in exp:
        print(f"Experiment #{exp_id} missing baseline or post metrics")
        return {}

    baseline = exp["baseline_metrics"]
    post = exp["post_metrics"]

    report = {
        "id": exp_id,
        "channel": exp.get("channel"),
        "content_type": exp.get("content_type"),
        "post_url": exp.get("post_url"),
        "posted_at": exp.get("posted_at"),
        "deltas": {},
    }

    for key in set(list(baseline.keys()) + list(post.keys())):
        b = baseline.get(key, 0)
        p = post.get(key, 0)
        delta = p - b
        pct = (delta / b * 100) if b > 0 else float("inf") if delta > 0 else 0
        report["deltas"][key] = {"baseline": b, "post": p, "delta": delta, "pct_change": round(pct, 1)}

    return report


def print_plan():
    experiments = load_experiments()
    if not experiments:
        print("No experiments queued. Run seed_experiments() first.")
        return

    for exp in experiments:
        status_icon = {"queued": "⏳", "baseline_captured": "📊", "monitoring": "🔴", "completed": "✅"}.get(exp["status"], "?")
        print(f"  {status_icon} #{exp['id']} [{exp['status']}] {exp['channel']}: {exp.get('content_type', '')} — {exp.get('description', '')[:60]}")


def print_report(exp_id: int):
    report = generate_report(exp_id)
    if not report:
        return

    print(f"\n{'='*60}")
    print(f"Experiment #{report['id']}: {report['channel']} / {report['content_type']}")
    print(f"Posted: {report['posted_at']}")
    if report.get("post_url"):
        print(f"URL: {report['post_url']}")
    print(f"{'='*60}")
    print(f"{'Metric':30s} {'Baseline':>10s} {'Post':>10s} {'Delta':>10s} {'%':>8s}")
    print("-" * 70)
    for key, vals in sorted(report["deltas"].items()):
        sign = "+" if vals["delta"] > 0 else ""
        print(f"  {key:30s} {vals['baseline']:10} {vals['post']:10} {sign}{vals['delta']:9} {sign}{vals['pct_change']:7.1f}%")


def print_report_all():
    experiments = load_experiments()
    completed = [e for e in experiments if e["status"] == "completed"]
    if not completed:
        print("No completed experiments yet.")
        return

    print(f"\n{'='*80}")
    print(f"{'#':>3} {'Channel':>12} {'Type':>15} {'Views Δ':>10} {'Clicks Δ':>10} {'Engagement':>12}")
    print("-" * 80)
    for exp in completed:
        report = generate_report(exp["id"])
        views_d = report["deltas"].get("pageviews", {}).get("delta", "?")
        clicks_d = report["deltas"].get("clicks", {}).get("delta", "?")
        engage_d = report["deltas"].get("engagement", {}).get("delta", "?")
        print(f"  {exp['id']:3} {exp['channel']:>12} {exp.get('content_type',''):>15} {views_d:>10} {clicks_d:>10} {engage_d:>12}")


def seed_experiments():
    """Seed initial experiment queue based on our research."""
    ideas = [
        # HN — insight only (no link)
        {
            "channel": "hn",
            "content_type": "insight_only",
            "target_thread": "https://news.ycombinator.com/item?id=48623434",
            "thread_title": "Good results fine tuning Qwen 3:0.6B to categorize questions",
            "description": "Share benchmark finding: logistic regression on embeddings vs LLMs on many-class classification. No Jeffy link.",
            "talking_points": [
                "Ran 5 local LLMs (gemma4, phi4, mistral, llama3.1, qwen3:32b) on banking77 (77 classes)",
                "Best LLM got 68%, logistic regression on embeddings got 94.3%",
                "LLMs collapse on many-class problems — qwen3:32b got 3%",
                "Latency: 100ms vs 8-17 seconds",
            ],
            "ref_tag": "hn-qwen-finetune-insight",
        },
        # HN — insight + link
        {
            "channel": "hn",
            "content_type": "insight_with_link",
            "target_thread": "https://news.ycombinator.com/item?id=48623434",
            "thread_title": "Good results fine tuning Qwen 3:0.6B to categorize questions",
            "description": "Same thread, but with Jeffy link as the tool used for the benchmark.",
            "talking_points": [
                "Same data as above",
                "Link to Jeffy as the embedding+logreg tool used",
                "pip install jeffy-classify",
            ],
            "ref_tag": "hn-qwen-finetune-link",
        },
        # HN — AI detection story
        {
            "channel": "hn",
            "content_type": "show_hn_story",
            "target_thread": None,
            "thread_title": None,
            "description": "Show HN: My handwritten post got flagged as AI, so I built a tool to understand why",
            "talking_points": [
                "HN flagged my Show HN, moderator confirmed classifier did it",
                "Built nicocheck: authorship classifier + Burrows-Delta + Binoculars + RoBERTa",
                "Key finding: product description register triggers detectors, not AI authorship",
                "Pangram flags my handwritten text as AI, but same style on different topic = 100% human",
                "RoBERTa says 0% AI, Pangram says AI — detectors disagree",
            ],
            "ref_tag": "hn-nicocheck-show",
        },
        # Reddit — r/MachineLearning
        {
            "channel": "reddit",
            "subreddit": "r/MachineLearning",
            "content_type": "insight_only",
            "description": "Post or comment about embedding+logreg vs LLM classification benchmarks",
            "talking_points": [
                "Benchmark table: 5 LLMs vs logistic regression on 3 classification tasks",
                "LLMs struggle with many classes — worse than random on 77-class with qwen3:32b",
                "Discussion: when is an LLM overkill for classification?",
            ],
            "ref_tag": "reddit-ml-benchmark",
        },
        # Reddit — r/LocalLLaMA
        {
            "channel": "reddit",
            "subreddit": "r/LocalLLaMA",
            "content_type": "insight_with_link",
            "description": "Local classification without needing an LLM at all",
            "talking_points": [
                "For structured classification, you might not need an LLM",
                "Embedding model + tiny head = 94% on banking77, 100ms, CPU only",
                "Link to Jeffy + benchmark comparison",
            ],
            "ref_tag": "reddit-localllama-link",
        },
        # Reddit — AI detection false positives
        {
            "channel": "reddit",
            "subreddit": "r/artificial",
            "content_type": "insight_with_link",
            "description": "My handwritten text was flagged as AI by Pangram — built a tool to investigate",
            "talking_points": [
                "Same text, same author: product description = AI, conversational = human",
                "Detectors flag register/structure, not authorship",
                "Screenshots showing the false positive",
                "Link to nicocheck or blog post",
            ],
            "ref_tag": "reddit-artificial-detection",
        },
        # LinkedIn
        {
            "channel": "linkedin",
            "content_type": "insight_with_link",
            "description": "Post about LLM vs classical ML for text classification",
            "talking_points": [
                "Not everything needs an LLM",
                "Benchmark data showing logistic regression beating billion-param models",
                "The right tool for the job",
                "Link to blog post or GitHub",
            ],
            "ref_tag": "linkedin-benchmark",
        },
    ]

    for idea in ideas:
        exp_id = add_experiment(idea)
        print(f"  Created experiment #{exp_id}: {idea['channel']} / {idea['content_type']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python post_experiments.py <command> [args]")
        print("Commands: seed, plan, baseline <id>, posted <id>, post-metrics <id>, report <id>, report-all")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "seed":
        seed_experiments()
    elif cmd == "plan":
        print_plan()
    elif cmd == "baseline":
        exp_id = int(sys.argv[2])
        # In practice, pull from analytics API here
        print(f"Capture baseline metrics for experiment #{exp_id}")
        print("Enter metrics as JSON (e.g. {{\"pageviews\": 45, \"clicks\": 12}}):")
        metrics = json.loads(input("> "))
        capture_baseline(exp_id, metrics)
    elif cmd == "posted":
        exp_id = int(sys.argv[2])
        if len(sys.argv) > 3:
            # Optional: pass the URL of the post
            experiments = load_experiments()
            exp = next(e for e in experiments if e["id"] == exp_id)
            exp["post_url"] = sys.argv[3]
            save_experiments(experiments)
        mark_posted(exp_id)
    elif cmd == "post-metrics":
        exp_id = int(sys.argv[2])
        print(f"Capture post-period metrics for experiment #{exp_id}")
        print("Enter metrics as JSON:")
        metrics = json.loads(input("> "))
        capture_post_metrics(exp_id, metrics)
    elif cmd == "report":
        exp_id = int(sys.argv[2])
        print_report(exp_id)
    elif cmd == "report-all":
        print_report_all()
    else:
        print(f"Unknown command: {cmd}")
