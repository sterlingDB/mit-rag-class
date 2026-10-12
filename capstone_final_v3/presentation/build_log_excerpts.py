"""Format saved slide evidence for reading; makes no LLM or network calls."""
import html
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read_events(relative_path):
    text = (ROOT / relative_path).read_text()
    events = []
    for match in re.finditer(r"^\[([^\n]+)\] (\w+)\n", text, re.MULTILINE):
        if match[2] not in ("AGENT_STEP", "EFFICIENT_RESULT"):
            continue
        data, _ = json.JSONDecoder().raw_decode(text[match.end():])
        events.append({
            "timestamp": match[1], "type": match[2], "data": data,
            "line": text.count("\n", 0, match.start()) + 1,
        })
    return events


def make_excerpt(record, events):
    question = record["question"]
    steps = [event for event in events
             if event["type"] == "AGENT_STEP"
             and event["data"]["question"] == question]
    result = next(event for event in events
                  if event["type"] == "EFFICIENT_RESULT"
                  and event["data"]["question"] == question)
    assert result["data"]["answer"] == record["answer"]
    lines = [f"[question] {question}"]
    for event in steps:
        step = event["data"]
        if step.get("evidence_check", {}).get("action") == "replan":
            lines.append("    evidence missing; asking the planner for a focused next action")
        lines.append(f"  step {step['step']}: action={step['action']}  ({step['reasoning'][:60]})")
        lines.append(f"    collected articles: {sorted(step['collected_article_ids'])}")
    lines.extend([f"  stopped: {record['stop_reason']}", f"Path: {record['path']}",
                  "", "Answer:", record["answer"]])
    return "\n".join(lines), steps, result


def main():
    latest = json.loads((ROOT / "docs/latest-run-results.json").read_text())
    source = latest["source"]
    events = read_events(source)
    records = {record["case"]: record for record in latest["records"]}
    cases = [
        (6, "Jeremiyah Love draft pick", records["draft_pick"]),
        (7, "Arizona draft comparison", records["arizona_comparison"]),
        (8, "The Dukes of Hazzard", records["dukes_capital"]),
        (9, "Brady Bunch house", records["brady_house"]),
    ]
    cards, text_blocks, links = [], [], []
    for slide, title, record in cases:
        excerpt, steps, result = make_excerpt(record, events)
        label = latest["run_label"]
        settings = "1,500,000-character budget; no per-article cap; separate empty answer cache per question"
        references = [f"{source}:{event['line']} — {event['type']} ({event['timestamp']})"
                      for event in [*steps, result]]
        metrics = (f"Latency: {record['elapsed_seconds']:.2f} s | Model calls: {record['model_calls']} | "
                   f"Chat tokens: {record['reported_chat_tokens']:,} | "
                   f"Estimated chat cost: ${record['estimated_chat_cost_usd']:.5f}")
        cards.append(f'''<section id="slide-{slide}">
<div class="label">SLIDE {slide} · {html.escape(label)}</div>
<h2>{html.escape(title)}</h2><p class="settings">{settings}</p>
<pre class="console">{html.escape(excerpt)}</pre>
<p class="metrics">{html.escape(metrics)}</p>
<p class="small">Latency excludes setup. Chat tokens total all model calls; cost excludes embeddings.</p>
<details><summary>Source files and line numbers</summary>
<pre>{html.escape(chr(10).join(references))}</pre>
<p>Console formatting: capstone.py:603, 661, 686, 704, 950, 955; security_cost_helpers.py:99.</p>
</details></section>''')
        links.append(f'<a href="#slide-{slide}">{slide} · {html.escape(title)}</a>')
        text_blocks.append(f"SLIDE {slide}: {title}\n{label}\n{settings}\n\n{excerpt}\n\n{metrics}\n\nSources:\n" + "\n".join(references))
    notice = ("Console-style excerpts reconstructed from saved AGENT_STEP and EFFICIENT_RESULT records. "
              "Questions and answers are verbatim; action lines follow the program’s console formatting. "
              "This view is not a captured stdout file. Generating it makes no new LLM requests.")
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Capstone — Log excerpts supporting the slides</title>
<style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f5f3ed;color:#182a35;font:16px/1.55 system-ui,sans-serif}
header,main{max-width:1120px;margin:auto;padding:24px 32px}header{padding-bottom:8px}
h1{font-size:30px;margin:8px 0}h2{font-size:25px;margin:6px 0}nav{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}
a,button{font:inherit;color:#19594f}nav a,button{padding:7px 12px;border:1px solid #a8b9b4;border-radius:6px;background:white;text-decoration:none;cursor:pointer}
section{background:white;border:1px solid #d7ddd9;border-radius:10px;padding:24px;margin-bottom:28px;scroll-margin-top:20px}
.label{font-size:13px;font-weight:700;letter-spacing:.06em;color:#21695e}.settings,.small{font-size:14px;color:#51626c}
pre{font:14px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere}
.console{background:#16242e;color:#e9f1ee;border-radius:8px;padding:22px}.metrics{font-weight:650;font-size:15px}
summary{cursor:pointer;color:#19594f}details{border-top:1px solid #d7ddd9;padding-top:12px}.notice{max-width:950px;font-size:14px}
@media(max-width:600px){header,main{padding:16px}section{padding:16px}.console{padding:14px}}
@media print{button,nav{display:none}section{break-before:page}.console{background:white;color:black;border:1px solid #ddd}details{display:block}}
</style><header><h1>Logs supporting the slides</h1>
<p class="notice">NOTICE</p>
<button id="separate">Open separate window</button> <a href="log-excerpts.txt" download>Download text</a>
<nav aria-label="Slide examples">LINKS</nav></header><main>CARDS</main>
<script>document.getElementById('separate').onclick=()=>{window.open(location.href,'capstone-log-excerpts','popup,width=1120,height=900,resizable=yes,scrollbars=yes');};</script></html>'''
    page = page.replace("NOTICE", html.escape(notice)).replace("LINKS", "".join(links)).replace("CARDS", "".join(cards))
    (HERE / "log-excerpts.html").write_text(page)
    (HERE / "log-excerpts.txt").write_text(notice + "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(text_blocks) + "\n")
    print("Created log-excerpts.html and log-excerpts.txt from four saved results.")


if __name__ == "__main__":
    main()
