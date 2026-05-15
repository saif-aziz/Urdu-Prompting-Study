"""
Generates dashboard.html — a self-contained interactive frontend.
Run from the project root:   python make_dashboard.py
Then open dashboard.html in any browser (no server needed).
"""
import json, csv, pathlib, html as html_mod

ROOT = pathlib.Path(".")
PRED = ROOT / "results" / "predictions"

# ── Load metrics ─────────────────────────────────────────────────────────────
metrics = []
with open(ROOT / "results" / "metrics.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        metrics.append({
            "model":   row["model"],
            "prompt":  row["prompt"],
            "acc":     round(float(row["accuracy"]), 3),
            "f1":      round(float(row["macro_f1"]), 3),
            "unk":     round(float(row["unknown_rate"]) * 100, 1),
            "f1_neg":  round(float(row["f1_negative"]), 3),
            "f1_neu":  round(float(row["f1_neutral"]), 3),
            "f1_pos":  round(float(row["f1_positive"]), 3),
        })

# ── Load significance ─────────────────────────────────────────────────────────
sig_rows = []
with open(ROOT / "results" / "significance.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        sig_rows.append({
            "model":   row["model"],
            "setting": row["setting"],
            "b":       int(row["b"]),
            "c":       int(row["c"]),
            "p":       float(row["p_value"]),
        })

# ── Load predictions (all 12 cells) ──────────────────────────────────────────
VARIANTS = ["zs_en","zs_ur","fs_en","fs_ur","cot_en","cot_ur"]
MODELS   = ["qwen05","qwen15"]

predictions = {}
for m in MODELS:
    for v in VARIANTS:
        path = PRED / f"{m}_{v}.jsonl"
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").strip().splitlines()]
        predictions[f"{m}|{v}"] = rows

# Prompt templates (reconstructed)
PROMPTS = {
    "zs_en":  ("You are a precise sentiment classifier for Urdu text. Respond with exactly one word from: negative, neutral, positive.",
               "Classify the sentiment of the following Urdu text. Answer with one word: negative, neutral, or positive.\n\nText: {text}\nAnswer:"),
    "zs_ur":  ("آپ ایک درست اردو جذبات کی درجہ بندی کرنے والے ہیں۔ صرف ایک لفظ سے جواب دیں: منفی، غیر جانبدار، یا مثبت۔",
               "درج ذیل اردو متن کے جذبات کی درجہ بندی کریں۔\n\nمتن: {text}\nجواب:"),
    "fs_en":  ("You are a precise sentiment classifier for Urdu text. Respond with exactly one word from: negative, neutral, positive.",
               "Classify the sentiment of the following Urdu text.\n\n[3 class-balanced examples from training set]\n\nText: {text}\nAnswer:"),
    "fs_ur":  ("آپ ایک درست اردو جذبات کی درجہ بندی کرنے والے ہیں۔ صرف ایک لفظ سے جواب دیں۔",
               "درج ذیل اردو متن کے جذبات کی درجہ بندی کریں۔\n\n[3 تربیتی مثالیں]\n\nمتن: {text}\nجواب:"),
    "cot_en": ("You are a precise sentiment classifier for Urdu text.",
               "Classify the sentiment of the following Urdu text. First briefly reason step by step in English, then write 'Answer: <label>'.\n\nText: {text}"),
    "cot_ur": ("آپ ایک درست اردو جذبات کی درجہ بندی کرنے والے ہیں۔",
               "درج ذیل اردو متن کے جذبات کی درجہ بندی کریں۔ پہلے قدم بہ قدم اردو میں سوچیں، پھر 'جواب: <لیبل>' لکھیں۔\n\nمتن: {text}"),
}

# ── Embed data as JS ─────────────────────────────────────────────────────────
metrics_js  = json.dumps(metrics,      ensure_ascii=False)
sig_js      = json.dumps(sig_rows,     ensure_ascii=False)
preds_js    = json.dumps(predictions,  ensure_ascii=False)
prompts_js  = json.dumps(PROMPTS,      ensure_ascii=False)

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Urdu Prompting Study — Dashboard</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#222}}
  header{{background:#1a237e;color:white;padding:20px 32px}}
  header h1{{font-size:1.5rem;font-weight:700}}
  header p{{font-size:.9rem;opacity:.8;margin-top:4px}}
  .tabs{{display:flex;background:#283593;padding:0 24px}}
  .tab{{padding:12px 20px;cursor:pointer;color:#9fa8da;font-size:.9rem;border-bottom:3px solid transparent;transition:.2s}}
  .tab.active,.tab:hover{{color:white;border-bottom-color:#5c6bc0}}
  .pane{{display:none;padding:24px;max-width:1200px;margin:0 auto}}
  .pane.active{{display:block}}
  /* cards */
  .card{{background:white;border-radius:10px;padding:20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
  .card h2{{font-size:1rem;color:#1a237e;margin-bottom:14px;border-bottom:2px solid #e8eaf6;padding-bottom:8px}}
  /* table */
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  th{{background:#e8eaf6;color:#1a237e;padding:8px 10px;text-align:left}}
  td{{padding:8px 10px;border-bottom:1px solid #f0f0f0}}
  tr:hover td{{background:#f5f5f5}}
  .green{{background:#c6efce;color:#276221;font-weight:700;border-radius:4px;padding:2px 6px}}
  .yellow{{background:#ffeb9c;color:#9c5700;font-weight:700;border-radius:4px;padding:2px 6px}}
  .red{{background:#ffc7ce;color:#9c0006;font-weight:700;border-radius:4px;padding:2px 6px}}
  /* controls */
  .controls{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;align-items:flex-end}}
  .controls label{{font-size:.8rem;color:#555;display:block;margin-bottom:4px}}
  select,input[type=number]{{padding:6px 10px;border:1px solid #ccc;border-radius:6px;font-size:.9rem;width:100%}}
  button{{background:#1a237e;color:white;border:none;padding:8px 18px;border-radius:6px;cursor:pointer;font-size:.9rem}}
  button:hover{{background:#283593}}
  /* prompt box */
  .prompt-box{{background:#f5f5f5;border-radius:6px;padding:12px;font-family:monospace;
               font-size:.82rem;white-space:pre-wrap;border-left:4px solid #5c6bc0;margin:8px 0}}
  .output-box{{background:#fff9c4;border-radius:6px;padding:12px;font-family:monospace;
               font-size:.82rem;white-space:pre-wrap;border-left:4px solid #f9a825;margin:8px 0}}
  .badge{{display:inline-block;padding:4px 12px;border-radius:20px;font-weight:700;font-size:.85rem}}
  .badge.ok{{background:#c6efce;color:#276221}}
  .badge.wrong{{background:#ffc7ce;color:#9c0006}}
  /* distribution bars */
  .dist-bar{{display:flex;align-items:center;margin:6px 0;font-size:.83rem}}
  .dist-bar .label{{width:80px;color:#555}}
  .dist-bar .bar{{height:18px;border-radius:3px;margin:0 8px;min-width:2px;transition:.4s}}
  .dist-bar .count{{color:#333;font-weight:600}}
  .neg-bar{{background:#e57373}}
  .neu-bar{{background:#64b5f6}}
  .pos-bar{{background:#81c784}}
  .unk-bar{{background:#bdbdbd}}
  /* sig table */
  .sig-ok{{background:#c6efce;color:#276221;font-weight:700;padding:2px 8px;border-radius:4px}}
  .sig-no{{background:#ffc7ce;color:#9c0006;font-weight:700;padding:2px 8px;border-radius:4px}}
  .side-by-side{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  @media(max-width:700px){{.side-by-side{{grid-template-columns:1fr}}}}
  .col-en{{border-top:4px solid #1565c0}}
  .col-ur{{border-top:4px solid #6a1b9a}}
  .col-header{{padding:8px;font-weight:700;font-size:.85rem;color:white;border-radius:4px 4px 0 0}}
  .col-en .col-header{{background:#1565c0}}
  .col-ur .col-header{{background:#6a1b9a}}
</style>
</head>
<body>

<header>
  <h1>Urdu Prompting Study — Results Dashboard</h1>
  <p>CS4063 NLP &nbsp;|&nbsp; Roman-Urdu Sentiment Classification &nbsp;|&nbsp; 498 test examples &nbsp;|&nbsp; 12 experiment cells</p>
</header>

<div class="tabs">
  <div class="tab active" onclick="showTab('summary')">📊 Results Summary</div>
  <div class="tab" onclick="showTab('explorer')">🔍 Example Explorer</div>
  <div class="tab" onclick="showTab('collapse')">📉 Label Collapse</div>
  <div class="tab" onclick="showTab('significance')">📐 Significance Tests</div>
</div>

<!-- ═══ TAB 1: SUMMARY ═══════════════════════════════════════════════════ -->
<div id="tab-summary" class="pane active">
  <div class="card">
    <h2>Macro-F1 across all 12 cells</h2>
    <p style="font-size:.82rem;color:#777;margin-bottom:12px">
      Green ≥ 0.5 &nbsp;|&nbsp; Yellow ≥ 0.3 &nbsp;|&nbsp; Red &lt; 0.3 &nbsp;|&nbsp; Bold = best per model
    </p>
    <table id="metrics-table"></table>
  </div>
  <div class="card">
    <h2>Per-class F1 breakdown</h2>
    <table id="perclass-table"></table>
  </div>
</div>

<!-- ═══ TAB 2: EXPLORER ══════════════════════════════════════════════════ -->
<div id="tab-explorer" class="pane">
  <div class="card">
    <h2>Pick an example to inspect</h2>
    <div class="controls">
      <div style="flex:1;min-width:120px">
        <label>Model</label>
        <select id="sel-model">
          <option value="qwen05">Qwen 0.5B</option>
          <option value="qwen15">Qwen 1.5B</option>
        </select>
      </div>
      <div style="flex:1;min-width:160px">
        <label>Prompt variant</label>
        <select id="sel-variant">
          <option value="zs_en">zs_en — zero-shot English</option>
          <option value="zs_ur">zs_ur — zero-shot Urdu</option>
          <option value="fs_en" selected>fs_en — few-shot English</option>
          <option value="fs_ur">fs_ur — few-shot Urdu</option>
          <option value="cot_en">cot_en — CoT English</option>
          <option value="cot_ur">cot_ur — CoT Urdu</option>
        </select>
      </div>
      <div style="flex:0;min-width:100px">
        <label>Example # (0–497)</label>
        <input type="number" id="sel-idx" value="0" min="0" max="497">
      </div>
      <div><button onclick="showExample()">Show</button></div>
      <div><button onclick="randomExample()" style="background:#37474f">Random</button></div>
    </div>
    <div id="example-out"></div>
  </div>
  <div class="card">
    <h2>Side-by-side: same input, English vs Urdu prompt</h2>
    <div class="controls">
      <div style="flex:1;min-width:120px">
        <label>Model</label>
        <select id="sbs-model">
          <option value="qwen05">Qwen 0.5B</option>
          <option value="qwen15">Qwen 1.5B</option>
        </select>
      </div>
      <div style="flex:1;min-width:120px">
        <label>Regime</label>
        <select id="sbs-regime">
          <option value="zs">zero-shot</option>
          <option value="fs">few-shot</option>
          <option value="cot">chain-of-thought</option>
        </select>
      </div>
      <div style="flex:0;min-width:100px">
        <label>Example # (0–497)</label>
        <input type="number" id="sbs-idx" value="0" min="0" max="497">
      </div>
      <div><button onclick="showSideBySide()">Compare</button></div>
    </div>
    <div id="sbs-out"></div>
  </div>
</div>

<!-- ═══ TAB 3: COLLAPSE ══════════════════════════════════════════════════ -->
<div id="tab-collapse" class="pane">
  <div class="card">
    <h2>Prediction distribution per cell</h2>
    <p style="font-size:.82rem;color:#777;margin-bottom:16px">
      Urdu prompts collapse to a single label. English prompts spread across all three classes.
    </p>
    <div id="collapse-out"></div>
  </div>
</div>

<!-- ═══ TAB 4: SIGNIFICANCE ═════════════════════════════════════════════ -->
<div id="tab-significance" class="pane">
  <div class="card">
    <h2>Paired McNemar Tests — English vs Urdu</h2>
    <p style="font-size:.82rem;color:#777;margin-bottom:12px">
      <b>b</b> = EN correct &amp; UR wrong &nbsp;|&nbsp;
      <b>c</b> = UR correct &amp; EN wrong &nbsp;|&nbsp;
      b &gt; c means English is better
    </p>
    <table id="sig-table"></table>
  </div>
  <div class="card">
    <h2>What this means</h2>
    <p style="font-size:.88rem;line-height:1.7">
      A <b>significant result</b> (p &lt; 0.05) means the difference between English and Urdu prompts
      is statistically real — not due to chance. All 4 significant results favour English.
      The 2 non-significant pairs (qwen05 few-shot, qwen15 CoT) are cases where
      <em>both</em> conditions fail badly, just in different ways — not genuine parity.
    </p>
  </div>
</div>

<script>
const METRICS  = {metrics_js};
const SIG      = {sig_js};
const PREDS    = {preds_js};
const PROMPTS  = {prompts_js};

// ── Utilities ───────────────────────────────────────────────────────────────
function showTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  event.target.classList.add('active');
}}

function f1Class(v) {{
  if (v >= 0.5) return 'green';
  if (v >= 0.3) return 'yellow';
  return 'red';
}}

function sigLabel(p) {{
  if (p < 0.001) return '*** p<0.001';
  if (p < 0.01)  return '** p<0.01';
  if (p < 0.05)  return '* p<0.05';
  return 'n.s.';
}}

function esc(s) {{
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ── Tab 1: Metrics table ────────────────────────────────────────────────────
function buildMetrics() {{
  // Find best F1 per model group
  const bestF1 = {{}};
  METRICS.forEach(r => {{
    if (r.prompt === '-') return;
    if (!bestF1[r.model] || r.f1 > bestF1[r.model]) bestF1[r.model] = r.f1;
  }});

  let html = `<tr><th>Model</th><th>Prompt</th><th>Accuracy</th><th>Macro-F1</th><th>Unknown %</th></tr>`;
  METRICS.forEach(r => {{
    const cls   = f1Class(r.f1);
    const bold  = (r.f1 === bestF1[r.model]) ? 'font-weight:700' : '';
    const model = r.model === 'majority' ? 'Majority baseline'
                : r.model === 'xlm-roberta-base' ? 'XLM-RoBERTa (fine-tuned)'
                : r.model === 'qwen05' ? 'Qwen 0.5B' : 'Qwen 1.5B';
    html += `<tr>
      <td style="${{bold}}">${{esc(model)}}</td>
      <td>${{esc(r.prompt)}}</td>
      <td>${{r.acc.toFixed(3)}}</td>
      <td><span class="${{cls}}">${{r.f1.toFixed(3)}}</span></td>
      <td>${{r.unk > 0 ? '<span style="color:#e65100">'+r.unk+'%</span>' : r.unk+'%'}}</td>
    </tr>`;
  }});
  document.getElementById('metrics-table').innerHTML = html;

  let html2 = `<tr><th>Model</th><th>Prompt</th><th>F1 Negative</th><th>F1 Neutral</th><th>F1 Positive</th></tr>`;
  METRICS.filter(r => r.prompt !== '-').forEach(r => {{
    const model = r.model === 'qwen05' ? 'Qwen 0.5B' : 'Qwen 1.5B';
    html2 += `<tr>
      <td>${{esc(model)}}</td><td>${{esc(r.prompt)}}</td>
      <td><span class="${{f1Class(r.f1_neg)}}">${{r.f1_neg.toFixed(3)}}</span></td>
      <td><span class="${{f1Class(r.f1_neu)}}">${{r.f1_neu.toFixed(3)}}</span></td>
      <td><span class="${{f1Class(r.f1_pos)}}">${{r.f1_pos.toFixed(3)}}</span></td>
    </tr>`;
  }});
  document.getElementById('perclass-table').innerHTML = html2;
}}

// ── Tab 2: Example explorer ─────────────────────────────────────────────────
function showExample() {{
  const model   = document.getElementById('sel-model').value;
  const variant = document.getElementById('sel-variant').value;
  const idx     = parseInt(document.getElementById('sel-idx').value);
  const key     = model + '|' + variant;
  const rows    = PREDS[key];
  if (!rows || idx >= rows.length) {{ alert('No data for that combination.'); return; }}
  const ex      = rows[idx];
  const tmpl    = PROMPTS[variant];
  const sys     = tmpl[0];
  const usr     = tmpl[1].replace('{{text}}', ex.text);
  const ok      = ex.pred === ex.gold;
  const modelLabel = model === 'qwen05' ? 'Qwen 0.5B' : 'Qwen 1.5B';

  document.getElementById('example-out').innerHTML = `
    <div style="background:#e8eaf6;padding:10px;border-radius:6px;margin:8px 0">
      <b>Input (Roman-Urdu):</b> ${{esc(ex.text)}}
    </div>
    <p style="font-size:.82rem;font-weight:600;margin:8px 0 4px">SYSTEM PROMPT</p>
    <div class="prompt-box">${{esc(sys)}}</div>
    <p style="font-size:.82rem;font-weight:600;margin:8px 0 4px">USER PROMPT</p>
    <div class="prompt-box">${{esc(usr)}}</div>
    <p style="font-size:.82rem;font-weight:600;margin:8px 0 4px">MODEL RAW OUTPUT</p>
    <div class="output-box">${{esc(ex.raw)}}</div>
    <div style="display:flex;gap:12px;margin-top:10px;align-items:center">
      <span style="background:#e3f2fd;padding:5px 12px;border-radius:6px"><b>Gold:</b> ${{esc(ex.gold)}}</span>
      <span style="background:#e3f2fd;padding:5px 12px;border-radius:6px"><b>Predicted:</b> ${{esc(ex.pred)}}</span>
      <span class="badge ${{ok ? 'ok' : 'wrong'}}">${{ok ? '✓ CORRECT' : '✗ WRONG'}}</span>
    </div>`;
}}

function randomExample() {{
  document.getElementById('sel-idx').value = Math.floor(Math.random() * 498);
  showExample();
}}

function showSideBySide() {{
  const model  = document.getElementById('sbs-model').value;
  const regime = document.getElementById('sbs-regime').value;
  const idx    = parseInt(document.getElementById('sbs-idx').value);
  const enKey  = model + '|' + regime + '_en';
  const urKey  = model + '|' + regime + '_ur';
  const en     = PREDS[enKey][idx];
  const ur     = PREDS[urKey][idx];
  const okEn   = en.pred === en.gold;
  const okUr   = ur.pred === ur.gold;
  const modelLabel = model === 'qwen05' ? 'Qwen 0.5B' : 'Qwen 1.5B';

  document.getElementById('sbs-out').innerHTML = `
    <div style="background:#e8eaf6;padding:10px;border-radius:6px;margin-bottom:12px">
      <b>Input:</b> ${{esc(en.text)}} &nbsp;|&nbsp; <b>Gold:</b> <b>${{esc(en.gold)}}</b>
    </div>
    <div class="side-by-side">
      <div class="card col-en" style="margin:0">
        <div class="col-header">English Prompt (${{regime}}_en)</div>
        <div class="output-box" style="margin-top:8px">${{esc(en.raw.substring(0,600))}}</div>
        <span class="badge ${{okEn ? 'ok' : 'wrong'}}" style="margin-top:8px;display:inline-block">
          Pred: ${{esc(en.pred)}} ${{okEn ? '✓' : '✗'}}
        </span>
      </div>
      <div class="card col-ur" style="margin:0">
        <div class="col-header">Urdu Prompt (${{regime}}_ur)</div>
        <div class="output-box" style="margin-top:8px">${{esc(ur.raw.substring(0,600))}}</div>
        <span class="badge ${{okUr ? 'ok' : 'wrong'}}" style="margin-top:8px;display:inline-block">
          Pred: ${{esc(ur.pred)}} ${{okUr ? '✓' : '✗'}}
        </span>
      </div>
    </div>`;
}}

// ── Tab 3: Label collapse ────────────────────────────────────────────────────
function buildCollapse() {{
  const CELLS = [
    ['qwen05','zs_en'],['qwen05','zs_ur'],
    ['qwen05','fs_en'],['qwen05','fs_ur'],
    ['qwen05','cot_en'],['qwen05','cot_ur'],
    ['qwen15','zs_en'],['qwen15','zs_ur'],
    ['qwen15','fs_en'],['qwen15','fs_ur'],
    ['qwen15','cot_en'],['qwen15','cot_ur'],
  ];
  let html = '';
  CELLS.forEach(([m, v]) => {{
    const rows  = PREDS[m + '|' + v];
    const total = rows.length;
    const counts = {{negative:0, neutral:0, positive:0, unknown:0}};
    rows.forEach(r => {{ counts[r.pred] = (counts[r.pred]||0) + 1; }});
    const modelLabel = m === 'qwen05' ? 'Qwen 0.5B' : 'Qwen 1.5B';
    const isUrdu  = v.endsWith('_ur');
    const correct = rows.filter(r => r.pred === r.gold).length;
    const f1      = METRICS.find(x => x.model === m && x.prompt === v);

    html += `<div style="margin-bottom:20px;padding:12px;background:${{isUrdu?'#fce4ec':'#e8eaf6'}};border-radius:8px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
        <b>${{modelLabel}} — ${{v}}</b>
        <span class="${{f1Class(f1?f1.f1:0)}}" style="font-size:.82rem">${{f1?f1.f1.toFixed(3):'-'}} F1</span>
      </div>`;
    ['negative','neutral','positive','unknown'].forEach(label => {{
      const n    = counts[label] || 0;
      const pct  = (n / total * 100).toFixed(0);
      const barcls = label === 'negative' ? 'neg-bar'
                   : label === 'neutral'  ? 'neu-bar'
                   : label === 'positive' ? 'pos-bar' : 'unk-bar';
      const width = Math.max(n / total * 300, n > 0 ? 4 : 0);
      html += `<div class="dist-bar">
        <span class="label">${{label}}</span>
        <div class="bar ${{barcls}}" style="width:${{width}}px"></div>
        <span class="count">${{n}} (${{pct}}%)</span>
      </div>`;
    }});
    html += `</div>`;
  }});
  document.getElementById('collapse-out').innerHTML = html;
}}

// ── Tab 4: Significance ─────────────────────────────────────────────────────
function buildSig() {{
  let html = `<tr><th>Model</th><th>Regime</th><th>b (EN better)</th><th>c (UR better)</th><th>p-value</th><th>Result</th></tr>`;
  SIG.forEach(r => {{
    const label = sigLabel(r.p);
    const isSig = r.p < 0.05;
    const modelLabel = r.model === 'qwen05' ? 'Qwen 0.5B' : 'Qwen 1.5B';
    html += `<tr>
      <td>${{esc(modelLabel)}}</td>
      <td>${{esc(r.setting)}}</td>
      <td style="font-weight:700">${{r.b}}</td>
      <td>${{r.c}}</td>
      <td>${{r.p < 0.0001 ? r.p.toExponential(2) : r.p.toFixed(4)}}</td>
      <td><span class="${{isSig ? 'sig-ok' : 'sig-no'}}">${{label}}</span></td>
    </tr>`;
  }});
  document.getElementById('sig-table').innerHTML = html;
}}

// ── Init ─────────────────────────────────────────────────────────────────────
buildMetrics();
buildCollapse();
buildSig();
showExample();
</script>
</body>
</html>"""

out = ROOT / "dashboard.html"
out.write_text(HTML, encoding="utf-8")
print(f"Done! Open this file in your browser:")
print(f"  {out.resolve()}")
