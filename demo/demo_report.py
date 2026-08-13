"""Demo: Tellurium multi-configuration simulation report.

Runs three distinct SBML/Antimony models (Lotka-Volterra predator-prey,
Repressilator gene oscillator, stochastic dimerization), each wrapped as
a TelluriumProcess. Generates a self-contained HTML report with Plotly
time-series + phase-portrait charts, colored bigraph-viz architecture
diagrams, and interactive PBG composite-document trees.
"""

import base64
import json
import os
import subprocess
import tempfile
import time as _time
import numpy as np

from process_bigraph import allocate_core
from viva_tellurium.processes import TelluriumProcess
from viva_tellurium.composites import make_tellurium_document


# ── Models ──────────────────────────────────────────────────────────

LOTKA_VOLTERRA = """
model lotka
  // Predator-prey oscillator
  P = 10; W = 5
  J1:    -> P;  kg*P
  J2: P  -> W;  kc*P*W
  J3: W  -> ;   kd*W
  kg = 1.0; kc = 0.1; kd = 1.0
end
"""

REPRESSILATOR = """
model repressilator
  // Elowitz & Leibler 2000 ring oscillator: three mutually repressing genes
  m1 = 0; m2 = 0; m3 = 0
  p1 = 5; p2 = 0; p3 = 15

  // mRNA production with Hill-type repression + basal rate
  Rm1: -> m1; alpha * (K^n / (K^n + p3^n)) + alpha0
  Rm2: -> m2; alpha * (K^n / (K^n + p1^n)) + alpha0
  Rm3: -> m3; alpha * (K^n / (K^n + p2^n)) + alpha0

  // mRNA degradation
  Dm1: m1 -> ; beta_m * m1
  Dm2: m2 -> ; beta_m * m2
  Dm3: m3 -> ; beta_m * m3

  // Protein production and degradation
  Rp1: -> p1; beta_p * m1
  Rp2: -> p2; beta_p * m2
  Rp3: -> p3; beta_p * m3
  Dp1: p1 -> ; beta_p * p1
  Dp2: p2 -> ; beta_p * p2
  Dp3: p3 -> ; beta_p * p3

  // Oscillatory regime: n=3 cooperativity pushes past the Hopf
  // bifurcation for alpha=216, K=40 (Elowitz-Leibler parameters).
  alpha = 216; alpha0 = 0.2; K = 40; n = 3
  beta_m = 1.0; beta_p = 0.2
end
"""

DIMERIZATION = """
model dimer
  // Reversible dimerization — stochastic-friendly small system
  M = 80; D = 0
  Jf: 2 M -> D; kf*M*(M-1)/2
  Jr:   D -> 2 M; kr*D
  kf = 0.01; kr = 0.1
end
"""


# ── Config list ─────────────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'lotka',
        'title': 'Lotka-Volterra Oscillator',
        'subtitle': 'Predator-prey dynamics with sustained oscillations',
        'description': (
            'A classic two-species oscillator: prey (P) grows exponentially, '
            'predators (W) consume prey, and predators die off without food. '
            'The coupled ODEs produce stable closed orbits in the (P, W) plane. '
            'A canonical test for ODE integrators and phase-space visualization.'
        ),
        'model': LOTKA_VOLTERRA,
        'integrator': 'cvode',
        'total_time': 30.0,
        'n_snapshots': 300,
        'phase_species': ('P', 'W'),
        'color_scheme': 'indigo',
    },
    {
        'id': 'repress',
        'title': 'Repressilator Gene Circuit',
        'subtitle': 'Three-gene ring oscillator (Elowitz & Leibler 2000)',
        'description': (
            'A synthetic gene network of three mutually repressing genes '
            'arranged in a ring. Each protein represses the next gene via '
            'Hill kinetics, producing limit-cycle oscillations in all three '
            'protein species with 120° phase shifts. Demonstrates bigraph '
            'wiring of a multi-species biochemical network.'
        ),
        'model': REPRESSILATOR,
        'integrator': 'cvode',
        'total_time': 400.0,
        'n_snapshots': 400,
        'phase_species': ('p1', 'p2'),
        'color_scheme': 'emerald',
    },
    {
        'id': 'dimer',
        'title': 'Stochastic Dimerization',
        'subtitle': 'Gillespie SSA trajectory of M + M ⇌ D',
        'description': (
            'Reversible dimerization of a small monomer pool simulated with '
            'the Gillespie stochastic algorithm. The noisy trajectory reveals '
            'fluctuations around the equilibrium that the deterministic ODE '
            'would smooth over. Shows integrator selection and stochastic '
            'simulation support in the wrapper.'
        ),
        'model': DIMERIZATION,
        'integrator': 'gillespie',
        'total_time': 50.0,
        'n_snapshots': 500,
        'phase_species': ('M', 'D'),
        'color_scheme': 'rose',
        'seed': 7,
    },
]


# ── Simulation runner ───────────────────────────────────────────────

def run_simulation(cfg):
    """Run a TelluriumProcess stepwise and collect snapshots + runtime."""
    core = allocate_core()
    core.register_link('TelluriumProcess', TelluriumProcess)

    t0 = _time.perf_counter()

    proc_cfg = {
        'model': cfg['model'],
        'integrator': cfg['integrator'],
    }
    if cfg.get('seed', -1) >= 0:
        proc_cfg['seed'] = cfg['seed']

    proc = TelluriumProcess(config=proc_cfg, core=core)
    state0 = proc.initial_state()

    species_ids = proc.get_species_ids()
    reaction_ids = proc.get_reaction_ids()

    interval = cfg['total_time'] / cfg['n_snapshots']
    snapshots = [{
        'time': state0['time'],
        'species': dict(state0['species']),
        'rates': dict(state0['rates']),
    }]

    for _i in range(cfg['n_snapshots']):
        result = proc.update({}, interval=interval)
        snapshots.append({
            'time': result['time'],
            'species': dict(result['species']),
            'rates': dict(result['rates']),
        })

    runtime = _time.perf_counter() - t0
    return species_ids, reaction_ids, snapshots, runtime


def generate_bigraph_image(cfg, species_ids):
    """Render a colored bigraph-viz PNG (as base64) for the document."""
    from bigraph_viz import plot_bigraph

    # Simplified doc: show only a few species as output ports to keep
    # the diagram readable.
    sample_species = species_ids[:3]
    outputs = {
        sid: ['stores', 'species', sid] for sid in sample_species
    }
    outputs['time'] = ['stores', 'time']

    doc = {
        'tellurium': {
            '_type': 'process',
            'address': 'local:TelluriumProcess',
            'config': {'integrator': cfg['integrator']},
            'inputs': {},
            'outputs': outputs,
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'inputs': {
                **{sid: ['stores', 'species', sid] for sid in sample_species},
                'time': ['global_time'],
            },
        },
    }

    node_colors = {
        ('tellurium',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg):
    """Build the PBG composite document for the JSON tree viewer."""
    return make_tellurium_document(
        model=cfg['model'],
        integrator=cfg['integrator'],
        interval=cfg['total_time'] / cfg['n_snapshots'],
    )


COLOR_SCHEMES = {
    'indigo':  {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose':    {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
}


SPECIES_COLORS = [
    '#6366f1', '#10b981', '#f43f5e', '#f59e0b', '#06b6d4',
    '#8b5cf6', '#ec4899', '#14b8a6', '#eab308', '#3b82f6',
]


# ── HTML ────────────────────────────────────────────────────────────

def generate_html(sim_results, output_path):
    sections_html = []
    all_js_data = {}

    for idx, (cfg, (species_ids, reaction_ids, snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n_species = len(species_ids)
        n_reactions = len(reaction_ids)
        n_snaps = len(snapshots)

        # Build time-series arrays
        times = [s['time'] for s in snapshots]
        species_series = {
            spid: [s['species'][spid] for s in snapshots]
            for spid in species_ids
        }
        rate_series = {
            rid: [s['rates'][rid] for s in snapshots]
            for rid in reaction_ids
        }

        # Phase portrait data
        px, py = cfg['phase_species']
        phase_x = species_series[px]
        phase_y = species_series[py]

        all_js_data[sid] = {
            'times': times,
            'species': species_series,
            'rates': rate_series,
            'species_ids': species_ids,
            'reaction_ids': reaction_ids,
            'phase': {'x_name': px, 'y_name': py, 'x': phase_x, 'y': phase_y},
            'color': cs['primary'],
            'species_colors': SPECIES_COLORS,
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg, species_ids)

        # Metric summaries
        final_species = snapshots[-1]['species']
        max_species_val = max(final_species.values())

        # Strip leading newline from antimony model for display
        model_src = cfg['model'].strip()

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Species</span><span class="metric-value">{n_species}</span></div>
        <div class="metric"><span class="metric-label">Reactions</span><span class="metric-value">{n_reactions}</span></div>
        <div class="metric"><span class="metric-label">Integrator</span><span class="metric-value">{cfg['integrator']}</span></div>
        <div class="metric"><span class="metric-label">Total Time</span><span class="metric-value">{cfg['total_time']:g}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{n_snaps:,}</span></div>
        <div class="metric"><span class="metric-label">Max Level</span><span class="metric-value">{max_species_val:.2f}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.2f}s</span></div>
      </div>

      <h3 class="subsection-title">Species Trajectories</h3>
      <div class="chart-box-full"><div id="chart-species-{sid}" class="chart-wide"></div></div>

      <div class="charts-row">
        <div class="chart-box"><div id="chart-phase-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-rates-{sid}" class="chart"></div></div>
      </div>

      <h3 class="subsection-title">Antimony Model Source</h3>
      <pre class="antimony-src">{_html_escape(model_src)}</pre>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tellurium × process-bigraph Simulation Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:720px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100;
        flex-wrap:wrap; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             color:#334155; transition:all .15s; background:#fff; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:820px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.25rem; font-weight:700; color:#1e293b; }}
.chart-box-full {{ background:#f8fafc; border:1px solid #e2e8f0;
                   border-radius:10px; overflow:hidden; margin-bottom:1rem; }}
.chart-wide {{ height:340px; }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:320px; }}
.antimony-src {{ background:#1e293b; color:#e2e8f0; padding:1.1rem 1.3rem;
                 border-radius:10px; font-family:'SF Mono',Menlo,Monaco,'Courier New',monospace;
                 font-size:.78rem; line-height:1.55; overflow-x:auto;
                 margin-bottom:1.2rem; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header,.nav {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>Tellurium × process-bigraph</h1>
  <p>Three biochemical models — a Lotka-Volterra oscillator, the
  Elowitz-Leibler repressilator, and stochastic dimerization —
  wrapped as <strong>process-bigraph</strong> Processes via
  <strong>tellurium</strong> / libroadrunner. Each configuration
  demonstrates how an SBML/Antimony model composes into a PBG
  Composite with lazy RoadRunner instantiation.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>viva-tellurium</strong> &mdash;
  Tellurium + process-bigraph &mdash;
  SBML &amp; Antimony model execution in bigraph composites
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') {{
    const short = obj.length > 120 ? obj.slice(0, 120) + '…' : obj;
    return '<span class="jt-str">"' + short.replace(/</g,'&lt;').replace(/\\n/g,'\\\\n') + '"</span>';
  }}
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (!el) return;
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:20, t:40, b:45 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];

  // Species trajectories (one trace per species)
  const speciesTraces = d.species_ids.map((spid, i) => ({{
    x: d.times, y: d.species[spid],
    type: 'scatter', mode: 'lines',
    line: {{ color: d.species_colors[i % d.species_colors.length], width: 1.8 }},
    name: spid,
  }}));
  Plotly.newPlot('chart-species-'+sid, speciesTraces, {{
    ...pLayout,
    title:{{ text:'Species Concentrations vs Time', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{ ...pLayout.xaxis, title:{{ text:'Time', font:{{ size:10 }} }} }},
    yaxis:{{ ...pLayout.yaxis, title:{{ text:'Concentration', font:{{ size:10 }} }} }},
    legend:{{ font:{{ size:10 }}, bgcolor:'rgba(255,255,255,0.6)' }}, showlegend:true,
  }}, pCfg);

  // Phase portrait
  Plotly.newPlot('chart-phase-'+sid, [{{
    x: d.phase.x, y: d.phase.y,
    type:'scatter', mode:'lines',
    line:{{ color: d.color, width: 1.4 }},
  }}], {{
    ...pLayout,
    title:{{ text:'Phase Portrait ('+d.phase.x_name+' vs '+d.phase.y_name+')',
             font:{{ size:12, color:'#334155' }} }},
    xaxis:{{ ...pLayout.xaxis, title:{{ text:d.phase.x_name, font:{{ size:10 }} }} }},
    yaxis:{{ ...pLayout.yaxis, title:{{ text:d.phase.y_name, font:{{ size:10 }} }} }},
    showlegend:false,
  }}, pCfg);

  // Reaction rates
  const rateTraces = d.reaction_ids.map((rid, i) => ({{
    x: d.times, y: d.rates[rid],
    type:'scatter', mode:'lines',
    line:{{ color: d.species_colors[(i+2) % d.species_colors.length], width: 1.5 }},
    name: rid,
  }}));
  Plotly.newPlot('chart-rates-'+sid, rateTraces, {{
    ...pLayout,
    title:{{ text:'Reaction Rates vs Time', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{ ...pLayout.xaxis, title:{{ text:'Time', font:{{ size:10 }} }} }},
    yaxis:{{ ...pLayout.yaxis, title:{{ text:'Rate', font:{{ size:10 }} }} }},
    legend:{{ font:{{ size:10 }}, bgcolor:'rgba(255,255,255,0.6)' }}, showlegend:true,
  }}, pCfg);
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def _html_escape(s):
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;'))


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        species_ids, reaction_ids, snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (species_ids, reaction_ids, snapshots, runtime)))
        print(f'  Runtime: {runtime:.3f}s, snapshots: {len(snapshots)}, '
              f'species: {len(species_ids)}, reactions: {len(reaction_ids)}')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    # Auto-open in Safari
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
