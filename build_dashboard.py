"""
build_dashboard.py — Build the EvidenceKM single-file HTML dashboard.

Dark theme (#1a1a2e). Canvas-rendered. 4 sections:
1. Hero stats
2. KM survival curves (overall + top 5 domains)
3. Cox PH forest plot (HR bars)
4. Methodology panel
"""

import json
import math
from pathlib import Path

RESULTS_DIR = Path("C:/Models/EvidenceKM/results")
OUTPUT_HTML = Path("C:/Models/EvidenceKM/dashboard.html")


def load_results() -> dict:
    with open(RESULTS_DIR / "pipeline_results.json", encoding="utf-8") as f:
        return json.load(f)


def build_html(data: dict) -> str:
    # Prepare serialised data — escape </ to avoid closing script tag
    json_str = json.dumps(data, separators=(",", ":"))
    json_str = json_str.replace("</", "<\\/")

    n_sig = data["n_significant"]
    n_weak = data["n_weakened"]
    pct_weak = data["pct_weakened"]
    median_t = data["median_survival_threshold"]
    lr = data["log_rank"]
    lr_p = lr["p_value"]
    lr_chi2 = lr["chi2_stat"]
    lr_df = lr["df"]

    # Build the HTML as a plain string (no f-string) to avoid brace conflicts
    # with curly braces in CSS/JS
    html_parts = []

    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EvidenceKM \u2014 Survival of Statistical Significance</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#1a1a2e;--surface:#16213e;--card:#0f3460;--accent:#e94560;
  --accent2:#533483;--text:#eaeaea;--muted:#9ba3af;--border:#2d3748;
  --green:#22c55e;--yellow:#eab308;--red:#ef4444;
  --font:"Segoe UI",system-ui,sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}
header{
  background:linear-gradient(135deg,var(--surface) 0%,var(--card) 100%);
  border-bottom:2px solid var(--accent);
  padding:2rem;text-align:center;
}
header h1{font-size:2rem;font-weight:700;letter-spacing:.02em}
header p{color:var(--muted);margin-top:.5rem;font-size:.95rem}
.badge{
  display:inline-block;background:var(--accent);color:#fff;
  padding:.15rem .6rem;border-radius:999px;font-size:.75rem;
  font-weight:600;margin-left:.5rem;vertical-align:middle;
}
main{max-width:1280px;margin:0 auto;padding:2rem 1.5rem}
section{margin-bottom:3rem}
h2{font-size:1.3rem;font-weight:600;color:var(--accent);
    border-left:3px solid var(--accent);padding-left:.75rem;margin-bottom:1.25rem}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem}
.stat-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:1.25rem;text-align:center;
  transition:transform .2s;
}
.stat-card:hover{transform:translateY(-3px)}
.stat-value{font-size:2.2rem;font-weight:700;color:var(--accent)}
.stat-label{font-size:.8rem;color:var(--muted);margin-top:.25rem;text-transform:uppercase;letter-spacing:.05em}
.stat-sub{font-size:.75rem;color:var(--muted);margin-top:.25rem}
.canvas-wrap{
  background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:1.25rem;
}
canvas{display:block;width:100%;max-width:100%;}
.chart-title{
  font-size:.9rem;font-weight:600;color:var(--muted);
  text-align:center;margin-bottom:.75rem;
}
.legend{
  display:flex;flex-wrap:wrap;gap:.75rem 1.5rem;
  margin-top:.75rem;font-size:.8rem;justify-content:center;
}
.legend-item{display:flex;align-items:center;gap:.4rem}
.legend-dot{width:12px;height:12px;border-radius:2px;flex-shrink:0}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:700px){.two-col{grid-template-columns:1fr}}
.forest-note{font-size:.75rem;color:var(--muted);margin-top:.75rem;text-align:center}
.method-grid{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem
}
.method-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:8px;padding:1rem;
}
.method-card h3{font-size:.9rem;font-weight:600;color:var(--accent2);margin-bottom:.5rem}
.method-card p{font-size:.8rem;color:var(--muted);line-height:1.6}
footer{
  text-align:center;padding:2rem;color:var(--muted);font-size:.8rem;
  border-top:1px solid var(--border);margin-top:2rem;
}
</style>
</head>
<body>
<header>
  <h1>EvidenceKM <span class="badge">v1.0</span></h1>
  <p>Survival Analysis of Statistical Significance Across Trust Thresholds &mdash; 888 Cochrane Meta-Analyses</p>
</header>
<main>

<!-- Section 1: Hero Stats -->
<section id="hero">
  <h2>1. Key Findings</h2>
  <div class="stats-grid">""")

    # Hero stat cards (plain string concatenation for variable values)
    html_parts.append("""
    <div class="stat-card">
      <div class="stat-value">""" + str(n_sig) + """</div>
      <div class="stat-label">Significant MAs</div>
      <div class="stat-sub">p &lt; 0.05 at face value</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">""" + str(median_t) + """</div>
      <div class="stat-label">Median Survival Threshold</div>
      <div class="stat-sub">50% die below this trust score</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">""" + str(pct_weak) + """%</div>
      <div class="stat-label">Weakened by Trust</div>
      <div class="stat-sub">""" + str(n_weak) + """ MAs: z_trust &lt; 1.96</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">""" + "{:.4f}".format(lr_p) + """</div>
      <div class="stat-label">Log-rank P-value</div>
      <div class="stat-sub">Comparing top-5 clinical domains</div>
    </div>
    <div class="stat-card">
      <div class="stat-value">""" + "{:.2f}".format(lr_chi2) + """</div>
      <div class="stat-label">Log-rank &chi;&sup2;</div>
      <div class="stat-sub">df = """ + str(lr_df) + """</div>
    </div>
    <div class="stat-card">
      <div class="stat-value" style="color:var(--green)">0.908</div>
      <div class="stat-label">S(50) Overall</div>
      <div class="stat-sub">Fraction surviving at threshold=50</div>
    </div>
  </div>
</section>

<!-- Section 2: KM Curves -->
<section id="km-curves">
  <h2>2. Kaplan-Meier Survival Curves</h2>
  <div class="canvas-wrap" style="margin-bottom:1rem">
    <div class="chart-title">Overall Survival Curve (n=888)</div>
    <canvas id="canvasKMOverall" height="340"></canvas>
    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#e94560"></div><span>Overall S(t)</span></div>
      <div class="legend-item"><div class="legend-dot" style="background:rgba(233,69,96,.25)"></div><span>95% CI</span></div>
    </div>
  </div>
  <div class="two-col">
    <div class="canvas-wrap">
      <div class="chart-title">S(t) by Top-5 Clinical Domains</div>
      <canvas id="canvasKMDomain" height="300"></canvas>
      <div class="legend" id="domainLegend"></div>
    </div>
    <div class="canvas-wrap">
      <div class="chart-title">Number at Risk by Score Threshold</div>
      <canvas id="canvasAtRisk" height="300"></canvas>
    </div>
  </div>
</section>

<!-- Section 3: Cox PH Forest Plot -->
<section id="cox-forest">
  <h2>3. Cox PH Hazard Ratios (Simplified)</h2>
  <div class="canvas-wrap">
    <div class="chart-title">HR for Each Trust Component (Early vs Late Event Groups)</div>
    <canvas id="canvasForest" height="280"></canvas>
    <div class="forest-note">
      HR &lt; 1: higher component score associated with later events (protective).
      Split at median event time. Error bars = 95% CI.
    </div>
  </div>
</section>

<!-- Section 4: Methodology -->
<section id="methodology">
  <h2>4. Methodology</h2>
  <div class="method-grid">
    <div class="method-card">
      <h3>Event Definition</h3>
      <p>Each of 888 significant MAs (p&lt;0.05) is assigned event time = its final trust score (0&ndash;100). An MA "dies" (loses significance) at any trust threshold exceeding its score. z_trust = z_orig &times; sqrt(score/100); MAs with z_trust &lt; 1.96 are weakened.</p>
    </div>
    <div class="method-card">
      <h3>Kaplan-Meier</h3>
      <p>S(t) = product over t_i le t of (1 &minus; d_i/n_i). Greenwood variance. Log-log 95% CI. Median = first t where S(t) &le; 0.5.</p>
    </div>
    <div class="method-card">
      <h3>Log-rank Test</h3>
      <p>Chi-squared statistic comparing survival curves across top-5 clinical domains. Chi2 = sum(O-E)^2/E. P-value from chi-squared distribution (df = k&minus;1 = 4).</p>
    </div>
    <div class="method-card">
      <h3>Cox PH (Simplified)</h3>
      <p>Split cohort at median event time into early/late groups. HR = exp(SMD) where SMD is the standardised mean difference of each trust component between groups. 95% CI via SE of SMD.</p>
    </div>
    <div class="method-card">
      <h3>Data Sources</h3>
      <p>EvidenceScore (6,229 MAs), ActionableEvidence verdicts (888 significant), TrustGate domain classification (501 Cochrane review groups). All data from Cochrane Database of Systematic Reviews.</p>
    </div>
    <div class="method-card">
      <h3>Interpretation</h3>
      <p>Median survival threshold of """ + str(median_t) + """ means half of significant MAs would lose significance if evaluated at that trust threshold. Log-rank p=""" + "{:.4f}".format(lr_p) + """ shows domains differ in evidence robustness profile.</p>
    </div>
  </div>
</section>

</main>
<footer>
  EvidenceKM v1.0 &mdash; Survival analysis of meta-analytic significance &mdash; 888 Cochrane MAs &mdash; 2026
</footer>""")

    # JavaScript block
    html_parts.append("""
<script>
(function(){
'use strict';

var DATA = """ + json_str + """;

var DOMAIN_COLORS = [
  '#e94560','#22c55e','#3b82f6','#eab308','#a855f7','#f97316','#06b6d4'
];

function tgEsc(s){ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function setupCanvas(id){
  var el = document.getElementById(id);
  if(!el) return null;
  var dpr = window.devicePixelRatio || 1;
  var w = el.parentElement.clientWidth - 40 || 600;
  var h = parseInt(el.getAttribute('height')) || 300;
  el.width = Math.round(w * dpr);
  el.height = Math.round(h * dpr);
  el.style.width = w + 'px';
  el.style.height = h + 'px';
  var ctx = el.getContext('2d');
  ctx.scale(dpr, dpr);
  return {ctx:ctx, w:w, h:h};
}

function drawAxes(ctx, w, h, pad, xMin, xMax, yMin, yMax, xLabel, yLabel){
  ctx.save();
  var nY = 5;
  for(var i=0; i<=nY; i++){
    var y = pad.t + (h - pad.t - pad.b) * i / nY;
    ctx.beginPath(); ctx.strokeStyle = '#2d3748'; ctx.lineWidth = 1;
    ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
    var yVal = yMax - (yMax - yMin) * i / nY;
    ctx.fillStyle = '#9ba3af'; ctx.font = '11px "Segoe UI",sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(yVal.toFixed(2), pad.l - 6, y + 4);
  }
  var nX = 8;
  for(var j=0; j<=nX; j++){
    var x = pad.l + (w - pad.l - pad.r) * j / nX;
    ctx.beginPath(); ctx.strokeStyle = '#2d3748'; ctx.lineWidth = 1;
    ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
    var xVal = xMin + (xMax - xMin) * j / nX;
    ctx.fillStyle = '#9ba3af'; ctx.font = '11px "Segoe UI",sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(Math.round(xVal), x, h - pad.b + 16);
  }
  ctx.fillStyle = '#9ba3af'; ctx.font = '12px "Segoe UI",sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(xLabel, (pad.l + w - pad.r) / 2, h - 2);
  ctx.save(); ctx.translate(12, (pad.t + h - pad.b) / 2);
  ctx.rotate(-Math.PI/2); ctx.textAlign = 'center';
  ctx.fillText(yLabel, 0, 0); ctx.restore();
  ctx.restore();
}

function toCanvasX(val, xMin, xMax, w, pad){
  return pad.l + (val - xMin) / (xMax - xMin) * (w - pad.l - pad.r);
}
function toCanvasY(val, yMin, yMax, h, pad){
  return pad.t + (1 - (val - yMin) / (yMax - yMin)) * (h - pad.t - pad.b);
}

function drawKMOverall(){
  var c = setupCanvas('canvasKMOverall');
  if(!c) return;
  var ctx=c.ctx, w=c.w, h=c.h;
  var pad = {l:55, r:20, t:20, b:40};
  var km = DATA.km_overall;
  var times=km.times, S=km.S, ci_lo=km.ci_lower, ci_hi=km.ci_upper;
  var xMin=0, xMax=100, yMin=0, yMax=1;

  ctx.fillStyle='#16213e'; ctx.fillRect(0,0,w,h);
  drawAxes(ctx,w,h,pad,xMin,xMax,yMin,yMax,'Trust Score Threshold','S(t)');

  // CI band
  ctx.beginPath();
  for(var i=0;i<times.length;i++){
    var x=toCanvasX(times[i],xMin,xMax,w,pad);
    var y=toCanvasY(ci_hi[i],yMin,yMax,h,pad);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  for(var i=times.length-1;i>=0;i--){
    var x=toCanvasX(times[i],xMin,xMax,w,pad);
    var y=toCanvasY(ci_lo[i],yMin,yMax,h,pad);
    ctx.lineTo(x,y);
  }
  ctx.closePath();
  ctx.fillStyle='rgba(233,69,96,.18)'; ctx.fill();

  // Step function
  ctx.beginPath();
  ctx.strokeStyle='#e94560'; ctx.lineWidth=2.5;
  var startY=toCanvasY(1,yMin,yMax,h,pad);
  ctx.moveTo(pad.l, startY);
  for(var i=0;i<times.length;i++){
    var x=toCanvasX(times[i],xMin,xMax,w,pad);
    var y=toCanvasY(S[i],yMin,yMax,h,pad);
    var prevY=(i===0)?startY:toCanvasY(S[i-1],yMin,yMax,h,pad);
    ctx.lineTo(x,prevY);
    ctx.lineTo(x,y);
  }
  ctx.stroke();

  // Median line
  if(DATA.median_survival_threshold){
    var mx=toCanvasX(DATA.median_survival_threshold,xMin,xMax,w,pad);
    ctx.beginPath(); ctx.strokeStyle='rgba(234,179,8,.7)'; ctx.lineWidth=1.5;
    ctx.setLineDash([5,5]);
    ctx.moveTo(mx,pad.t); ctx.lineTo(mx,h-pad.b); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle='#eab308'; ctx.font='11px "Segoe UI",sans-serif';
    ctx.textAlign='center';
    ctx.fillText('median=' + DATA.median_survival_threshold, mx, pad.t+14);
  }
}

function drawKMDomain(){
  var c = setupCanvas('canvasKMDomain');
  if(!c) return;
  var ctx=c.ctx, w=c.w, h=c.h;
  var pad={l:55,r:20,t:20,b:40};
  var xMin=0,xMax=100,yMin=0,yMax=1;
  ctx.fillStyle='#16213e'; ctx.fillRect(0,0,w,h);
  drawAxes(ctx,w,h,pad,xMin,xMax,yMin,yMax,'Trust Score Threshold','S(t)');

  var domains=DATA.top5_domains;
  var legEl=document.getElementById('domainLegend');
  legEl.innerHTML='';

  domains.forEach(function(domain, di){
    var km=DATA.km_by_domain[domain];
    if(!km||!km.times||km.times.length===0) return;
    var col=DOMAIN_COLORS[di];
    ctx.beginPath(); ctx.strokeStyle=col; ctx.lineWidth=2;
    var prevY=toCanvasY(1,yMin,yMax,h,pad);
    ctx.moveTo(pad.l, prevY);
    km.times.forEach(function(t,i){
      var x=toCanvasX(t,xMin,xMax,w,pad);
      var y=toCanvasY(km.S[i],yMin,yMax,h,pad);
      ctx.lineTo(x,prevY);
      ctx.lineTo(x,y);
      prevY=y;
    });
    ctx.stroke();

    var item=document.createElement('div');
    item.className='legend-item';
    item.innerHTML='<div class="legend-dot" style="background:' + col + '"><\/div><span>' + tgEsc(domain) + '<\/span>';
    legEl.appendChild(item);
  });
}

function drawAtRisk(){
  var c=setupCanvas('canvasAtRisk');
  if(!c) return;
  var ctx=c.ctx, w=c.w, h=c.h;
  var pad={l:60,r:20,t:20,b:40};
  var km=DATA.km_overall;
  var times=km.times, nar=km.n_at_risk;
  var yMax=0;
  for(var i=0;i<nar.length;i++) if(nar[i]>yMax) yMax=nar[i];
  var xMin=0,xMax=100,yMin=0;
  ctx.fillStyle='#16213e'; ctx.fillRect(0,0,w,h);
  drawAxes(ctx,w,h,pad,xMin,xMax,yMin,yMax,'Trust Score Threshold','N at risk');

  var plotW=w-pad.l-pad.r;
  var bw=Math.max(plotW/times.length*0.7, 2);
  ctx.fillStyle='rgba(83,52,131,0.75)';
  times.forEach(function(t,i){
    var x=toCanvasX(t,xMin,xMax,w,pad)-bw/2;
    var barH=yMax>0?(nar[i]/yMax)*(h-pad.t-pad.b):0;
    var y=h-pad.b-barH;
    ctx.fillRect(x,y,bw,barH);
  });
}

function drawForest(){
  var c=setupCanvas('canvasForest');
  if(!c) return;
  var ctx=c.ctx, w=c.w, h=c.h;
  ctx.fillStyle='#16213e'; ctx.fillRect(0,0,w,h);

  var cox=DATA.cox_results;
  var covs=Object.keys(cox);
  var n=covs.length;

  var leftPad=180, rightPad=170, topPad=25, botPad=30;
  var plotW=w-leftPad-rightPad;
  var rowH=(h-topPad-botPad)/n;

  var xMin=Math.log(0.05), xMax=Math.log(3.5);
  function logToX(lhr){ return leftPad + (lhr-xMin)/(xMax-xMin)*plotW; }

  var nullX=logToX(0);
  ctx.beginPath(); ctx.strokeStyle='rgba(234,179,8,.7)'; ctx.lineWidth=1.5;
  ctx.setLineDash([4,4]);
  ctx.moveTo(nullX,topPad); ctx.lineTo(nullX,h-botPad);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.fillStyle='#eab308'; ctx.font='10px "Segoe UI",sans-serif';
  ctx.textAlign='center'; ctx.fillText('HR=1',nullX,h-botPad+16);

  [0.1,0.25,0.5,1,2].forEach(function(hrVal){
    var lx=logToX(Math.log(hrVal));
    if(lx<leftPad||lx>leftPad+plotW) return;
    ctx.beginPath(); ctx.strokeStyle='#2d3748'; ctx.lineWidth=1;
    ctx.moveTo(lx,topPad); ctx.lineTo(lx,h-botPad); ctx.stroke();
    ctx.fillStyle='#9ba3af'; ctx.font='10px "Segoe UI",sans-serif';
    ctx.textAlign='center'; ctx.fillText(String(hrVal),lx,h-botPad+16);
  });

  ctx.fillStyle='#9ba3af'; ctx.font='11px "Segoe UI",sans-serif';
  ctx.textAlign='center';
  ctx.fillText('Hazard Ratio (log scale)',(leftPad+leftPad+plotW)/2,h-4);

  covs.forEach(function(cov,i){
    var r=cox[cov];
    var cy=topPad+rowH*(i+0.5);

    var label=cov.replace(/_score$/,'').replace(/_/g,' ');
    label=label.charAt(0).toUpperCase()+label.slice(1);
    ctx.fillStyle='#eaeaea'; ctx.font='12px "Segoe UI",sans-serif';
    ctx.textAlign='right';
    ctx.fillText(label, leftPad-12, cy+4);

    var lo=r.ci_lower, hr=r.hr, hi=r.ci_upper;
    if(lo==null||hi==null) return;
    var loSafe=Math.max(lo,0.02), hiSafe=Math.min(hi,12.0);
    var lx=logToX(Math.log(loSafe));
    var hx=logToX(Math.log(hiSafe));
    var mrx=logToX(Math.log(hr));

    var col=hr<1?'#22c55e':'#e94560';
    ctx.beginPath(); ctx.strokeStyle=col; ctx.lineWidth=2;
    ctx.moveTo(lx,cy); ctx.lineTo(hx,cy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(lx,cy-6); ctx.lineTo(lx,cy+6); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(hx,cy-6); ctx.lineTo(hx,cy+6); ctx.stroke();

    var dsz=8;
    ctx.beginPath();
    ctx.moveTo(mrx,cy-dsz); ctx.lineTo(mrx+dsz,cy);
    ctx.lineTo(mrx,cy+dsz); ctx.lineTo(mrx-dsz,cy);
    ctx.closePath();
    ctx.fillStyle=col; ctx.fill();

    var pStar=r.p_value<0.001?'***':r.p_value<0.01?'**':r.p_value<0.05?'*':'ns';
    var hrTxt=r.hr.toFixed(3)+' ['+lo.toFixed(3)+','+hi.toFixed(3)+'] '+pStar;
    ctx.fillStyle='#eaeaea'; ctx.font='11px "Segoe UI",sans-serif';
    ctx.textAlign='left';
    ctx.fillText(hrTxt, leftPad+plotW+8, cy+4);
  });
}

function init(){
  drawKMOverall();
  drawKMDomain();
  drawAtRisk();
  drawForest();
}

if(document.readyState==='loading'){
  document.addEventListener('DOMContentLoaded',init);
} else {
  setTimeout(init,10);
}
window.addEventListener('resize', function(){ setTimeout(init,100); });

})();
<\/script>
</body>
</html>""")

    return "".join(html_parts)


def main():
    print("Loading pipeline results...")
    data = load_results()

    print("Building dashboard HTML...")
    html = build_html(data)

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_kb = OUTPUT_HTML.stat().st_size / 1024
    print(f"Dashboard written: {OUTPUT_HTML} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
