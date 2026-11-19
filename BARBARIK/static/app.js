/* BARBARIK dashboard interactivity */
(function () {
  "use strict";

  const TERRAIN_COLORS = {
    Open:   [[0, "rgb(180,210,140)"], [1, "rgb(120,160,90)"]],
    Hilly:  [[0, "rgb(120,100,60)"],  [0.5, "rgb(90,120,70)"], [1, "rgb(60,90,50)"]],
    Forest: [[0, "rgb(40,80,40)"],   [1, "rgb(20,50,20)"]],
    Desert: [[0, "rgb(220,200,130)"],[1, "rgb(180,150,90)"]],
    Urban:  [[0, "rgb(120,120,130)"],[1, "rgb(70,70,80)"]],
    Mixed:  [[0, "rgb(140,150,90)"], [0.5, "rgb(90,110,70)"], [1, "rgb(70,90,100)"]],
  };

  const OUTCOME_LABELS = {
    SIDE_A_ADVANTAGE: "BLUE FORCE ADVANTAGE",
    SIDE_B_ADVANTAGE: "RED FORCE ADVANTAGE",
    STALEMATE: "STALEMATE",
  };

  let currentScenarioId = "SCN-2024-0356";

  // ---------- Utility: build scenario dict from inputs ----------
  function buildScenario() {
    const v = (id) => parseFloat(document.getElementById(id).value);
    const s = (id) => document.getElementById(id).value;
    return {
      scenario_id: currentScenarioId,
      side_a_force_strength: v("a_force"),
      side_b_force_strength: v("b_force"),
      side_a_mobility: v("a_mobility"),
      side_b_mobility: v("b_mobility"),
      side_a_logistics: v("a_logistics"),
      side_b_logistics: v("b_logistics"),
      side_a_intelligence: v("a_intelligence"),
      side_b_intelligence: v("b_intelligence"),
      side_a_communication: v("a_communication"),
      side_b_communication: v("b_communication"),
      side_a_readiness: v("a_readiness"),
      side_b_readiness: v("b_readiness"),
      side_a_morale: 0.7,
      side_b_morale: 0.6,
      side_a_training: 0.75,
      side_b_training: 0.65,
      side_a_supply_distance: 200,
      side_b_supply_distance: 250,
      side_a_surprise: 0.3,
      side_b_surprise: 0.25,
      terrain_type: s("terrain_type"),
      weather: s("weather"),
      visibility: s("visibility"),
      time_of_day: s("time_of_day"),
      environmental_difficulty: v("environmental_difficulty"),
      electronic_warfare_level: 0.25,
      objective_type: "Attack",
    };
  }

  // ---------- Slider value display ----------
  document.querySelectorAll('input[type="range"]').forEach((el) => {
    el.addEventListener("input", () => {
      const val = el.value;
      const disp = parseFloat(val) < 10 && parseFloat(val) % 1 !== 0 ? parseFloat(val).toFixed(2) : val;
      document.getElementById(el.id + "_val").textContent = disp;
    });
    el.addEventListener("change", () => {
      if (el.id === "terrain_type") return;
    });
  });

  document.getElementById("terrain_type").addEventListener("change", () => {
    drawBattlefield(document.getElementById("terrain_type").value);
  });

  // ---------- Perlin-like noise ----------
  function makeNoiseGrid(size) {
    // Simple value-noise based on hashed grid points (deterministic per terrain).
    const grid = [];
    for (let y = 0; y < size; y++) {
      const row = [];
      for (let x = 0; x < size; x++) {
        const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
        const f = n - Math.floor(n);
        const n2 = Math.sin(x * 4.1 + y * 1.7) * 0.5 + 0.5;
        row.push(f * 0.6 + n2 * 0.4);
      }
      grid.push(row);
    }
    return grid;
  }

  function drawBattlefield(terrain) {
    const z = makeNoiseGrid(50);
    const colorscale = TERRAIN_COLORS[terrain] || TERRAIN_COLORS.Mixed;

    const blueUnits = { x: [8, 12, 10], y: [20, 28, 35] };
    const redUnits = { x: [38, 42, 40, 44], y: [22, 30, 38, 26] };

    const data = [
      {
        z: z,
        type: "heatmap",
        colorscale: colorscale,
        showscale: false,
        hoverinfo: "skip",
      },
      {
        x: blueUnits.x,
        y: blueUnits.y,
        mode: "markers",
        type: "scatter",
        marker: { color: "#3b82f6", size: 14, symbol: "circle" },
        name: "Friendly",
        showlegend: false,
      },
      {
        x: redUnits.x,
        y: redUnits.y,
        mode: "markers",
        type: "scatter",
        marker: { color: "#ef4444", size: 12, symbol: "square" },
        name: "Enemy",
        showlegend: false,
      },
    ];

    const layout = {
      margin: { t: 4, r: 4, b: 4, l: 4 },
      xaxis: { visible: false, showgrid: false, zeroline: false, range: [0, 49] },
      yaxis: { visible: false, showgrid: false, zeroline: false, range: [0, 49], scaleanchor: "x" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    Plotly.newPlot("battlefield-chart", data, layout, { displayModeBar: false, responsive: true });
  }

  // ---------- Donut chart ----------
  function drawDonut(probA, probB, probDraw, winner) {
    const data = [
      {
        values: [probA, probB, probDraw],
        labels: ["Blue", "Red", "Draw"],
        type: "pie",
        hole: 0.72,
        marker: {
          colors: [
            winner === "SIDE_A_ADVANTAGE" ? "#3b82f6" : "#1e3a5f",
            winner === "SIDE_B_ADVANTAGE" ? "#ef4444" : "#5f1e1e",
            "#4a5568",
          ],
        },
        textinfo: "none",
        hoverinfo: "label+percent",
        showlegend: false,
      },
    ];
    const layout = {
      margin: { t: 0, r: 0, b: 0, l: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      annotations: [
        {
          text: `<b>${(Math.max(probA, probB, probDraw) * 100).toFixed(0)}%</b><br><span style="font-size:10px;color:#6b7280">Confidence</span>`,
          showarrow: false,
          font: { size: 20, color: "#e5e7eb" },
        },
      ],
    };
    Plotly.newPlot("donut-chart", data, layout, { displayModeBar: false, responsive: true });
  }

  // ---------- Top factors bar chart ----------
  function drawFactors(factors) {
    const labels = factors.map((f) => f.feature);
    const values = factors.map((f) => f.value);
    const colors = values.map((v) => (v >= 0 ? "#3b82f6" : "#ef4444"));
    const data = [
      {
        x: values,
        y: labels,
        type: "bar",
        orientation: "h",
        marker: { color: colors },
        text: values.map((v) => v.toFixed(2)),
        textposition: "outside",
        textfont: { color: "#e5e7eb", size: 10 },
      },
    ];
    const layout = {
      margin: { t: 4, r: 40, b: 4, l: 110 },
      xaxis: { visible: false, showgrid: false, zeroline: true, zerolinecolor: "#2d3748" },
      yaxis: { autorange: "reversed", tickfont: { size: 10, color: "#e5e7eb" } },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      bargap: 0.3,
    };
    Plotly.newPlot("factors-chart", data, layout, { displayModeBar: false, responsive: true });
  }

  // ---------- SHAP beeswarm ----------
  function drawShap(summary) {
    if (!summary.available || summary.features.length === 0) {
      Plotly.newPlot("shap-chart", [], {
        margin: { t: 40, r: 10, b: 30, l: 120 },
        annotations: [{ text: "SHAP data unavailable", showarrow: false, font: { color: "#6b7280" } }],
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
      }, { displayModeBar: false });
      return;
    }
    const features = summary.features;
    const shapVals = summary.shap_values;
    const featVals = summary.feature_values;
    const traces = [];
    for (let fi = 0; fi < features.length; fi++) {
      const xs = shapVals.map((row) => row[fi]);
      const fv = featVals.map((row) => row[fi]);
      const ys = xs.map(() => fi + (Math.random() - 0.5) * 0.5);
      traces.push({
        x: xs,
        y: ys,
        mode: "markers",
        type: "scatter",
        name: features[fi],
        marker: {
          size: 5,
          color: fv,
          colorscale: "RdBu",
          cmin: Math.min(...fv.flat()),
          cmax: Math.max(...fv.flat()),
          showscale: fi === features.length - 1,
          colorbar: fi === features.length - 1 ? {
            title: { text: "Feature Value", font: { size: 9, color: "#e5e7eb" } },
            tickfont: { size: 8, color: "#e5e7eb" },
            thickness: 8, len: 0.6,
          } : undefined,
        },
        showlegend: false,
      });
    }
    const layout = {
      margin: { t: 10, r: 50, b: 30, l: 120 },
      xaxis: { title: "SHAP Value (Impact on Model Output)", range: [-0.4, 0.4], zeroline: true, zerolinecolor: "#2d3748", gridcolor: "#1a2332", tickfont: { size: 9, color: "#e5e7eb" }, titlefont: { size: 9, color: "#6b7280" } },
      yaxis: { tickvals: features.map((_, i) => i), ticktext: features, tickfont: { size: 9, color: "#e5e7eb" }, gridcolor: "#1a2332" },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
    };
    Plotly.newPlot("shap-chart", traces, layout, { displayModeBar: false, responsive: true });
  }

  // ---------- ROC curve ----------
  function drawRoc(roc) {
    if (!roc) return;
    const data = [
      {
        x: roc.fpr,
        y: roc.tpr,
        type: "scatter",
        mode: "lines",
        line: { color: "#10b981", width: 2 },
        name: "ROC",
      },
      {
        x: [0, 1],
        y: [0, 1],
        type: "scatter",
        mode: "lines",
        line: { color: "#4a5568", width: 1, dash: "dash" },
        showlegend: false,
      },
    ];
    const layout = {
      margin: { t: 20, r: 10, b: 30, l: 40 },
      xaxis: { range: [0, 1], title: "FPR", tickfont: { size: 9 }, titlefont: { size: 9, color: "#6b7280" }, gridcolor: "#1a2332" },
      yaxis: { range: [0, 1], title: "TPR", tickfont: { size: 9 }, titlefont: { size: 9, color: "#6b7280" }, gridcolor: "#1a2332" },
      annotations: [{ text: `AUC = ${roc.auc}`, x: 0.6, y: 0.15, showarrow: false, font: { size: 11, color: "#10b981" } }],
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      showlegend: false,
    };
    Plotly.newPlot("roc-chart", data, layout, { displayModeBar: false, responsive: true });
  }

  // ---------- API calls ----------
  async function fetchJSON(url, opts) {
    const res = await fetch(url, opts);
    if (!res.ok) throw new Error(`${url} -> ${res.status}`);
    return res.json();
  }

  async function loadMetrics() {
    try {
      const m = await fetchJSON("/api/model/metrics");
      document.getElementById("model-used").textContent = m.model_used;
      document.getElementById("m-accuracy").textContent = m.accuracy.toFixed(2);
      document.getElementById("m-precision").textContent = m.precision.toFixed(2);
      document.getElementById("m-recall").textContent = m.recall.toFixed(2);
      document.getElementById("m-f1").textContent = m.f1.toFixed(2);
      document.getElementById("m-roc").textContent = m.roc_auc ? m.roc_auc.toFixed(2) : "—";
      drawRoc(m.roc_curve);
    } catch (e) {
      document.querySelectorAll(".perf-metrics .perf-row span:last-child").forEach((el) => (el.textContent = "—"));
    }
  }

  async function loadHistory() {
    try {
      const data = await fetchJSON("/api/history?limit=5");
      const tbody = document.getElementById("history-body");
      tbody.innerHTML = "";
      data.history.forEach((r) => {
        const tr = document.createElement("tr");
        const outcomeClass =
          r.predicted_outcome === "SIDE_A_ADVANTAGE" ? "blue" :
          r.predicted_outcome === "SIDE_B_ADVANTAGE" ? "red" : "gray";
        tr.innerHTML = `
          <td>${r.id}</td>
          <td>${r.timestamp.split(" ")[0]}</td>
          <td>${(r.prob_a * 100).toFixed(0)}%</td>
          <td class="history-outcome ${outcomeClass}">${r.predicted_outcome.replace("_", " ").replace("SIDE ", "").replace(" ADVANTAGE", " Advantage")}</td>
          <td><button class="view-btn" title="View"><i class="fa-solid fa-eye"></i></button></td>`;
        tbody.appendChild(tr);
      });
    } catch (e) { /* empty */ }
  }

  async function loadShapSummary() {
    try {
      const s = await fetchJSON("/api/shap/summary?sample_size=50");
      drawShap(s);
    } catch (e) { /* empty */ }
  }

  // ---------- Run prediction ----------
  async function runPrediction() {
    const btn = document.getElementById("btn-run");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> RUNNING...';
    try {
      const result = await fetchJSON("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario: buildScenario() }),
      });
      updateOutcome(result);
      drawFactors(result.shap_values);
      await loadHistory();
    } catch (e) {
      document.getElementById("outcome-title").textContent = "ERROR";
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-play"></i> RUN SIMULATION';
    }
  }

  function updateOutcome(r) {
    const title = document.getElementById("outcome-title");
    const outcome = r.outcome;
    title.textContent = OUTCOME_LABELS[outcome] || outcome;
    title.className = "outcome-title " + (outcome === "SIDE_A_ADVANTAGE" ? "blue" : outcome === "SIDE_B_ADVANTAGE" ? "red" : "gray");

    const pa = r.probabilities.SIDE_A_ADVANTAGE || 0;
    const pb = r.probabilities.SIDE_B_ADVANTAGE || 0;
    const pd = r.probabilities.STALEMATE || 0;
    drawDonut(pa, pb, pd, outcome);

    document.getElementById("bar-a").style.width = (pa * 100).toFixed(0) + "%";
    document.getElementById("bar-b").style.width = (pb * 100).toFixed(0) + "%";
    document.getElementById("bar-d").style.width = (pd * 100).toFixed(0) + "%";
    document.getElementById("pct-a").textContent = (pa * 100).toFixed(0) + "%";
    document.getElementById("pct-b").textContent = (pb * 100).toFixed(0) + "%";
    document.getElementById("pct-d").textContent = (pd * 100).toFixed(0) + "%";

    const sm = r.simulation_metrics;
    document.getElementById("metric-duration").textContent = sm.duration + " Days";
    const blueCas = Math.round(300 + (1 - pa) * 400);
    const redCas = Math.round(300 + (1 - pb) * 400);
    document.getElementById("metric-casualties").textContent = `Blue: ${blueCas}-${blueCas + 100} / Red: ${redCas}-${redCas + 120}`;
    document.getElementById("metric-resource").textContent = `Blue: ${sm.resource_a} / Red: ${sm.resource_b}`;
  }

  // ---------- New scenario ----------
  async function newScenario() {
    try {
      const r = await fetchJSON("/api/new-scenario", { method: "POST" });
      currentScenarioId = r.scenario_id;
      document.getElementById("scenario-id").textContent = currentScenarioId;
      if (r.defaults) applyDefaults(r.defaults);
    } catch (e) {
      currentScenarioId = "SCN-" + new Date().toISOString().slice(0, 10) + "-" + Math.floor(1000 + Math.random() * 9000);
      document.getElementById("scenario-id").textContent = currentScenarioId;
    }
    document.getElementById("btn-reset").click();
  }

  function applyDefaults(d) {
    const map = {
      a_force: "side_a_force_strength", b_force: "side_b_force_strength",
      a_mobility: "side_a_mobility", b_mobility: "side_b_mobility",
      a_logistics: "side_a_logistics", b_logistics: "side_b_logistics",
      a_intelligence: "side_a_intelligence", b_intelligence: "side_b_intelligence",
      a_communication: "side_a_communication", b_communication: "side_b_communication",
      a_readiness: "side_a_readiness", b_readiness: "side_b_readiness",
      environmental_difficulty: "environmental_difficulty",
    };
    for (const [id, key] of Object.entries(map)) {
      if (d[key] !== undefined) {
        document.getElementById(id).value = d[key];
        document.getElementById(id + "_val").textContent = d[key];
      }
    }
  }

  // ---------- Reset ----------
  function resetScenario() {
    const defaults = {
      a_force: 80, b_force: 70,
      a_mobility: 0.75, b_mobility: 0.60,
      a_logistics: 0.70, b_logistics: 0.55,
      a_intelligence: 0.80, b_intelligence: 0.60,
      a_communication: 0.85, b_communication: 0.65,
      a_readiness: 0.78, b_readiness: 0.62,
      environmental_difficulty: 0.35,
    };
    for (const [id, val] of Object.entries(defaults)) {
      document.getElementById(id).value = val;
      document.getElementById(id + "_val").textContent = val;
    }
    document.getElementById("terrain_type").value = "Hilly";
    document.getElementById("weather").value = "Cloudy";
    document.getElementById("visibility").value = "Medium";
    document.getElementById("time_of_day").value = "Day";
    drawBattlefield("Hilly");
  }

  // ---------- Wire up ----------
  document.getElementById("btn-run").addEventListener("click", runPrediction);
  document.getElementById("btn-reset").addEventListener("click", resetScenario);
  document.getElementById("btn-new-scenario").addEventListener("click", newScenario);
  document.getElementById("btn-shap").addEventListener("click", loadShapSummary);

  // ---------- Init ----------
  function init() {
    drawBattlefield(document.getElementById("terrain_type").value);
    loadMetrics();
    loadHistory();
    loadShapSummary();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
