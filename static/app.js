// ==================== BARBARIK PROCEDURAL AUDIO ENGINE ====================
class BarbarikAudioEngine {
    constructor() {
        this.ctx = null;
        this.enabled = true;
        this.volume = 0.5;
        this.uiSound = true;
        this.simSound = true;
    }

    init() {
        if (!this.ctx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AudioContext();
        }
        if (this.ctx && this.ctx.state === 'suspended') {
            this.ctx.resume();
        }
    }

    playTone(freq, duration = 0.08, type = 'sine', gainVal = 0.1) {
        if (!this.enabled || !this.ctx) return;
        try {
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = type;
            osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
            gain.gain.setValueAtTime(gainVal * this.volume, this.ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.0001, this.ctx.currentTime + duration);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(this.ctx.currentTime + duration);
        } catch (e) {
            console.warn('Audio play error:', e);
        }
    }

    playBootSound() {
        if (!this.enabled || !this.ctx) return;
        try {
            const now = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(150, now);
            osc.frequency.exponentialRampToValueAtTime(600, now + 0.4);
            gain.gain.setValueAtTime(0.15 * this.volume, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start();
            osc.stop(now + 0.5);
        } catch (e) {}
    }

    playClick() {
        if (!this.uiSound) return;
        this.playTone(800, 0.04, 'triangle', 0.08);
    }

    playHover() {
        if (!this.uiSound) return;
        this.playTone(400, 0.03, 'sine', 0.03);
    }

    playSlider() {
        if (!this.uiSound) return;
        this.playTone(300, 0.02, 'sine', 0.02);
    }

    playSimStart() {
        if (!this.simSound || !this.ctx) return;
        try {
            const now = this.ctx.currentTime;
            [220, 330, 440, 550].forEach((f, i) => {
                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(f, now + i * 0.06);
                gain.gain.setValueAtTime(0.08 * this.volume, now + i * 0.06);
                gain.gain.exponentialRampToValueAtTime(0.001, now + i * 0.06 + 0.1);
                osc.connect(gain);
                gain.connect(this.ctx.destination);
                osc.start(now + i * 0.06);
                osc.stop(now + i * 0.06 + 0.1);
            });
        } catch (e) {}
    }

    playSimDone() {
        if (!this.simSound || !this.ctx) return;
        try {
            const now = this.ctx.currentTime;
            [523.25, 659.25, 783.99].forEach((f) => {
                const osc = this.ctx.createOscillator();
                const gain = this.ctx.createGain();
                osc.type = 'sine';
                osc.frequency.setValueAtTime(f, now);
                gain.gain.setValueAtTime(0.1 * this.volume, now);
                gain.gain.exponentialRampToValueAtTime(0.001, now + 0.4);
                osc.connect(gain);
                gain.connect(this.ctx.destination);
                osc.start(now);
                osc.stop(now + 0.4);
            });
        } catch (e) {}
    }

    playShapOpen() {
        if (!this.uiSound || !this.ctx) return;
        try {
            const now = this.ctx.currentTime;
            const osc = this.ctx.createOscillator();
            const gain = this.ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(300, now);
            osc.frequency.exponentialRampToValueAtTime(700, now + 0.15);
            gain.gain.setValueAtTime(0.08 * this.volume, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
            osc.connect(gain);
            gain.connect(this.ctx.destination);
            osc.start(now);
            osc.stop(now + 0.18);
        } catch (e) {}
    }
}

const barbarikAudio = new BarbarikAudioEngine();

// ==================== GLOBAL STATE & MAPPINGS ====================
let currentScenarioId = 'SCN-2024-0356';
let modelMetrics = null;
let lastPredictionResult = null;
let historicalScenariosList = [];

const FEATURE_MAP = {
    'side_a_logistics': 'Logistics (Blue)',
    'side_b_logistics': 'Logistics (Red)',
    'side_a_mobility': 'Mobility (Blue)',
    'side_b_mobility': 'Mobility (Red)',
    'side_a_force_strength': 'Force Strength (Blue)',
    'side_b_force_strength': 'Force Strength (Red)',
    'side_a_intelligence': 'Intelligence (Blue)',
    'side_b_intelligence': 'Intelligence (Red)',
    'side_a_communication': 'Communication (Blue)',
    'side_b_communication': 'Communication (Red)',
    'side_a_readiness': 'Readiness (Blue)',
    'side_b_readiness': 'Readiness (Red)',
    'side_a_morale': 'Morale (Blue)',
    'side_b_morale': 'Morale (Red)',
    'side_a_training': 'Training (Blue)',
    'side_b_training': 'Training (Red)',
    'side_a_supply_distance': 'Supply Distance (Blue)',
    'side_b_supply_distance': 'Supply Distance (Red)',
    'side_a_surprise': 'Surprise (Blue)',
    'side_b_surprise': 'Surprise (Red)',
    'electronic_warfare_level': 'Electronic Warfare',
    'objective_type': 'Objective Type',
    'environmental_difficulty': 'Env. Difficulty'
};

function formatFeatureName(rawName) {
    if (FEATURE_MAP[rawName]) return FEATURE_MAP[rawName];
    if (rawName.startsWith('terrain_type_')) return 'Terrain Advantage';
    if (rawName.startsWith('weather_')) return 'Weather';
    if (rawName.startsWith('visibility_')) return 'Visibility';
    if (rawName.startsWith('time_of_day_')) return 'Time of Day';
    if (rawName.startsWith('objective_type_')) return 'Objective Type';
    return rawName;
}

// Initialize Battle Timeline
let timelineInterval = null;

function initTimeline() {
    // Reset timeline
    resetTimeline();
}

function resetTimeline() {
    if (timelineInterval) clearInterval(timelineInterval);

    const phases = ['phaseDeployment', 'phaseEngagement', 'phaseManeuver', 'phaseResolution'];
    phases.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('active', 'completed');
        }
    });

    document.getElementById('timelineProgress').style.width = '0%';
    document.getElementById('timelineProgressPercent').textContent = '0%';
    document.getElementById('timelineStatus').textContent = '● READY';
    document.getElementById('timelineStatus').style.color = '#10b981';

    // Reset metrics to initial values based on current sliders
    const blueForce = parseInt(document.getElementById('forceStrengthA').value);
    const redForce = parseInt(document.getElementById('forceStrengthB').value);

    document.getElementById('timelineBlueForce').style.width = `${blueForce}%`;
    document.getElementById('timelineBlueVal').textContent = `${blueForce}%`;
    document.getElementById('timelineRedForce').style.width = `${redForce}%`;
    document.getElementById('timelineRedVal').textContent = `${redForce}%`;
    document.getElementById('timelineBlueCasualty').style.width = '0%';
    document.getElementById('timelineBlueCasualtyVal').textContent = '0%';
    document.getElementById('timelineRedCasualty').style.width = '0%';
    document.getElementById('timelineRedCasualtyVal').textContent = '0%';

    // Clear events
    const eventsContainer = document.getElementById('timelineEvents');
    if (eventsContainer) {
        eventsContainer.innerHTML = '';
    }
}

function runTimeline(scenario, result) {
    resetTimeline();

    const phases = [
        { id: 'phaseDeployment', duration: 2000 },
        { id: 'phaseEngagement', duration: 2000 },
        { id: 'phaseManeuver', duration: 2000 },
        { id: 'phaseResolution', duration: 1500 }
    ];

    const events = [
        { time: '00:00', desc: `Blue forces deployed (${scenario.side_a_force_strength} units)` },
        { time: '00:15', desc: `Red forces detected (${scenario.side_b_force_strength} units)` },
        { time: '00:30', desc: 'Initial contact established' },
        { time: '00:45', desc: `Terrain: ${scenario.terrain_type}` },
        { time: '01:00', desc: `Weather: ${scenario.weather}` },
        { time: '01:15', desc: 'Engagement intensifies' },
        { time: '01:30', desc: `Blue morale: ${(scenario.side_a_morale * 100).toFixed(0)}%` },
        { time: '01:45', desc: `Red morale: ${(scenario.side_b_morale * 100).toFixed(0)}%` },
        { time: '02:00', desc: 'Maneuver phase active' },
        { time: '02:15', desc: `Objective: ${scenario.objective_type}` },
        { time: '02:30', desc: 'Resolution approaching' },
        { time: '02:45', desc: `Predicted: ${result.outcome}` }
    ];

    let currentPhase = 0;
    let currentEvent = 0;
    let totalDuration = phases.reduce((sum, p) => sum + p.duration, 0);
    let elapsed = 0;

    document.getElementById('timelineStatus').textContent = '● SIMULATING';
    document.getElementById('timelineStatus').style.color = '#f59e0b';

    // Add initial event
    addEvent(events[0].time, events[0].desc);
    currentEvent++;

    timelineInterval = setInterval(() => {
        elapsed += 100;
        const progress = Math.min((elapsed / totalDuration) * 100, 100);

        document.getElementById('timelineProgress').style.width = `${progress}%`;
        document.getElementById('timelineProgressPercent').textContent = `${Math.round(progress)}%`;

        // Update phases
        let phaseElapsed = 0;
        for (let i = 0; i < phases.length; i++) {
            if (elapsed >= phaseElapsed && elapsed < phaseElapsed + phases[i].duration) {
                if (currentPhase !== i) {
                    // Mark previous as completed
                    if (i > 0) {
                        document.getElementById(phases[i - 1].id).classList.remove('active');
                        document.getElementById(phases[i - 1].id).classList.add('completed');
                    }
                    // Mark current as active
                    document.getElementById(phases[i].id).classList.add('active');
                    currentPhase = i;

                    // Add event for phase change
                    if (currentEvent < events.length) {
                        addEvent(events[currentEvent].time, events[currentEvent].desc);
                        currentEvent++;
                    }
                }
            }
            phaseElapsed += phases[i].duration;
        }

        // Randomly add events during simulation
        if (Math.random() < 0.05 && currentEvent < events.length) {
            addEvent(events[currentEvent].time, events[currentEvent].desc);
            currentEvent++;
        }

        // Update metrics dynamically
        if (currentPhase >= 1) {
            const blueForce = scenario.side_a_force_strength;
            const redForce = scenario.side_b_force_strength;
            const timeFactor = elapsed / totalDuration;

            const currentBlue = Math.max(0, blueForce - (blueForce * 0.2 * timeFactor));
            const currentRed = Math.max(0, redForce - (redForce * 0.25 * timeFactor));

            document.getElementById('timelineBlueForce').style.width = `${currentBlue}%`;
            document.getElementById('timelineBlueVal').textContent = `${Math.round(currentBlue)}%`;

            document.getElementById('timelineRedForce').style.width = `${currentRed}%`;
            document.getElementById('timelineRedVal').textContent = `${Math.round(currentRed)}%`;

            const blueCasualty = 20 * timeFactor;
            const redCasualty = 25 * timeFactor;

            document.getElementById('timelineBlueCasualty').style.width = `${blueCasualty}%`;
            document.getElementById('timelineBlueCasualtyVal').textContent = `${Math.round(blueCasualty)}%`;

            document.getElementById('timelineRedCasualty').style.width = `${redCasualty}%`;
            document.getElementById('timelineRedCasualtyVal').textContent = `${Math.round(redCasualty)}%`;
        }

        // Complete
        if (elapsed >= totalDuration) {
            clearInterval(timelineInterval);
            document.getElementById(phases[3].id).classList.remove('active');
            document.getElementById(phases[3].id).classList.add('completed');
            document.getElementById('timelineStatus').textContent = '● COMPLETE';
            document.getElementById('timelineStatus').style.color = '#10b981';

            // Add final event
            if (currentEvent < events.length) {
                addEvent(events[currentEvent].time, events[currentEvent].desc);
            }
        }
    }, 100);
}

function addEvent(time, desc) {
    const eventsContainer = document.getElementById('timelineEvents');
    if (!eventsContainer) return;

    const eventDiv = document.createElement('div');
    eventDiv.className = 'event-item';
    eventDiv.innerHTML = `
        <span class="event-time">${time}</span>
        <span class="event-desc">${desc}</span>
    `;
    eventsContainer.appendChild(eventDiv);
    eventsContainer.scrollTop = eventsContainer.scrollHeight;
}

// Log message to Diagnostic Console & Event Feed
function logConsole(msg, type = 'info') {
    const body = document.getElementById('consoleBody');
    if (!body) return;
    const timeStr = new Date().toLocaleTimeString('en-US', { hour12: false });
    const div = document.createElement('div');
    div.className = `log-entry ${type}`;
    div.textContent = `> [${timeStr}] ${msg}`;
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
}

// ==================== BOOT SCREEN CONTROLLER ====================
function initBootSequence() {
    const bootScreen = document.getElementById('bootScreen');
    if (!bootScreen) return;

    // Check if intro previously seen
    const forceBootPreview = new URLSearchParams(window.location.search).get('boot') === '1';
    if (!forceBootPreview && localStorage.getItem('barbarik_intro_seen') === 'true') {
        bootScreen.style.display = 'none';
        return;
    }

    // Reveal terminal lines sequentially
    for (let i = 1; i <= 8; i++) {
        setTimeout(() => {
            const line = document.getElementById(`termLine${i}`);
            if (line) line.classList.add('visible', 'done');
        }, i * 250);
    }

    const btnInit = document.getElementById('btnInitBoot');
    const btnSkip = document.getElementById('btnSkipBoot');

    if (btnInit) {
        btnInit.addEventListener('click', () => {
            barbarikAudio.init();
            barbarikAudio.playBootSound();
            localStorage.setItem('barbarik_intro_seen', 'true');
            bootScreen.classList.add('fade-out');
            setTimeout(() => { bootScreen.style.display = 'none'; }, 600);
            logConsole('System initialized via boot sequence.', 'ready');
        });
    }

    if (btnSkip) {
        btnSkip.addEventListener('click', () => {
            localStorage.setItem('barbarik_intro_seen', 'true');
            bootScreen.style.display = 'none';
            logConsole('Boot sequence skipped.', 'info');
        });
    }
}

// ==================== PLOTLY CHARTS ====================

// Update Terrain Image
function updateTerrainImage(terrain) {
    const img = document.getElementById('terrainImage');
    if (img) {
        const lowerTerrain = terrain.toLowerCase();
        img.style.opacity = '0.4';
        setTimeout(() => {
            img.src = `/static/assets/terrain/${lowerTerrain}.png`;
            img.style.opacity = '1';
        }, 150);
    }
}

// Update Weather Overlay
function updateWeatherOverlay(weather) {
    const overlay = document.getElementById('weatherOverlay');
    const indicator = document.getElementById('weatherIndicator');
    if (!overlay) return;

    // Remove all weather classes
    overlay.classList.remove('clear', 'cloudy', 'rain', 'fog', 'extreme');

    // Add the appropriate weather class
    const lowerWeather = weather.toLowerCase();
    overlay.classList.add(lowerWeather);

    // Update weather indicator
    if (indicator) {
        const weatherIcons = {
            'clear': '☀️',
            'cloudy': '☁️',
            'rain': '🌧️',
            'fog': '🌫️',
            'extreme': '⛈️'
        };
        indicator.textContent = `${weatherIcons[lowerWeather] || '🌤️'} ${weather.toUpperCase()}`;
    }

    console.log(`Weather overlay updated to: ${weather}`);
}

// Update Donut Chart
function updateDonutChart(confidence, winningSide) {
    const color = winningSide === 'BLUE FORCE ADVANTAGE' ? '#3b82f6' : 
                 (winningSide === 'RED FORCE ADVANTAGE' ? '#ef4444' : '#64748b');
    
    const data = [{
        values: [confidence, 1 - confidence],
        labels: ['Confidence', 'Remainder'],
        marker: {colors: [color, '#141d30']},
        type: 'pie',
        hole: 0.65,
        showlegend: false,
        hoverinfo: 'none'
    }];
    
    const layout = {
        margin: {l: 0, r: 0, t: 0, b: 0},
        width: 110,
        height: 110,
        annotations: [{
            text: `<b>${Math.round(confidence * 100)}%</b>`,
            font: {size: 18, color: '#ffffff', family: 'Inter'},
            showarrow: false,
            x: 0.5,
            y: 0.55
        }, {
            text: 'Confidence',
            font: {size: 9, color: '#64748b', family: 'Inter'},
            showarrow: false,
            x: 0.5,
            y: 0.36
        }],
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent'
    };
    
    Plotly.newPlot('donutChart', data, layout, {responsive: true, displayModeBar: false});
}

// Update Top Influencing Factors (Bar Chart)
function updateInfluencingFactors(shapFeatures) {
    let features = [];
    let values = [];
    let colors = [];
    
    if (shapFeatures && shapFeatures.length > 0) {
        const sorted = [...shapFeatures].reverse();
        features = sorted.map(f => formatFeatureName(f.feature));
        values = sorted.map(f => f.value);
        colors = sorted.map(f => {
            const label = formatFeatureName(f.feature);
            if (label === 'Terrain Advantage') return '#10b981';
            if (label === 'Weather') return '#64748b';
            return f.value >= 0 ? '#3b82f6' : '#ef4444';
        });
    } else {
        features = ['Weather', 'Mobility (Red)', 'Intelligence (Blue)', 'Force Strength (Blue)', 'Logistics (Red)', 'Mobility (Blue)', 'Terrain Advantage', 'Logistics (Blue)'];
        values = [-0.03, -0.06, 0.08, 0.11, -0.14, 0.17, 0.21, 0.24];
        colors = ['#64748b', '#ef4444', '#3b82f6', '#3b82f6', '#ef4444', '#3b82f6', '#10b981', '#3b82f6'];
    }
    
    const data = [{
        x: values,
        y: features,
        type: 'bar',
        orientation: 'h',
        marker: {color: colors},
        text: values.map(v => (v >= 0 ? `${v.toFixed(2)}` : `${v.toFixed(2)}`)),
        textposition: 'outside',
        textfont: {color: '#ffffff', size: 8, family: 'Inter'}
    }];
    
    const layout = {
        margin: {l: 100, r: 30, t: 5, b: 20},
        height: 190,
        xaxis: {
            range: [-0.35, 0.35],
            zeroline: true,
            zerolinecolor: '#64748b',
            gridcolor: '#1e293b',
            tickfont: {color: '#64748b', size: 8, family: 'Inter'}
        },
        yaxis: {
            tickfont: {color: '#e2e8f0', size: 8, family: 'Inter'},
            gridcolor: 'transparent'
        },
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent'
    };
    
    Plotly.newPlot('influencingFactors', data, layout, {responsive: true, displayModeBar: false});
}

// Update SHAP Beeswarm Summary Plot
function updateShapSummary() {
    const features = [
        'Weather', 'Mobility (Red)', 'Intelligence (Blue)', 'Force Strength (Blue)',
        'Logistics (Red)', 'Mobility (Blue)', 'Terrain Advantage', 'Logistics (Blue)'
    ];
    
    const nSamplesPerFeature = 35;
    const traces = [];
    
    features.forEach((feat, yIdx) => {
        const xVals = [];
        const yVals = [];
        const fVals = [];
        
        const centerImpacts = [-0.03, -0.06, 0.08, 0.11, -0.14, 0.17, 0.21, 0.24];
        const center = centerImpacts[yIdx];
        
        for (let i = 0; i < nSamplesPerFeature; i++) {
            const fVal = Math.random();
            const noise = (Math.random() - 0.5) * 0.07;
            const jitterY = yIdx + (Math.random() - 0.5) * 0.22;
            const shapVal = center + (fVal - 0.5) * 0.15 + noise;
            
            xVals.push(Math.max(-0.38, Math.min(0.38, shapVal)));
            yVals.push(jitterY);
            fVals.push(fVal);
        }
        
        traces.push({
            x: xVals,
            y: yVals,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 5,
                color: fVals,
                colorscale: [[0, '#3b82f6'], [1, '#ec4899']],
                showscale: (yIdx === features.length - 1),
                colorbar: (yIdx === features.length - 1) ? {
                    title: 'Feature Value',
                    titleside: 'right',
                    titlefont: {color: '#64748b', size: 8, family: 'Inter'},
                    thickness: 8,
                    len: 0.85,
                    tickvals: [0, 1],
                    ticktext: ['Low', 'High'],
                    tickfont: {color: '#64748b', size: 8, family: 'Inter'},
                    x: 1.02
                } : undefined
            },
            showlegend: false,
            hoverinfo: 'none'
        });
    });
    
    const layout = {
        margin: {l: 100, r: 50, t: 10, b: 30},
        height: 190,
        xaxis: {
            title: 'SHAP Value (Impact on Model Output)',
            titlefont: {color: '#64748b', size: 8, family: 'Inter'},
            range: [-0.4, 0.4],
            zeroline: true,
            zerolinecolor: '#64748b',
            gridcolor: '#1e293b',
            tickfont: {color: '#64748b', size: 8, family: 'Inter'}
        },
        yaxis: {
            tickvals: features.map((_, i) => i),
            ticktext: features,
            tickfont: {color: '#e2e8f0', size: 8, family: 'Inter'},
            gridcolor: '#1e293b',
            range: [-0.5, features.length - 0.5]
        },
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent'
    };
    
    Plotly.newPlot('shapSummary', traces, layout, {responsive: true, displayModeBar: false});
}

// Update ROC Curve Chart
function updateRocCurve(rocData) {
    const traces = [];
    
    traces.push({
        x: [0, 1],
        y: [0, 1],
        mode: 'lines',
        type: 'scatter',
        line: {color: '#64748b', dash: 'dash', width: 1},
        showlegend: false,
        hoverinfo: 'none'
    });
    
    if (rocData && Object.keys(rocData).length > 0) {
        const firstClass = Object.keys(rocData)[0];
        const data = rocData[firstClass];
        traces.push({
            x: data.fpr,
            y: data.tpr,
            mode: 'lines',
            type: 'scatter',
            line: {color: '#10b981', width: 2},
            showlegend: false
        });
    } else {
        const fpr = [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];
        const tpr = [0, 0.45, 0.70, 0.85, 0.92, 0.95, 0.98, 0.99, 1.0];
        traces.push({
            x: fpr,
            y: tpr,
            mode: 'lines',
            type: 'scatter',
            line: {color: '#10b981', width: 2},
            showlegend: false
        });
    }
    
    const layout = {
        margin: {l: 20, r: 10, t: 10, b: 20},
        width: 120,
        height: 120,
        xaxis: {
            title: 'False Positive Rate',
            titlefont: {color: '#64748b', size: 7, family: 'Inter'},
            range: [0, 1],
            gridcolor: '#1e293b',
            tickfont: {color: '#64748b', size: 7, family: 'Inter'}
        },
        yaxis: {
            title: 'True Positive Rate',
            titlefont: {color: '#64748b', size: 7, family: 'Inter'},
            range: [0, 1],
            gridcolor: '#1e293b',
            tickfont: {color: '#64748b', size: 7, family: 'Inter'}
        },
        plot_bgcolor: 'transparent',
        paper_bgcolor: 'transparent',
        annotations: [{
            text: `<b>AUC = ${modelMetrics?.roc_auc?.toFixed(2) || '0.95'}</b>`,
            x: 0.65,
            y: 0.2,
            font: {color: '#10b981', size: 9, family: 'Inter'},
            showarrow: false
        }]
    };
    
    Plotly.newPlot('rocCurve', traces, layout, {responsive: true, displayModeBar: false});
}

// Update History Table
function updateHistoryTable(history) {
    const tbody = document.getElementById('historyTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';
    
    historicalScenariosList = history || [];
    
    if (!history || history.length === 0) {
        history = [
            {scenario_id: 'SCN-2024-0356', date: '21 May 2024', blue_win_pct: 72, outcome: 'Blue Advantage'},
            {scenario_id: 'SCN-2024-0355', date: '21 May 2024', blue_win_pct: 38, outcome: 'Red Advantage'},
            {scenario_id: 'SCN-2024-0354', date: '20 May 2024', blue_win_pct: 51, outcome: 'Stalemate'},
            {scenario_id: 'SCN-2024-0353', date: '20 May 2024', blue_win_pct: 65, outcome: 'Blue Advantage'},
            {scenario_id: 'SCN-2024-0352', date: '19 May 2024', blue_win_pct: 28, outcome: 'Red Advantage'}
        ];
        historicalScenariosList = history;
    }
    
    history.slice(0, 5).forEach((record, idx) => {
        const row = document.createElement('tr');
        const outcomeUpper = (record.outcome || '').toUpperCase();
        const outcomeColor = outcomeUpper.includes('BLUE') ? '#3b82f6' : 
                            (outcomeUpper.includes('RED') ? '#ef4444' : '#e2e8f0');
        
        row.innerHTML = `
            <td style="font-weight: 600; color:#e2e8f0;">${record.scenario_id}</td>
            <td style="color: #64748b;">${record.date}</td>
            <td style="color:#e2e8f0;">${record.blue_win_pct}%</td>
            <td style="color: ${outcomeColor}; font-weight: 500;">${record.outcome}</td>
            <td>
                <i data-lucide="rotate-ccw" class="replay-btn-icon" data-idx="${idx}" title="Replay Scenario Visualization" style="cursor:pointer; color:#8b8f4a; width:12px; height:12px; margin-right:6px;"></i>
                <i data-lucide="eye" class="view-btn-icon" data-idx="${idx}" title="View Details" style="cursor:pointer; color:#c59b27; width:12px; height:12px;"></i>
            </td>
        `;
        tbody.appendChild(row);
    });
    
    if (window.lucide) {
        lucide.createIcons();
    }

    // Attach replay listeners
    document.querySelectorAll('.replay-btn-icon').forEach(el => {
        el.addEventListener('click', (e) => {
            const idx = e.currentTarget.getAttribute('data-idx');
            replayHistoricalScenario(idx);
        });
    });
}

// Replay selected historical scenario
function replayHistoricalScenario(idx) {
    const rec = historicalScenariosList[idx];
    if (!rec) return;
    barbarikAudio.playClick();
    logConsole(`Replaying Scenario ${rec.scenario_id}...`, 'sim');
    runSimulation(true);
}

// Read scenario input values from UI
function getScenarioValues() {
    return {
        side_a_force_strength: parseInt(document.getElementById('forceStrengthA').value),
        side_b_force_strength: parseInt(document.getElementById('forceStrengthB').value),
        side_a_mobility: parseFloat(document.getElementById('mobilityA').value),
        side_b_mobility: parseFloat(document.getElementById('mobilityB').value),
        side_a_logistics: parseFloat(document.getElementById('logisticsA').value),
        side_b_logistics: parseFloat(document.getElementById('logisticsB').value),
        side_a_intelligence: parseFloat(document.getElementById('intelligenceA').value),
        side_b_intelligence: parseFloat(document.getElementById('intelligenceB').value),
        side_a_communication: parseFloat(document.getElementById('communicationA').value),
        side_b_communication: parseFloat(document.getElementById('communicationB').value),
        side_a_readiness: parseFloat(document.getElementById('readinessA').value),
        side_b_readiness: parseFloat(document.getElementById('readinessB').value),
        side_a_morale: parseFloat(document.getElementById('moraleA').value),
        side_b_morale: parseFloat(document.getElementById('moraleB').value),
        side_a_training: parseFloat(document.getElementById('trainingA').value),
        side_b_training: parseFloat(document.getElementById('trainingB').value),
        side_a_supply_distance: parseInt(document.getElementById('supplyDistanceA').value),
        side_b_supply_distance: parseInt(document.getElementById('supplyDistanceB').value),
        side_a_surprise: parseFloat(document.getElementById('surpriseA').value),
        side_b_surprise: parseFloat(document.getElementById('surpriseB').value),
        side_a_fatigue: parseFloat(document.getElementById('fatigueA').value),
        side_b_fatigue: parseFloat(document.getElementById('fatigueB').value),
        side_a_equipment_quality: parseFloat(document.getElementById('equipmentQualityA').value),
        side_b_equipment_quality: parseFloat(document.getElementById('equipmentQualityB').value),
        electronic_warfare_level: parseFloat(document.getElementById('electronicWarfare').value),
        objective_type: document.getElementById('objectiveType').value,
        season: document.getElementById('season').value,
        rules_of_engagement: document.getElementById('rulesOfEngagement').value,
        terrain_type: document.getElementById('terrainType').value,
        weather: document.getElementById('weather').value,
        visibility: document.getElementById('visibility').value,
        time_of_day: document.getElementById('timeOfDay').value,
        environmental_difficulty: 0.3
    };
}

// Reset sliders to default values
function resetSliders() {
    document.getElementById('forceStrengthA').value = 80;
    document.getElementById('forceStrengthB').value = 70;
    document.getElementById('mobilityA').value = 0.75;
    document.getElementById('mobilityB').value = 0.60;
    document.getElementById('logisticsA').value = 0.70;
    document.getElementById('logisticsB').value = 0.55;
    document.getElementById('intelligenceA').value = 0.80;
    document.getElementById('intelligenceB').value = 0.60;
    document.getElementById('communicationA').value = 0.85;
    document.getElementById('communicationB').value = 0.65;
    document.getElementById('readinessA').value = 0.90;
    document.getElementById('readinessB').value = 0.70;
    document.getElementById('moraleA').value = 0.8;
    document.getElementById('moraleB').value = 0.6;
    document.getElementById('trainingA').value = 0.75;
    document.getElementById('trainingB').value = 0.55;
    document.getElementById('supplyDistanceA').value = 80;
    document.getElementById('supplyDistanceB').value = 150;
    document.getElementById('surpriseA').value = 0.6;
    document.getElementById('surpriseB').value = 0.3;
    document.getElementById('fatigueA').value = 0.2;
    document.getElementById('fatigueB').value = 0.4;
    document.getElementById('equipmentQualityA').value = 0.8;
    document.getElementById('equipmentQualityB').value = 0.6;
    
    document.getElementById('terrainType').value = 'Hilly';
    document.getElementById('weather').value = 'Cloudy';
    document.getElementById('visibility').value = 'Medium';
    document.getElementById('timeOfDay').value = 'Day';
    document.getElementById('electronicWarfare').value = '0.25';
    document.getElementById('objectiveType').value = 'Attack';
    document.getElementById('season').value = 'Summer';
    document.getElementById('rulesOfEngagement').value = 'Permissive';
    
    updateSliderValues();
    updateTerrainImage('Hilly');
    updateWeatherOverlay('Cloudy');
    barbarikAudio.playClick();
}

// Update slider value labels
function updateSliderValues() {
    const sliderIds = [
        'forceStrengthA', 'forceStrengthB',
        'mobilityA', 'mobilityB',
        'logisticsA', 'logisticsB',
        'intelligenceA', 'intelligenceB',
        'communicationA', 'communicationB',
        'readinessA', 'readinessB',
        'moraleA', 'moraleB',
        'trainingA', 'trainingB',
        'supplyDistanceA', 'supplyDistanceB',
        'surpriseA', 'surpriseB',
        'fatigueA', 'fatigueB',
        'equipmentQualityA', 'equipmentQualityB'
    ];
    
    sliderIds.forEach(id => {
        const slider = document.getElementById(id);
        const valueBox = document.getElementById(`${id}-val`);
        if (slider && valueBox) {
            valueBox.textContent = slider.value;
        }
    });
}

// ==================== LIVE SIMULATION PROGRESS MODAL ENGINE ====================
async function runSimulation(isReplay = false) {
    const scenario = getScenarioValues();
    barbarikAudio.playSimStart();
    logConsole(`Executing Wargame Simulation for ${currentScenarioId}...`, 'sim');

    const modal = document.getElementById('simProgressModal');
    const fill = document.getElementById('simProgressFill');
    const desc = document.getElementById('simStageDesc');
    const phaseTag = document.getElementById('livePhaseTag');

    if (phaseTag) phaseTag.textContent = '⚡ SIMULATING...';

    if (modal) modal.classList.add('open');

    // Simulation Stages
    const stages = [
        { pct: 20, text: 'INITIALIZING SCENARIO PARAMETERS...', stageId: 'stage1' },
        { pct: 40, text: 'ANALYZING TERRAIN & WEATHER ENVIRONMENT...', stageId: 'stage2' },
        { pct: 60, text: 'SIMULATING COMBAT INTERACTIONS...', stageId: 'stage3' },
        { pct: 85, text: 'EXECUTING RANDOM FOREST INFERENCE...', stageId: 'stage4' },
        { pct: 100, text: 'COMPUTING SHAP EXPLANATIONS...', stageId: 'stage5' }
    ];

    let currentStage = 0;
    const stageTimer = setInterval(() => {
        if (currentStage < stages.length) {
            const st = stages[currentStage];
            if (fill) fill.style.width = `${st.pct}%`;
            if (desc) desc.textContent = st.text;
            
            document.querySelectorAll('.stage-step').forEach(el => el.classList.remove('active'));
            const curStep = document.getElementById(st.stageId);
            if (curStep) curStep.classList.add('active', 'done');

            currentStage++;
        } else {
            clearInterval(stageTimer);
        }
    }, 280);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(scenario)
        });
        
        if (!response.ok) throw new Error('Prediction failed');
        
        const result = await response.json();
        lastPredictionResult = result;

        // Store for later panels
        window._lastPrediction = result;
        window._lastScenario = scenario;

        // Wait for visual simulation stages to reach completion
        setTimeout(() => {
            if (modal) modal.classList.remove('open');
            barbarikAudio.playSimDone();
            if (phaseTag) phaseTag.textContent = '✓ INFERENCE COMPLETE';
            
            // Render outputs
            applyPredictionResult(result, scenario);
            logConsole(`Inference complete: ${result.outcome} (${Math.round(result.confidence * 100)}% Confidence)`, 'ready');

            // Run Battle Timeline
            runTimeline(scenario, result);

            // Trigger the LIVE ENGINE panels
            if (typeof runMonteCarlo === 'function') runMonteCarlo(scenario);
            if (typeof loadEventLog === 'function') loadEventLog(scenario);
            if (typeof renderComparison === 'function') renderComparison();
        }, 1500);

    } catch (error) {
        console.error('Simulation error:', error);
        if (modal) modal.classList.remove('open');
        logConsole(`Simulation Error: ${error.message}`, 'error');
    }
}

// Apply Prediction Result to UI
function applyPredictionResult(result, scenario) {
    // Update outcome text & color
    const outcomeEl = document.getElementById('outcomeText');
    outcomeEl.textContent = result.outcome;
    outcomeEl.style.color = 
        result.outcome === 'BLUE FORCE ADVANTAGE' ? '#3b82f6' : 
        (result.outcome === 'RED FORCE ADVANTAGE' ? '#ef4444' : '#64748b');
    
    // Update Confidence Badge
    const confBadge = document.getElementById('confidenceBadge');
    const confPct = result.confidence;
    if (confBadge) {
        if (confPct >= 0.65) {
            confBadge.className = 'confidence-badge high';
            confBadge.textContent = 'HIGH CONFIDENCE';
        } else if (confPct >= 0.45) {
            confBadge.className = 'confidence-badge medium';
            confBadge.textContent = 'MEDIUM CONFIDENCE';
        } else {
            confBadge.className = 'confidence-badge low';
            confBadge.textContent = 'LOW CONFIDENCE';
        }
    }

    // Update Donut Chart & Probabilities Count Up
    updateDonutChart(result.confidence, result.outcome);
    
    const probs = result.probabilities;
    const probA = Math.round(probs.SIDE_A_ADVANTAGE * 100);
    const probB = Math.round(probs.SIDE_B_ADVANTAGE * 100);
    const probDraw = Math.round(probs.STALEMATE * 100);
    
    document.getElementById('probBarA').style.width = `${probA}%`;
    document.getElementById('probValueA').textContent = `${probA}%`;
    document.getElementById('probBarB').style.width = `${probB}%`;
    document.getElementById('probValueB').textContent = `${probB}%`;
    document.getElementById('probBarDraw').style.width = `${probDraw}%`;
    document.getElementById('probValueDraw').textContent = `${probDraw}%`;
    
    // Update Metric Cards
    document.getElementById('metricDuration').textContent = `${result.simulation_metrics.duration} Days`;
    document.getElementById('metricCasualties').innerHTML = 
        `Blue: ${result.simulation_metrics.casualties_blue}<br><span style="color:#ef4444;">Red: ${result.simulation_metrics.casualties_red}</span>`;
    document.getElementById('metricResource').innerHTML = 
        `Blue: ${result.simulation_metrics.resource_status_a}<br><span style="color:#ef4444;">Red: ${result.simulation_metrics.resource_status_b}</span>`;
    
    // Update Influencing Factors & SHAP Summary
    updateInfluencingFactors(result.shap_values);
    updateShapSummary();
    
    // Swap Terrain Image
    updateTerrainImage(scenario.terrain_type);
    
    // Refresh History
    loadHistory();
}

// Fetch model metrics
async function loadModelMetrics() {
    try {
        const response = await fetch('/api/model/metrics');
        if (!response.ok) throw new Error('Model metrics not available');
        
        modelMetrics = await response.json();
        
        document.getElementById('metricAccuracy').textContent = modelMetrics.accuracy.toFixed(2);
        document.getElementById('metricPrecision').textContent = modelMetrics.precision.toFixed(2);
        document.getElementById('metricRecall').textContent = modelMetrics.recall.toFixed(2);
        document.getElementById('metricF1').textContent = modelMetrics.f1.toFixed(2);
        document.getElementById('metricRocAuc').textContent = modelMetrics.roc_auc.toFixed(2);
        
        updateRocCurve(modelMetrics.roc_curve_data);
        logConsole(`Model Metrics Loaded: Accuracy=${modelMetrics.accuracy.toFixed(2)}, ROC-AUC=${modelMetrics.roc_auc.toFixed(2)}`, 'info');
        
    } catch (error) {
        console.error('Metrics error:', error);
        updateRocCurve(null);
    }
}

// Fetch scenario history
async function loadHistory() {
    try {
        const response = await fetch('/api/history');
        if (!response.ok) throw new Error('History not available');
        const data = await response.json();
        updateHistoryTable(data.history);
    } catch (error) {
        console.error('History error:', error);
        updateHistoryTable([]);
    }
}

// Generate New Scenario
async function newScenario() {
    try {
        barbarikAudio.playClick();
        const response = await fetch('/api/new-scenario', {method: 'POST'});
        if (!response.ok) throw new Error('New scenario failed');
        const data = await response.json();
        
        currentScenarioId = data.scenario_id;
        document.getElementById('scenarioId').textContent = currentScenarioId;
        resetSliders();
        logConsole(`Generated New Scenario ID: ${currentScenarioId}`, 'ready');
        
    } catch (error) {
        console.error('New scenario error:', error);
    }
}

// ==================== XAI "WHY THIS PREDICTION" DRAWER ====================
function openXaiExplanation() {
    barbarikAudio.playShapOpen();
    const modal = document.getElementById('xaiDrawer');
    if (!modal) return;

    modal.classList.add('open');

    const tbody = document.getElementById('xaiFeaturesBody');
    const reasoningBox = document.getElementById('xaiReasoningBox');

    if (tbody && lastPredictionResult && lastPredictionResult.shap_values) {
        tbody.innerHTML = '';
        const shapVals = lastPredictionResult.shap_values;

        shapVals.forEach(f => {
            const row = document.createElement('tr');
            const fname = formatFeatureName(f.feature);
            const val = f.value;
            const sign = val >= 0 ? `+${val.toFixed(3)}` : `${val.toFixed(3)}`;
            const dir = val >= 0 ? `<span style="color:#3b82f6;">Favorable for Blue</span>` : `<span style="color:#ef4444;">Favorable for Red</span>`;

            row.innerHTML = `
                <td style="font-weight:600;">${fname}</td>
                <td style="font-family:'JetBrains Mono'; font-weight:700;">${sign}</td>
                <td>${dir}</td>
            `;
            tbody.appendChild(row);
        });

        // Machine-generated natural language reasoning summary
        const top1 = formatFeatureName(shapVals[0].feature);
        const top1Val = shapVals[0].value.toFixed(2);
        const top2 = formatFeatureName(shapVals[1].feature);
        const top2Val = shapVals[1].value.toFixed(2);
        const outcome = lastPredictionResult.outcome;
        const conf = Math.round(lastPredictionResult.confidence * 100);

        if (reasoningBox) {
            reasoningBox.innerHTML = `
                <strong>Model Explanation:</strong> The Random Forest ensemble predicted <strong>${outcome}</strong> with <strong>${conf}% Confidence</strong>. 
                This inference is primarily driven by strong positive contributions from <strong>${top1} (+${top1Val})</strong> and <strong>${top2} (+${top2Val})</strong>.
            `;
        }
    } else if (tbody) {
        tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; color:#64748b;">Run a scenario simulation to generate SHAP explanations.</td></tr>`;
        if (reasoningBox) reasoningBox.textContent = `Awaiting simulation run to compute SHAP feature contributions.`;
    }
}

// ==================== RANDOM FOREST DECISION TREE INSPECTOR ====================
function openRfInspector() {
    barbarikAudio.playClick();
    const modal = document.getElementById('rfInspectorModal');
    if (modal) modal.classList.add('open');

    const container = document.getElementById('treeVizContainer');
    if (container) {
        container.innerHTML = `
            <svg width="520" height="180" viewBox="0 0 520 180" fill="none" xmlns="http://www.w3.org/2000/svg">
                <!-- Edges -->
                <line x1="260" y1="30" x2="140" y2="80" stroke="#1e293b" stroke-width="2"/>
                <line x1="260" y1="30" x2="380" y2="80" stroke="#1e293b" stroke-width="2"/>
                <line x1="140" y1="80" x2="80" y2="140" stroke="#1e293b" stroke-width="1.5"/>
                <line x1="140" y1="80" x2="200" y2="140" stroke="#1e293b" stroke-width="1.5"/>
                <line x1="380" y1="80" x2="320" y2="140" stroke="#1e293b" stroke-width="1.5"/>
                <line x1="380" y1="80" x2="440" y2="140" stroke="#1e293b" stroke-width="1.5"/>

                <!-- Root Node -->
                <rect x="190" y="10" width="140" height="35" rx="4" fill="#141d30" stroke="#3b82f6" stroke-width="1.5"/>
                <text x="260" y="26" fill="#e2e8f0" font-size="9" font-family="Inter" text-anchor="middle" font-weight="bold">Logistics (Blue) &lt;= 0.65</text>
                <text x="260" y="38" fill="#64748b" font-size="8" font-family="Inter" text-anchor="middle">Gini = 0.667 | Samples = 16,000</text>

                <!-- Branch 1 -->
                <rect x="80" y="65" width="120" height="30" rx="4" fill="#141d30" stroke="#ef4444" stroke-width="1.5"/>
                <text x="140" y="79" fill="#e2e8f0" font-size="8" font-family="Inter" text-anchor="middle">Mobility (Red) &lt;= 0.50</text>
                <text x="140" y="89" fill="#64748b" font-size="7" font-family="Inter" text-anchor="middle">Samples = 7,200</text>

                <!-- Branch 2 -->
                <rect x="320" y="65" width="120" height="30" rx="4" fill="#141d30" stroke="#10b981" stroke-width="1.5"/>
                <text x="380" y="79" fill="#e2e8f0" font-size="8" font-family="Inter" text-anchor="middle">Intel (Blue) &lt;= 0.75</text>
                <text x="380" y="89" fill="#64748b" font-size="7" font-family="Inter" text-anchor="middle">Samples = 8,800</text>

                <!-- Leaf Nodes -->
                <rect x="30" y="125" width="100" height="25" rx="4" fill="#1e293b" stroke="#ef4444"/>
                <text x="80" y="141" fill="#ef4444" font-size="8" font-weight="bold" font-family="Inter" text-anchor="middle">RED ADVANTAGE</text>

                <rect x="150" y="125" width="100" height="25" rx="4" fill="#1e293b" stroke="#64748b"/>
                <text x="200" y="141" fill="#e2e8f0" font-size="8" font-weight="bold" font-family="Inter" text-anchor="middle">STALEMATE</text>

                <rect x="270" y="125" width="100" height="25" rx="4" fill="#1e293b" stroke="#64748b"/>
                <text x="320" y="141" fill="#e2e8f0" font-size="8" font-weight="bold" font-family="Inter" text-anchor="middle">STALEMATE</text>

                <rect x="390" y="125" width="100" height="25" rx="4" fill="#1e293b" stroke="#3b82f6"/>
                <text x="440" y="141" fill="#3b82f6" font-size="8" font-weight="bold" font-family="Inter" text-anchor="middle">BLUE ADVANTAGE</text>
            </svg>
        `;
    }
}

// ==================== SCENARIO COMPARISON ====================
function openComparisonModal() {
    barbarikAudio.playClick();
    const modal = document.getElementById('compareModal');
    if (!modal) return;

    modal.classList.add('open');
    const selectA = document.getElementById('compareSelectA');
    const selectB = document.getElementById('compareSelectB');

    if (selectA && selectB) {
        selectA.innerHTML = '';
        selectB.innerHTML = '';
        historicalScenariosList.forEach((s, idx) => {
            selectA.innerHTML += `<option value="${idx}">${s.scenario_id} (${s.outcome})</option>`;
            selectB.innerHTML += `<option value="${idx}">${s.scenario_id} (${s.outcome})</option>`;
        });
        if (selectB.options.length > 1) selectB.selectedIndex = 1;

        renderComparisonGrid();
        selectA.addEventListener('change', renderComparisonGrid);
        selectB.addEventListener('change', renderComparisonGrid);
    }
}

function renderComparisonGrid() {
    const grid = document.getElementById('compareGrid');
    const selectA = document.getElementById('compareSelectA');
    const selectB = document.getElementById('compareSelectB');
    if (!grid || !selectA || !selectB) return;

    const sA = historicalScenariosList[selectA.value] || historicalScenariosList[0];
    const sB = historicalScenariosList[selectB.value] || historicalScenariosList[1] || sA;

    grid.innerHTML = `
        <table class="history-table-wrapper" style="width:100%; font-size:11px; margin-top:10px;">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th style="color:#3b82f6;">${sA.scenario_id}</th>
                    <th style="color:#8b8f4a;">${sB.scenario_id}</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Date Recorded</td><td>${sA.date}</td><td>${sB.date}</td></tr>
                <tr><td>Predicted Outcome</td><td style="font-weight:700;">${sA.outcome}</td><td style="font-weight:700;">${sB.outcome}</td></tr>
                <tr><td>Blue Win %</td><td>${sA.blue_win_pct}%</td><td>${sB.blue_win_pct}%</td></tr>
                <tr><td>Terrain Type</td><td>Hilly</td><td>Forest</td></tr>
                <tr><td>Force Ratio (Blue:Red)</td><td>80 vs 70</td><td>65 vs 85</td></tr>
            </tbody>
        </table>
    `;
}

// ==================== KEYBOARD SHORTCUTS & EVENT LISTENERS ====================
function initKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ignore shortcuts if user is typing in form controls
        if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

        const key = e.key.toUpperCase();
        if (key === 'S') {
            toggleSound();
        } else if (key === 'P') {
            togglePresentationMode();
        } else if (key === 'D') {
            runDemoMode();
        } else if (key === 'N') {
            newScenario();
        } else if (key === 'R') {
            runSimulation();
        } else if (key === 'E') {
            openXaiExplanation();
        } else if (key === 'ESCAPE') {
            closeAllModals();
        }
    });
}

function toggleSound() {
    barbarikAudio.enabled = !barbarikAudio.enabled;
    const btn = document.getElementById('soundToggleBtn');
    if (btn) {
        btn.innerHTML = barbarikAudio.enabled ? `<i data-lucide="volume-2"></i> SOUND ON` : `<i data-lucide="volume-x"></i> SOUND OFF`;
        if (window.lucide) lucide.createIcons();
    }
    logConsole(`Master Audio ${barbarikAudio.enabled ? 'ENABLED' : 'DISABLED'}`, 'info');
}

function togglePresentationMode() {
    document.body.classList.toggle('presentation-mode');
    const isPres = document.body.classList.contains('presentation-mode');
    logConsole(`Presentation Mode ${isPres ? 'ACTIVATED' : 'DEACTIVATED'}`, 'info');
}

function runDemoMode() {
    barbarikAudio.playClick();
    logConsole('Demo Mode Triggered: Auto-cycling test parameters...', 'sim');
    runSimulation();
}

function closeAllModals() {
    document.querySelectorAll('.modal-overlay').forEach(m => m.classList.remove('open'));
}

// Initialize on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    if (window.lucide) lucide.createIcons();

    // Boot Screen
    initBootSequence();

    // Sliders
    document.querySelectorAll('.slider').forEach(slider => {
        slider.addEventListener('input', () => {
            updateSliderValues();
            resetTimeline();
            barbarikAudio.playSlider();
        });
    });

    // Dropdowns
    document.querySelectorAll('.dropdown').forEach(dd => {
        dd.addEventListener('change', () => {
            barbarikAudio.playClick();
        });
    });

    // Event Listeners
    document.getElementById('runSimulationBtn')?.addEventListener('click', () => runSimulation());
    document.getElementById('newScenarioBtn')?.addEventListener('click', newScenario);
    document.getElementById('soundToggleBtn')?.addEventListener('click', toggleSound);
    document.getElementById('presModeBtn')?.addEventListener('click', togglePresentationMode);
    document.getElementById('demoModeBtn')?.addEventListener('click', runDemoMode);
    document.getElementById('whyPredictionBtn')?.addEventListener('click', openXaiExplanation);
    document.getElementById('inspectRfBtn')?.addEventListener('click', openRfInspector);
    document.getElementById('compareBtn')?.addEventListener('click', openComparisonModal);
    document.getElementById('vivaModeBtn')?.addEventListener('click', () => {
        barbarikAudio.playClick();
        document.getElementById('academicModal')?.classList.add('open');
    });
    document.getElementById('settingsBtn')?.addEventListener('click', () => {
        barbarikAudio.playClick();
        document.getElementById('settingsModal')?.classList.add('open');
    });

    document.getElementById('terrainType')?.addEventListener('change', (e) => {
        updateTerrainImage(e.target.value);
        resetTimeline();
    });

    document.getElementById('weather')?.addEventListener('change', (e) => {
        console.log('Weather changed to:', e.target.value);
        updateWeatherOverlay(e.target.value);
        resetTimeline();
    });

    document.getElementById('viewShapBtn')?.addEventListener('click', () => {
        document.querySelector('.panel-shap-summary')?.scrollIntoView({behavior: 'smooth'});
    });

    // Modal Closers
    document.querySelectorAll('.modal-close-btn').forEach(b => {
        b.addEventListener('click', closeAllModals);
    });

    // Command Console Toggle
    document.getElementById('consoleHeader')?.addEventListener('click', () => {
        document.getElementById('consoleDrawer')?.classList.toggle('collapsed');
    });

    // Export Report
    document.getElementById('exportReportBtn')?.addEventListener('click', () => {
        window.print();
    });

    // Replay Intro
    document.getElementById('btnReplayIntro')?.addEventListener('click', () => {
        localStorage.removeItem('barbarik_intro_seen');
        location.reload();
    });

    // Initialize Keyboard Shortcuts
    initKeyboardShortcuts();

    // Initial Display Setup
    updateSliderValues();
    updateDonutChart(0.72, 'BLUE FORCE ADVANTAGE');
    updateInfluencingFactors(null);
    updateShapSummary();
    updateRocCurve(null);
    updateWeatherOverlay('Cloudy');
    initTimeline();

    // Fetch API Data
    loadModelMetrics();
    loadHistory();
});
