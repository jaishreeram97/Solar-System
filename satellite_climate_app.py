"""
╔══════════════════════════════════════════════════════════════════╗
║   SATELLITE GROUND CONTROL + CLIMATE PREDICTION SYSTEM          ║
║   Flask + SocketIO · Scikit-Learn · Real-time Telemetry         ║
╚══════════════════════════════════════════════════════════════════╝
"""

import math, time, random, threading, json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

app = Flask(__name__)
app.config['SECRET_KEY'] = 'solar-system-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ═══════════════════════════════════════════════════════════════════
# SATELLITE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════
SATELLITES = {
    "ISS":        {"full":"International Space Station","alt":408,"inc":51.6,"period":92,"color":"#ffcc44","agency":"NASA/ESA/JAXA","purpose":"Research Lab","status":"NOMINAL","sensors":["temp","pressure","radiation","CO2"]},
    "HUBBLE":     {"full":"Hubble Space Telescope",     "alt":547,"inc":28.5,"period":95,"color":"#88ccff","agency":"NASA/ESA",     "purpose":"Deep Space Obs","status":"NOMINAL","sensors":["uv","optical","infrared"]},
    "JWST":       {"full":"James Webb Space Telescope", "alt":1500000,"inc":0,"period":365,"color":"#aaddff","agency":"NASA/ESA/CSA","purpose":"IR Astronomy","status":"NOMINAL","sensors":["infrared","spectroscopy"]},
    "GOES-16":    {"full":"GOES-16 Weather Sat",        "alt":35786,"inc":0,"period":1440,"color":"#44ff88","agency":"NOAA",        "purpose":"Weather Monitor","status":"NOMINAL","sensors":["temp","humidity","wind","cloud"]},
    "LANDSAT-9":  {"full":"Landsat 9",                  "alt":705,  "inc":98.2,"period":99,"color":"#ff8844","agency":"NASA/USGS",  "purpose":"Earth Imaging","status":"NOMINAL","sensors":["multispectral","thermal","ndvi"]},
    "SENTINEL-6": {"full":"Sentinel-6 Michael Freilich","alt":1336,"inc":66,"period":112,"color":"#cc88ff","agency":"ESA",         "purpose":"Sea Level Monitor","status":"NOMINAL","sensors":["altimeter","radiation","temp"]},
    "GPM":        {"full":"Global Precip Measurement",  "alt":407,  "inc":65,"period":92, "color":"#88ffcc","agency":"NASA/JAXA",  "purpose":"Precipitation","status":"NOMINAL","sensors":["microwave","radar","precip"]},
    "TERRA":      {"full":"Terra (EOS AM-1)",            "alt":705,  "inc":98.2,"period":99,"color":"#ffaa55","agency":"NASA",      "purpose":"Climate Monitor","status":"DEGRADED","sensors":["temp","aerosol","cloud","land"]},
}

GROUND_STATIONS = [
    {"id":"GS-1","name":"Goldstone",    "lat":35.4,"lon":-116.9,"country":"USA",    "status":"ONLINE", "color":"#4af"},
    {"id":"GS-2","name":"Madrid",       "lat":40.4,"lon":-4.2,  "country":"Spain",  "status":"ONLINE", "color":"#4af"},
    {"id":"GS-3","name":"Canberra",     "lat":-35.2,"lon":148.9,"country":"Australia","status":"ONLINE","color":"#4af"},
    {"id":"GS-4","name":"Svalbard",     "lat":78.2,"lon":15.4,  "country":"Norway", "status":"ONLINE", "color":"#4fa"},
    {"id":"GS-5","name":"Bangalore",    "lat":12.9,"lon":77.6,  "country":"India",  "status":"ONLINE", "color":"#4fa"},
    {"id":"GS-6","name":"McMurdo",      "lat":-77.8,"lon":166.7,"country":"Antarctica","status":"STANDBY","color":"#fa4"},
    {"id":"GS-7","name":"White Sands",  "lat":32.5,"lon":-106.6,"country":"USA",    "status":"ONLINE", "color":"#4af"},
    {"id":"GS-8","name":"Kourou",       "lat":5.2, "lon":-52.8, "country":"France", "status":"ONLINE", "color":"#4fa"},
]

# ═══════════════════════════════════════════════════════════════════
# TELEMETRY SIMULATION
# ═══════════════════════════════════════════════════════════════════
_sim_time = 0.0
_command_log = []
_alerts = []

def orbital_position(sat_name, t):
    """Compute simulated lat/lon for a satellite."""
    s = SATELLITES[sat_name]
    inc = math.radians(s["inc"])
    period = s["period"] * 60  # seconds
    angle = (t / period) * 2 * math.pi
    offset = hash(sat_name) % 628 / 100.0
    lon = math.degrees(angle + offset) % 360 - 180
    lat = math.degrees(math.asin(math.sin(inc) * math.sin(angle + offset))) 
    return round(lat, 3), round(lon, 3)

def compute_signal(sat_name, gs, sat_lat, sat_lon, t):
    """Signal strength based on geometry + noise."""
    dlat = sat_lat - gs["lat"]; dlon = sat_lon - gs["lon"]
    dist = math.sqrt(dlat**2 + dlon**2)
    base = max(0, 100 - dist * 0.6)
    noise = 5 * math.sin(t * 0.1 + hash(sat_name+gs["id"]) % 10)
    return round(max(0, min(100, base + noise)), 1)

def get_telemetry():
    global _sim_time
    t = _sim_time
    result = {}
    for name, s in SATELLITES.items():
        lat, lon = orbital_position(name, t)
        # Find best ground station
        best_gs, best_sig = None, -1
        signals = {}
        for gs in GROUND_STATIONS:
            sig = compute_signal(name, gs, lat, lon, t)
            signals[gs["id"]] = sig
            if sig > best_sig:
                best_sig = sig; best_gs = gs["id"]
        # Sensor readings (physics-based)
        phase = t * 0.001 + hash(name) % 628 / 100.0
        result[name] = {
            "lat": lat, "lon": lon,
            "alt": s["alt"],
            "signal": best_sig,
            "connected_gs": best_gs,
            "signals": signals,
            "status": s["status"],
            "battery": round(75 + 20*math.sin(phase), 1),
            "temp_internal": round(-15 + 45*math.sin(phase*0.7), 1),
            "data_rate": round(max(0, best_sig * 1.5 + random.gauss(0,5)), 1),
            "uptime_hrs": round(t/3600 + 1200, 1),
            "sensors": {
                "solar_irradiance": round(1361 + 2*math.sin(phase), 2),
                "magnetic_field":   round(25 + 10*math.sin(phase*0.5), 2),
                "particle_flux":    round(abs(500*math.sin(phase*2)), 1),
                "albedo":           round(0.3 + 0.05*math.sin(phase*0.3), 3),
            }
        }
    return result

# ═══════════════════════════════════════════════════════════════════
# ML MODEL: CLIMATE PREDICTION
# ═══════════════════════════════════════════════════════════════════
def generate_climate_dataset(n=4000):
    """
    Synthetic dataset linking orbital/planetary mechanics to climate signals.
    Features: Earth's revolution angle, axial tilt (season), solar activity,
              Jupiter-Sun alignment, lunar phase, orbital eccentricity phase,
              magnetic pole wander, particle flux.
    Targets: temperature anomaly, precipitation index, storm probability,
             polar ice index, climate disturbance score.
    """
    np.random.seed(42)
    t = np.linspace(0, 8*365*24, n)  # 8 years of hourly data

    # --- Orbital mechanics features ---
    rev_angle     = (t / (365.25*24)) * 2*np.pi              # Earth revolution 0→2π/year
    axial_tilt    = 23.44 * np.sin(rev_angle - 0.15)         # Seasonal tilt effect
    orbital_ecc   = 0.0167 * np.cos(rev_angle + 0.1)         # Eccentricity phase
    solar_dist    = 1.0 + orbital_ecc                         # 1 AU ± eccentricity
    solar_irr     = 1361 / (solar_dist**2)                    # Inverse-square law

    # Milankovitch-like cycles
    precession    = np.sin(t / (26000*365*24) * 2*np.pi)     # 26,000yr precession
    obliquity     = np.sin(t / (41000*365*24) * 2*np.pi)     # 41,000yr obliquity

    # Solar activity (11-year cycle)
    solar_cycle   = np.sin(t / (11*365*24) * 2*np.pi)
    solar_activity= 100 + 50*solar_cycle + 10*np.random.randn(n)

    # Lunar phase (29.5-day cycle)
    lunar_phase   = np.sin(t / (29.5*24) * 2*np.pi)

    # Jupiter-Sun gravitational alignment (11.86yr period)
    jupiter_align = np.sin(t / (11.86*365*24) * 2*np.pi)

    # Geomagnetic index
    geomag        = 20 + 15*solar_cycle + 5*np.random.randn(n)

    # Planetary disturbance score (composite)
    planetary_dist = (np.abs(jupiter_align) * 0.4 +
                      np.abs(solar_cycle) * 0.4 +
                      np.abs(lunar_phase) * 0.2)

    # Rotation speed variation (length-of-day)
    lod_variation = 0.002 * np.sin(rev_angle * 2) + 0.0005 * np.random.randn(n)

    X = pd.DataFrame({
        "rev_angle_sin":    np.sin(rev_angle),
        "rev_angle_cos":    np.cos(rev_angle),
        "axial_tilt":       axial_tilt,
        "orbital_ecc":      orbital_ecc,
        "solar_irradiance": solar_irr,
        "solar_activity":   solar_activity,
        "solar_cycle_sin":  solar_cycle,
        "lunar_phase":      lunar_phase,
        "jupiter_align":    jupiter_align,
        "geomagnetic_idx":  geomag,
        "planetary_dist":   planetary_dist,
        "precession":       precession,
        "obliquity":        obliquity,
        "lod_variation":    lod_variation,
    })

    # --- Target variables (physics-based + noise) ---
    temp_anomaly = (
        1.8 * axial_tilt / 23.44        # seasonal
        + 0.3 * solar_cycle              # solar forcing
        + 0.15 * jupiter_align           # gravitational nudge
        - 0.5 * orbital_ecc             # distance effect
        + 0.1 * obliquity               # long-cycle
        + np.random.randn(n) * 0.4
    )
    precip_idx = (
        0.5 * np.sin(rev_angle)
        + 0.3 * lunar_phase
        + 0.2 * solar_cycle
        + 0.1 * planetary_dist
        + np.random.randn(n) * 0.3
    )
    storm_prob = np.clip(
        0.3 + 0.2*np.abs(axial_tilt/23.44) + 0.15*planetary_dist + 0.1*np.abs(solar_cycle)
        + np.random.randn(n)*0.08, 0, 1
    )
    polar_ice = (
        -0.4*axial_tilt/23.44
        - 0.2*solar_cycle
        + 0.1*orbital_ecc
        + np.random.randn(n)*0.2
    )
    climate_score = (
        0.35*np.abs(temp_anomaly)
        + 0.25*np.abs(precip_idx)
        + 0.2*storm_prob
        + 0.1*planetary_dist
        + 0.1*np.abs(geomag/50)
        + np.random.randn(n)*0.05
    )
    climate_score = np.clip(climate_score, 0, 10)

    y = pd.DataFrame({
        "temp_anomaly":  temp_anomaly,
        "precip_idx":    precip_idx,
        "storm_prob":    storm_prob,
        "polar_ice_idx": polar_ice,
        "climate_score": climate_score,
    })
    return X, y, t

print("🔬 Generating climate dataset...")
X_data, y_data, t_series = generate_climate_dataset(4000)

print("🤖 Training ML models...")
models = {}
metrics = {}
feature_names = X_data.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

for target in y_data.columns:
    # Ensemble: Random Forest + Gradient Boosting
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42, n_jobs=-1))
    ])
    rf.fit(X_train, y_train[target])
    preds = rf.predict(X_test)
    mse = mean_squared_error(y_test[target], preds)
    r2  = r2_score(y_test[target], preds)

    models[target] = rf
    metrics[target] = {
        "r2": round(r2, 4),
        "mse": round(mse, 4),
        "rmse": round(math.sqrt(mse), 4),
        "accuracy_pct": round(max(0, r2)*100, 1)
    }
    importances = rf.named_steps['model'].feature_importances_
    feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
    metrics[target]["top_features"] = [(f, round(float(v)*100,2)) for f,v in feat_imp[:6]]
    print(f"  ✓ {target}: R²={r2:.4f}, RMSE={math.sqrt(mse):.4f}")

print("✅ All models trained!\n")

# Generate prediction timeline for dashboard
def get_climate_predictions(days_ahead=365):
    """Predict climate for next N days using current orbital mechanics."""
    future_hours = np.linspace(_sim_time, _sim_time + days_ahead*24, days_ahead)
    rev_angles = (future_hours / (365.25*24)) * 2*np.pi
    rows = []
    for i, (h, ra) in enumerate(zip(future_hours, rev_angles)):
        axial = 23.44 * math.sin(ra - 0.15)
        ecc   = 0.0167 * math.cos(ra + 0.1)
        sdist = 1.0 + ecc
        irr   = 1361 / (sdist**2)
        scyc  = math.sin(h / (11*365*24) * 2*math.pi)
        lphase= math.sin(h / (29.5*24) * 2*math.pi)
        jalign= math.sin(h / (11.86*365*24) * 2*math.pi)
        gmag  = 20 + 15*scyc
        pdist = abs(jalign)*0.4 + abs(scyc)*0.4 + abs(lphase)*0.2
        prec  = math.sin(h / (26000*365*24) * 2*math.pi)
        obli  = math.sin(h / (41000*365*24) * 2*math.pi)
        lod   = 0.002 * math.sin(ra*2)
        rows.append({
            "rev_angle_sin": math.sin(ra), "rev_angle_cos": math.cos(ra),
            "axial_tilt": axial, "orbital_ecc": ecc,
            "solar_irradiance": irr, "solar_activity": 100+50*scyc,
            "solar_cycle_sin": scyc, "lunar_phase": lphase,
            "jupiter_align": jalign, "geomagnetic_idx": gmag,
            "planetary_dist": pdist, "precession": prec,
            "obliquity": obli, "lod_variation": lod,
        })
    df = pd.DataFrame(rows)
    preds = {}
    for target, model in models.items():
        preds[target] = model.predict(df).tolist()
    dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days_ahead)]
    return dates, preds

# ═══════════════════════════════════════════════════════════════════
# BACKGROUND SIMULATION THREAD
# ═══════════════════════════════════════════════════════════════════
def simulation_loop():
    global _sim_time, _alerts
    while True:
        _sim_time += 30  # advance 30 simulation seconds per tick
        telemetry = get_telemetry()

        # Generate occasional alerts
        if random.random() < 0.04:
            sat = random.choice(list(SATELLITES.keys()))
            alert_types = [
                f"⚡ Solar flare affecting {sat} signal",
                f"🌡 Thermal spike on {sat} panel",
                f"📡 {sat} handoff: GS-{random.randint(1,8)} → GS-{random.randint(1,8)}",
                f"🔋 {sat} battery at {random.randint(20,40)}% (charging)",
                f"☢ Radiation belt crossing detected by {sat}",
                f"🌀 Magnetic storm — {sat} attitude correction",
            ]
            _alerts.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "msg": random.choice(alert_types),
                "level": random.choice(["INFO","WARN","WARN","CRIT"])
            })
            _alerts = _alerts[-30:]

        socketio.emit('telemetry', {
            "satellites": telemetry,
            "sim_time": _sim_time,
            "alerts": _alerts[-8:],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        time.sleep(1.5)

thread = threading.Thread(target=simulation_loop, daemon=True)
thread.start()

# ═══════════════════════════════════════════════════════════════════
# API ROUTES
# ═══════════════════════════════════════════════════════════════════
@app.route('/api/config')
def api_config():
    return jsonify({
        "satellites": SATELLITES,
        "ground_stations": GROUND_STATIONS,
        "metrics": metrics,
        "feature_names": feature_names,
    })

@app.route('/api/predictions')
def api_predictions():
    days = int(request.args.get('days', 365))
    dates, preds = get_climate_predictions(min(days, 730))
    return jsonify({"dates": dates, "predictions": preds})

@app.route('/api/command', methods=['POST'])
def api_command():
    data = request.json
    sat, cmd = data.get("satellite"), data.get("command")
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"time": ts, "satellite": sat, "command": cmd, "status": "ACK"}
    _command_log.append(entry)
    _alerts.append({"time": ts, "msg": f"📤 CMD → {sat}: {cmd}", "level": "INFO"})
    socketio.emit('command_ack', entry)
    return jsonify({"status": "ok", "entry": entry})

@app.route('/api/commands')
def api_commands():
    return jsonify({"commands": _command_log[-20:]})

@app.route('/api/planetary_disturbance')
def api_planetary():
    """Real-time planetary mechanics state."""
    t = _sim_time
    ra = (t / (365.25*24)) * 2*math.pi
    return jsonify({
        "earth_revolution_deg": round(math.degrees(ra) % 360, 2),
        "earth_rotation_rpm":   round(1/1436, 6),
        "axial_tilt_deg":       23.44,
        "season_factor":        round(math.sin(ra - 0.15), 4),
        "solar_cycle_phase":    round(math.sin(t/(11*365*24)*2*math.pi), 4),
        "lunar_phase":          round(math.sin(t/(29.5*24)*2*math.pi), 4),
        "jupiter_alignment":    round(math.sin(t/(11.86*365*24)*2*math.pi), 4),
        "orbital_eccentricity": round(0.0167*math.cos(ra+0.1), 6),
        "solar_irradiance":     round(1361/(1+0.0167*math.cos(ra+0.1))**2, 2),
        "magnetic_field_index": round(20+15*math.sin(t/(11*365*24)*2*math.pi), 2),
        "planetary_dist_score": round(
            abs(math.sin(t/(11.86*365*24)*2*math.pi))*0.4 +
            abs(math.sin(t/(11*365*24)*2*math.pi))*0.4 +
            abs(math.sin(t/(29.5*24)*2*math.pi))*0.2, 4)
    })

# ═══════════════════════════════════════════════════════════════════
# HTML DASHBOARD (single-file)
# ═══════════════════════════════════════════════════════════════════
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Satellite Ground Control + Climate AI</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@300;400;600&display=swap');
:root{
  --bg:#030a14; --panel:#060f1e; --border:rgba(40,120,255,.18); --accent:#1a8cff;
  --green:#0fdd88; --orange:#ff8822; --red:#ff3355; --purple:#9966ff;
  --yellow:#ffcc33; --cyan:#22eeff; --text:#b8d4f0; --dim:#4a6a8a;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);font-family:'Exo 2',sans-serif;color:var(--text);overflow:hidden;height:100vh}
#app{display:grid;grid-template-rows:52px 1fr;height:100vh}

/* HEADER */
header{
  background:rgba(6,15,30,.95);border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 20px;gap:24px;
  backdrop-filter:blur(10px);z-index:100;
}
header h1{font-family:'Orbitron',monospace;font-size:14px;font-weight:900;
  letter-spacing:.3em;color:#fff;text-shadow:0 0 20px var(--accent);}
.hdr-stat{display:flex;flex-direction:column;align-items:center;
  border-left:1px solid var(--border);padding-left:16px;min-width:80px;}
.hdr-stat .val{font-family:'Share Tech Mono';font-size:13px;color:var(--green)}
.hdr-stat .lbl{font-size:9px;letter-spacing:.15em;color:var(--dim);margin-top:1px}
.status-dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);animation:pulse 2s infinite;margin-right:8px}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.ml-badge{font-family:'Orbitron';font-size:8px;letter-spacing:.2em;
  background:rgba(150,80,255,.2);border:1px solid rgba(150,80,255,.4);
  color:#bb88ff;padding:3px 8px;border-radius:3px;margin-left:auto}

/* MAIN GRID */
main{display:grid;grid-template-columns:280px 1fr 300px;overflow:hidden}

/* ─ LEFT PANEL ─ */
.left-panel{
  background:var(--panel);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
}
.panel-section{border-bottom:1px solid var(--border);padding:12px}
.panel-section h3{font-family:'Orbitron';font-size:9px;letter-spacing:.25em;
  color:var(--accent);margin-bottom:10px}

/* Satellite list */
.sat-item{
  display:flex;align-items:center;gap:8px;padding:7px 8px;
  border-radius:6px;cursor:pointer;transition:all .15s;margin-bottom:3px;
  border:1px solid transparent;
}
.sat-item:hover,.sat-item.active{background:rgba(26,140,255,.08);border-color:var(--border)}
.sat-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.sat-name{font-size:11px;font-weight:600;letter-spacing:.05em;flex:1}
.sat-sig{font-family:'Share Tech Mono';font-size:10px}
.sig-good{color:var(--green)} .sig-warn{color:var(--orange)} .sig-bad{color:var(--red)}

/* Ground stations */
.gs-item{display:flex;align-items:center;gap:7px;padding:4px 6px;margin-bottom:2px;font-size:10px}
.gs-dot{width:7px;height:7px;border-radius:50%;background:var(--green)}
.gs-name{flex:1;color:var(--text)}
.gs-status{font-family:'Share Tech Mono';font-size:9px;color:var(--dim)}

/* Commands */
.cmd-panel{padding:10px}
.cmd-panel select,.cmd-panel input{
  width:100%;background:rgba(10,20,40,.8);border:1px solid var(--border);
  color:var(--text);font-family:'Share Tech Mono';font-size:11px;
  padding:6px 8px;border-radius:4px;margin-bottom:6px;outline:none;
}
.cmd-btn{
  width:100%;background:rgba(26,140,255,.15);border:1px solid var(--accent);
  color:var(--accent);font-family:'Orbitron';font-size:9px;letter-spacing:.2em;
  padding:7px;border-radius:4px;cursor:pointer;transition:all .15s;
}
.cmd-btn:hover{background:rgba(26,140,255,.3);color:#fff}
.cmd-log{max-height:100px;overflow-y:auto;margin-top:8px}
.cmd-log::-webkit-scrollbar{width:3px}
.cmd-log::-webkit-scrollbar-thumb{background:var(--border)}
.cmd-entry{font-family:'Share Tech Mono';font-size:9px;color:var(--green);
  padding:2px 0;border-bottom:1px solid rgba(40,120,255,.08)}

/* Alerts */
.alert-log{max-height:130px;overflow-y:auto;padding:8px}
.alert-log::-webkit-scrollbar{width:3px}
.alert-log::-webkit-scrollbar-thumb{background:var(--border)}
.alert-item{font-size:10px;padding:3px 0;border-bottom:1px solid rgba(40,120,255,.06)}
.alert-time{font-family:'Share Tech Mono';font-size:9px;color:var(--dim);margin-right:6px}
.al-INFO{color:var(--text)} .al-WARN{color:var(--orange)} .al-CRIT{color:var(--red)}

/* ─ CENTER: 3D Globe + Charts ─ */
.center-panel{display:flex;flex-direction:column;overflow:hidden}
.globe-wrap{position:relative;flex:0 0 55%;background:#020810}
#globe-canvas{width:100%;height:100%;display:block}
.globe-overlay{position:absolute;top:8px;left:12px;
  font-family:'Orbitron';font-size:9px;letter-spacing:.2em;color:rgba(100,160,255,.6)}
.charts-wrap{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--border);overflow:hidden}
.chart-panel{background:var(--panel);padding:12px;overflow:hidden;position:relative}
.chart-panel h4{font-family:'Orbitron';font-size:8px;letter-spacing:.2em;
  color:var(--dim);margin-bottom:8px}
canvas.chart{width:100%!important;height:calc(100% - 28px)!important}

/* ─ RIGHT PANEL ─ */
.right-panel{
  background:var(--panel);border-left:1px solid var(--border);
  display:flex;flex-direction:column;overflow-y:auto;
}
.right-panel::-webkit-scrollbar{width:3px}
.right-panel::-webkit-scrollbar-thumb{background:var(--border)}

/* Detail panel */
.detail-card{padding:14px;border-bottom:1px solid var(--border)}
.detail-card h3{font-family:'Orbitron';font-size:10px;letter-spacing:.2em;color:var(--accent);margin-bottom:10px}
.detail-row{display:flex;justify-content:space-between;align-items:center;
  padding:4px 0;border-bottom:1px solid rgba(40,120,255,.06);font-size:10px}
.detail-row .dk{color:var(--dim);letter-spacing:.05em}
.detail-row .dv{font-family:'Share Tech Mono';font-size:10px;color:var(--text)}
.dv.good{color:var(--green)} .dv.warn{color:var(--orange)} .dv.bad{color:var(--red)}

/* Signal bars */
.sig-bar{height:4px;border-radius:2px;background:rgba(40,120,255,.12);margin-bottom:6px;overflow:hidden}
.sig-fill{height:100%;border-radius:2px;transition:width .5s}

/* ML metrics */
.metric-card{padding:10px 14px;border-bottom:1px solid var(--border)}
.metric-card h4{font-family:'Orbitron';font-size:8px;letter-spacing:.2em;color:var(--purple);margin-bottom:8px}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.metric-box{background:rgba(150,80,255,.06);border:1px solid rgba(150,80,255,.15);
  border-radius:5px;padding:6px 8px;text-align:center}
.metric-box .mv{font-family:'Orbitron';font-size:12px;color:var(--purple)}
.metric-box .ml{font-size:9px;color:var(--dim);letter-spacing:.1em;margin-top:2px}

/* Feature importance */
.feat-bar-row{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:9px}
.feat-name{color:var(--dim);width:110px;flex-shrink:0;text-align:right;letter-spacing:.05em}
.feat-bar{flex:1;height:5px;background:rgba(40,120,255,.1);border-radius:2px;overflow:hidden}
.feat-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--purple))}
.feat-val{color:var(--text);width:30px;text-align:right;font-family:'Share Tech Mono'}

/* Planetary status */
.planet-status{padding:10px 14px;border-bottom:1px solid var(--border)}
.planet-status h4{font-family:'Orbitron';font-size:8px;letter-spacing:.2em;color:var(--cyan);margin-bottom:8px}
.planet-row{display:flex;justify-content:space-between;padding:3px 0;
  border-bottom:1px solid rgba(40,120,255,.06);font-size:10px}
.planet-row .pk{color:var(--dim)} .planet-row .pv{font-family:'Share Tech Mono';color:var(--cyan)}

/* Climate prediction timeline */
.climate-pred{padding:12px 14px;border-bottom:1px solid var(--border)}
.climate-pred h4{font-family:'Orbitron';font-size:8px;letter-spacing:.2em;color:var(--green);margin-bottom:10px}
.pred-gauge{margin-bottom:8px}
.pred-label{display:flex;justify-content:space-between;font-size:9px;margin-bottom:3px}
.pred-label .pk{color:var(--dim)} .pred-label .pval{font-family:'Share Tech Mono'}
.gauge-bar{height:6px;background:rgba(40,120,255,.1);border-radius:3px;overflow:hidden}
.gauge-fill{height:100%;border-radius:3px;transition:width .8s}
</style>
</head>
<body>
<div id="app">
<header>
  <div class="status-dot"></div>
  <h1>⚡ GROUND CONTROL · CLIMATE AI OBSERVATORY</h1>
  <div class="hdr-stat"><span class="val" id="hdr-sats">8</span><span class="lbl">SATELLITES</span></div>
  <div class="hdr-stat"><span class="val" id="hdr-gs">7</span><span class="lbl">STATIONS</span></div>
  <div class="hdr-stat"><span class="val" id="hdr-uplink">0</span><span class="lbl">UPLINK Mbps</span></div>
  <div class="hdr-stat"><span class="val" id="hdr-time">--:--:--</span><span class="lbl">UTC</span></div>
  <div class="hdr-stat"><span class="val" id="hdr-cscore" style="color:var(--orange)">--</span><span class="lbl">CLIMATE IDX</span></div>
  <span class="ml-badge">🤖 AI CLIMATE MODEL ACTIVE</span>
</header>

<main>
<!-- ── LEFT ── -->
<div class="left-panel">
  <div class="panel-section">
    <h3>📡 SATELLITES</h3>
    <div id="sat-list"></div>
  </div>
  <div class="panel-section">
    <h3>🗼 GROUND STATIONS</h3>
    <div id="gs-list"></div>
  </div>
  <div class="panel-section cmd-panel">
    <h3>📤 COMMAND UPLINK</h3>
    <select id="cmd-sat"></select>
    <select id="cmd-type">
      <option>ATTITUDE_ADJUST</option><option>SENSOR_CALIBRATE</option>
      <option>DATA_DUMP</option><option>SAFE_MODE</option>
      <option>BOOST_SIGNAL</option><option>ORBIT_CORRECTION</option>
      <option>CAPTURE_IMAGE</option><option>EMERGENCY_BEACON</option>
    </select>
    <button class="cmd-btn" onclick="sendCommand()">▶ TRANSMIT COMMAND</button>
    <div class="cmd-log" id="cmd-log"></div>
  </div>
  <div class="panel-section" style="flex:1;overflow:hidden;padding:0">
    <div style="padding:10px 12px 6px"><h3>⚠ SYSTEM ALERTS</h3></div>
    <div class="alert-log" id="alert-log"></div>
  </div>
</div>

<!-- ── CENTER ── -->
<div class="center-panel">
  <div class="globe-wrap">
    <canvas id="globe-canvas"></canvas>
    <div class="globe-overlay">🌍 REAL-TIME ORBITAL TRACKING</div>
  </div>
  <div class="charts-wrap">
    <div class="chart-panel">
      <h4>🌡 TEMPERATURE ANOMALY FORECAST (365 days)</h4>
      <canvas id="chart-temp" class="chart"></canvas>
    </div>
    <div class="chart-panel">
      <h4>🌀 STORM PROBABILITY · CLIMATE DISTURBANCE</h4>
      <canvas id="chart-storm" class="chart"></canvas>
    </div>
    <div class="chart-panel">
      <h4>🛰 SIGNAL STRENGTH BY GROUND STATION</h4>
      <canvas id="chart-signal" class="chart"></canvas>
    </div>
    <div class="chart-panel">
      <h4>📡 SATELLITE DATA THROUGHPUT (Mbps)</h4>
      <canvas id="chart-data" class="chart"></canvas>
    </div>
  </div>
</div>

<!-- ── RIGHT ── -->
<div class="right-panel">
  <div class="detail-card">
    <h3 id="detail-sat-name">SELECT A SATELLITE</h3>
    <div id="detail-body"></div>
    <div style="margin-top:8px">
      <h4 style="font-family:Orbitron;font-size:8px;letter-spacing:.15em;color:var(--dim);margin-bottom:6px">GROUND STATION LINKS</h4>
      <div id="gs-signal-bars"></div>
    </div>
  </div>

  <div class="metric-card">
    <h4>🤖 AI MODEL PERFORMANCE</h4>
    <div class="metric-grid" id="model-metrics"></div>
  </div>

  <div class="planet-status">
    <h4>🪐 PLANETARY MECHANICS (LIVE)</h4>
    <div id="planet-rows"></div>
  </div>

  <div class="metric-card" style="border-color:rgba(40,120,255,.1)">
    <h4 style="color:var(--accent)">📊 TOP CLIMATE PREDICTORS (FEATURE IMPORTANCE)</h4>
    <div id="feat-imp"></div>
  </div>

  <div class="climate-pred">
    <h4>🌍 30-DAY CLIMATE PREDICTION</h4>
    <div id="climate-gauges"></div>
  </div>
</div>
</main>
</div>

<script>
// ── Socket.IO ──────────────────────────────────────────────────────
const socket = io();
let config = {}, telemetry = {}, selectedSat = 'ISS';
let predDates = [], predData = {};
let charts = {};

// Chart colors
const COLORS = {
  temp:'#ff6644', storm:'#ff3355', precip:'#4488ff',
  polar:'#44ddff', climate:'#ff8822', signal:'#0fdd88', data:'#1a8cff'
};

// ── Bootstrap ──────────────────────────────────────────────────────
async function init() {
  const res = await fetch('/api/config');
  config = await res.json();
  buildSatList();
  buildGSList();
  buildCmdSatSelect();
  buildModelMetrics();
  buildFeatureImportance();
  await loadPredictions();
  buildCharts();
  setInterval(updatePlanetaryStatus, 3000);
  updatePlanetaryStatus();
}

async function loadPredictions() {
  const r = await fetch('/api/predictions?days=365');
  const d = await r.json();
  predDates = d.dates; predData = d.predictions;
}

// ── Satellite List ─────────────────────────────────────────────────
function buildSatList() {
  const el = document.getElementById('sat-list');
  el.innerHTML = Object.entries(config.satellites).map(([id, s]) => `
    <div class="sat-item ${id===selectedSat?'active':''}" id="si-${id}" onclick="selectSat('${id}')">
      <div class="sat-dot" style="background:${s.color};box-shadow:0 0 6px ${s.color}80"></div>
      <div class="sat-name">${id}</div>
      <span class="sat-sig" id="sig-${id}">--</span>
    </div>`).join('');
}

function selectSat(id) {
  selectedSat = id;
  document.querySelectorAll('.sat-item').forEach(e => e.classList.remove('active'));
  document.getElementById('si-'+id)?.classList.add('active');
  updateDetailPanel();
}

// ── GS List ───────────────────────────────────────────────────────
function buildGSList() {
  document.getElementById('gs-list').innerHTML = config.ground_stations.map(gs => `
    <div class="gs-item">
      <div class="gs-dot" style="background:${gs.status==='ONLINE'?'var(--green)':'var(--orange)'}"></div>
      <span class="gs-name">${gs.name}</span>
      <span class="gs-status">${gs.status}</span>
    </div>`).join('');
}

function buildCmdSatSelect() {
  document.getElementById('cmd-sat').innerHTML =
    Object.keys(config.satellites).map(id=>`<option>${id}</option>`).join('');
}

// ── Model Metrics ─────────────────────────────────────────────────
function buildModelMetrics() {
  const m = config.metrics;
  document.getElementById('model-metrics').innerHTML =
    Object.entries(m).map(([k,v])=>`
      <div class="metric-box">
        <div class="mv">${v.accuracy_pct}%</div>
        <div class="ml">${k.replace('_',' ').toUpperCase()}</div>
        <div style="font-size:8px;color:var(--dim);margin-top:2px">R²=${v.r2}</div>
      </div>`).join('');
}

function buildFeatureImportance() {
  // Use temp_anomaly features
  const feats = config.metrics.temp_anomaly?.top_features || [];
  const max = feats[0]?.[1] || 1;
  document.getElementById('feat-imp').innerHTML = feats.map(([name,val])=>`
    <div class="feat-bar-row">
      <span class="feat-name">${name.replace(/_/g,' ')}</span>
      <div class="feat-bar"><div class="feat-fill" style="width:${val/max*100}%"></div></div>
      <span class="feat-val">${val}%</span>
    </div>`).join('');
}

// ── Charts ────────────────────────────────────────────────────────
function buildCharts() {
  const chartCfg = { responsive:true, maintainAspectRatio:false,
    plugins:{ legend:{display:false} },
    scales:{
      x:{ ticks:{color:'#2a4a6a',maxTicksLimit:8,font:{size:8}}, grid:{color:'rgba(40,120,255,.06)'} },
      y:{ ticks:{color:'#2a4a6a',font:{size:8}}, grid:{color:'rgba(40,120,255,.06)'} }
    }
  };

  // Temperature forecast (every 30 days)
  const step = 30;
  const datesShort = predDates.filter((_,i)=>i%step===0);
  const tempShort  = (predData.temp_anomaly||[]).filter((_,i)=>i%step===0);
  const stormShort = (predData.storm_prob||[]).filter((_,i)=>i%step===0);
  const clsShort   = (predData.climate_score||[]).filter((_,i)=>i%step===0);

  charts.temp = new Chart(document.getElementById('chart-temp'), {
    type:'line', data:{ labels:datesShort, datasets:[
      { label:'Temp Anomaly', data:tempShort,
        borderColor:COLORS.temp, backgroundColor:'rgba(255,100,68,.08)',
        borderWidth:1.5, pointRadius:2, tension:.4, fill:true }
    ]}, options:{...chartCfg, plugins:{...chartCfg.plugins}}
  });

  charts.storm = new Chart(document.getElementById('chart-storm'), {
    type:'line', data:{ labels:datesShort, datasets:[
      { label:'Storm Prob', data:stormShort,
        borderColor:COLORS.storm, backgroundColor:'rgba(255,51,85,.08)',
        borderWidth:1.5, pointRadius:2, tension:.4, fill:true },
      { label:'Climate Score', data:clsShort,
        borderColor:COLORS.climate, backgroundColor:'rgba(255,136,34,.05)',
        borderWidth:1.5, pointRadius:0, tension:.4, fill:false }
    ]}, options:{...chartCfg, plugins:{legend:{display:true, labels:{color:'#4a6a8a',font:{size:8}}}}}
  });

  // Signal & data charts — updated by telemetry
  charts.signal = new Chart(document.getElementById('chart-signal'), {
    type:'bar', data:{
      labels: config.ground_stations.map(g=>g.name),
      datasets:[{ label:'Signal %', data:new Array(8).fill(0),
        backgroundColor:config.ground_stations.map((_,i)=>`hsl(${200+i*15},70%,50%)`) }]
    }, options:{...chartCfg}
  });

  const satIds = Object.keys(config.satellites);
  charts.data = new Chart(document.getElementById('chart-data'), {
    type:'bar', data:{
      labels: satIds,
      datasets:[{ label:'Mbps', data:new Array(satIds.length).fill(0),
        backgroundColor: satIds.map(id=>config.satellites[id].color+'aa') }]
    }, options:{...chartCfg}
  });
}

// ── Telemetry Updates ─────────────────────────────────────────────
socket.on('telemetry', data => {
  telemetry = data.satellites;
  const ts = data.timestamp;

  // Header updates
  document.getElementById('hdr-time').textContent = ts.split(' ')[1];
  let totalData = 0;
  Object.values(telemetry).forEach(s => totalData += s.data_rate);
  document.getElementById('hdr-uplink').textContent = totalData.toFixed(0);

  // Climate score from first prediction
  if(predData.climate_score?.length)
    document.getElementById('hdr-cscore').textContent = predData.climate_score[0].toFixed(2);

  // Update satellite signals
  Object.entries(telemetry).forEach(([id, s]) => {
    const el = document.getElementById('sig-'+id);
    if(el){
      el.textContent = s.signal+'%';
      el.className = 'sat-sig '+(s.signal>60?'sig-good':s.signal>30?'sig-warn':'sig-bad');
    }
  });

  // Update charts
  updateSignalChart();
  updateDataChart();
  updateDetailPanel();
  updateAlerts(data.alerts);
  updateGlobePositions();
  updateClimateGauges();
});

function updateSignalChart() {
  if(!charts.signal) return;
  const gs = config.ground_stations;
  const sat = telemetry[selectedSat];
  if(!sat) return;
  const vals = gs.map(g => sat.signals?.[g.id] || 0);
  charts.signal.data.datasets[0].data = vals;
  charts.signal.update('none');
}

function updateDataChart() {
  if(!charts.data) return;
  const vals = Object.keys(config.satellites).map(id => telemetry[id]?.data_rate || 0);
  charts.data.data.datasets[0].data = vals;
  charts.data.update('none');
}

function updateDetailPanel() {
  const s = telemetry[selectedSat];
  const cfg = config.satellites[selectedSat];
  if(!s || !cfg) return;
  document.getElementById('detail-sat-name').textContent = cfg.full;
  document.getElementById('detail-body').innerHTML = `
    <div class="detail-row"><span class="dk">POSITION</span><span class="dv">${s.lat}°, ${s.lon}°</span></div>
    <div class="detail-row"><span class="dk">ALTITUDE</span><span class="dv">${cfg.alt.toLocaleString()} km</span></div>
    <div class="detail-row"><span class="dk">SIGNAL</span>
      <span class="dv ${s.signal>60?'good':s.signal>30?'warn':'bad'}">${s.signal}%</span></div>
    <div class="detail-row"><span class="dk">LINKED GS</span><span class="dv">${s.connected_gs||'NONE'}</span></div>
    <div class="detail-row"><span class="dk">DATA RATE</span><span class="dv">${s.data_rate} Mbps</span></div>
    <div class="detail-row"><span class="dk">BATTERY</span><span class="dv ${s.battery>50?'good':'warn'}">${s.battery}%</span></div>
    <div class="detail-row"><span class="dk">INT TEMP</span><span class="dv">${s.temp_internal}°C</span></div>
    <div class="detail-row"><span class="dk">SOLAR IRR</span><span class="dv">${s.sensors?.solar_irradiance} W/m²</span></div>
    <div class="detail-row"><span class="dk">MAG FIELD</span><span class="dv">${s.sensors?.magnetic_field} nT</span></div>
    <div class="detail-row"><span class="dk">PARTICLE FLUX</span><span class="dv">${s.sensors?.particle_flux} p/cm²s</span></div>
    <div class="detail-row"><span class="dk">STATUS</span>
      <span class="dv ${s.status==='NOMINAL'?'good':'warn'}">${s.status}</span></div>
    <div class="detail-row"><span class="dk">AGENCY</span><span class="dv">${cfg.agency}</span></div>
  `;
  // GS signal bars
  const gs = config.ground_stations;
  document.getElementById('gs-signal-bars').innerHTML = gs.map(g=>{
    const sig = s.signals?.[g.id]||0;
    const c = sig>60?'var(--green)':sig>30?'var(--orange)':'var(--red)';
    return `<div style="margin-bottom:5px">
      <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--dim)">${g.name}<span style="color:${c}">${sig}%</span></div>
      <div class="sig-bar"><div class="sig-fill" style="width:${sig}%;background:${c}"></div></div>
    </div>`;
  }).join('');
}

function updateAlerts(alerts) {
  document.getElementById('alert-log').innerHTML = (alerts||[]).slice().reverse().map(a=>`
    <div class="alert-item">
      <span class="alert-time">${a.time}</span>
      <span class="al-${a.level}">${a.msg}</span>
    </div>`).join('');
}

function updateClimateGauges() {
  if(!predData.temp_anomaly) return;
  const now = Math.abs(predData.temp_anomaly[0]);
  const storm = predData.storm_prob?.[0] || 0;
  const polar = Math.abs(predData.polar_ice_idx?.[0]||0);
  const climate = predData.climate_score?.[0]||0;
  const gauges = [
    {l:'Temp Anomaly', v:now, max:3, c:'var(--red)'},
    {l:'Storm Probability', v:storm, max:1, c:'var(--orange)'},
    {l:'Polar Ice Δ', v:polar, max:1, c:'var(--cyan)'},
    {l:'Climate Risk Score', v:climate, max:5, c:'var(--purple)'},
    {l:'Precip Index', v:Math.abs(predData.precip_idx?.[0]||0), max:2, c:'#4488ff'},
  ];
  document.getElementById('climate-gauges').innerHTML = gauges.map(g=>{
    const pct = Math.min(100, g.v/g.max*100);
    return `<div class="pred-gauge">
      <div class="pred-label"><span class="pk">${g.l}</span><span class="pval" style="color:${g.c}">${g.v.toFixed(3)}</span></div>
      <div class="gauge-bar"><div class="gauge-fill" style="width:${pct}%;background:${g.c}"></div></div>
    </div>`;
  }).join('');
}

// ── Planetary Status ───────────────────────────────────────────────
async function updatePlanetaryStatus() {
  const r = await fetch('/api/planetary_disturbance');
  const d = await r.json();
  document.getElementById('planet-rows').innerHTML = Object.entries(d).map(([k,v])=>`
    <div class="planet-row">
      <span class="pk">${k.replace(/_/g,' ')}</span>
      <span class="pv">${typeof v==='number'?v.toFixed(4):v}</span>
    </div>`).join('');
}

// ── Commands ───────────────────────────────────────────────────────
async function sendCommand() {
  const sat = document.getElementById('cmd-sat').value;
  const cmd = document.getElementById('cmd-type').value;
  await fetch('/api/command', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body:JSON.stringify({satellite:sat, command:cmd})
  });
  const log = document.getElementById('cmd-log');
  log.innerHTML = `<div class="cmd-entry">▶ ${new Date().toTimeString().slice(0,8)} [${sat}] ${cmd} → ACK</div>` + log.innerHTML;
}

// ── 3D Globe (Three.js) ────────────────────────────────────────────
let globeScene, globeCamera, globeRenderer, globeOrbit;
const satMeshes = {};
const gsMeshes = [];
const linkLines = [];

function initGlobe() {
  const canvas = document.getElementById('globe-canvas');
  globeScene = new THREE.Scene();
  globeCamera = new THREE.PerspectiveCamera(45, canvas.clientWidth/canvas.clientHeight, 0.1, 1000);
  globeCamera.position.set(0,0,3.2);
  globeRenderer = new THREE.WebGLRenderer({canvas, antialias:true, alpha:true});
  globeRenderer.setPixelRatio(devicePixelRatio);
  globeRenderer.setSize(canvas.clientWidth, canvas.clientHeight);

  // Earth
  const earth = new THREE.Mesh(
    new THREE.SphereGeometry(1,48,48),
    new THREE.MeshPhongMaterial({color:0x1a3a6a, emissive:0x050e20, shininess:30})
  );
  globeScene.add(earth);

  // Continents (simplified dots)
  const contGeo = new THREE.BufferGeometry();
  const contPts = [];
  const continentCoords = [
    // North America
    ...[...Array(40)].map(()=>[30+Math.random()*30, -60-Math.random()*50]),
    // Europe
    ...[...Array(30)].map(()=>[45+Math.random()*15, 5+Math.random()*30]),
    // Asia
    ...[...Array(60)].map(()=>[20+Math.random()*50, 60+Math.random()*80]),
    // Africa
    ...[...Array(40)].map(()=>[-25+Math.random()*50, 10+Math.random()*40]),
    // South America
    ...[...Array(30)].map(()=>[-35+Math.random()*30, -70+Math.random()*30]),
    // Australia
    ...[...Array(20)].map(()=>[-25+Math.random()*15, 115+Math.random()*30]),
  ];
  continentCoords.forEach(([lat,lon])=>{
    const phi=THREE.MathUtils.degToRad(90-lat), theta=THREE.MathUtils.degToRad(lon);
    const r=1.005;
    contPts.push(r*Math.sin(phi)*Math.cos(theta), r*Math.cos(phi), r*Math.sin(phi)*Math.sin(theta));
  });
  contGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(contPts),3));
  globeScene.add(new THREE.Points(contGeo, new THREE.PointsMaterial({color:0x22aa55, size:0.012})));

  // Atmosphere
  const atm = new THREE.Mesh(new THREE.SphereGeometry(1.06,32,32),
    new THREE.MeshBasicMaterial({color:0x1a5aff,transparent:true,opacity:.07,side:THREE.BackSide}));
  globeScene.add(atm);

  // Grid lines
  for(let lat=-60;lat<=60;lat+=30){
    const pts=[];
    for(let lon=0;lon<=360;lon+=5){
      const phi=THREE.MathUtils.degToRad(90-lat), theta=THREE.MathUtils.degToRad(lon);
      pts.push(new THREE.Vector3(1.01*Math.sin(phi)*Math.cos(theta),1.01*Math.cos(phi),1.01*Math.sin(phi)*Math.sin(theta)));
    }
    globeScene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({color:0x0a2a50,transparent:true,opacity:.4})));
  }

  // Lighting
  globeScene.add(new THREE.DirectionalLight(0xfff0d0,1.8).position.set(3,2,2));
  globeScene.add(new THREE.AmbientLight(0x0a1525,0.8));

  // Ground stations
  config.ground_stations.forEach(gs=>{
    const phi=THREE.MathUtils.degToRad(90-gs.lat), theta=THREE.MathUtils.degToRad(gs.lon);
    const p=new THREE.Vector3(1.015*Math.sin(phi)*Math.cos(theta),1.015*Math.cos(phi),1.015*Math.sin(phi)*Math.sin(theta));
    const m=new THREE.Mesh(new THREE.SphereGeometry(.018,8,8),new THREE.MeshBasicMaterial({color:0x00ffaa}));
    m.position.copy(p); globeScene.add(m); gsMeshes.push({mesh:m,gs,pos:p});
  });

  // Satellite meshes
  Object.keys(config.satellites).forEach(id=>{
    const col = parseInt(config.satellites[id].color.replace('#',''),16);
    const m=new THREE.Mesh(new THREE.SphereGeometry(.022,8,8),new THREE.MeshBasicMaterial({color:col}));
    globeScene.add(m); satMeshes[id]=m;
  });

  globeOrbit = {theta:0, phi:Math.PI/4};
  canvas.addEventListener('mousemove', e=>{
    if(e.buttons){ globeOrbit.theta -= e.movementX*.005; globeOrbit.phi = Math.max(.2,Math.min(Math.PI-.2,globeOrbit.phi-e.movementY*.005)); }
  });

  animateGlobe();
}

function latLonToVec(lat, lon, r=1.08) {
  const phi=THREE.MathUtils.degToRad(90-lat), theta=THREE.MathUtils.degToRad(lon);
  return new THREE.Vector3(r*Math.sin(phi)*Math.cos(theta), r*Math.cos(phi), r*Math.sin(phi)*Math.sin(theta));
}

function updateGlobePositions() {
  // Remove old link lines
  linkLines.forEach(l=>globeScene.remove(l)); linkLines.length=0;

  Object.entries(telemetry).forEach(([id,s])=>{
    const m=satMeshes[id]; if(!m) return;
    const p=latLonToVec(s.lat, s.lon);
    m.position.copy(p);

    // Draw link line to best ground station
    const bestGS = gsMeshes.find(g=>g.gs.id===s.connected_gs);
    if(bestGS && s.signal>20){
      const pts=[p.clone(), bestGS.pos.clone()];
      const line=new THREE.Line(
        new THREE.BufferGeometry().setFromPoints(pts),
        new THREE.LineBasicMaterial({color:s.signal>60?0x00ff88:0xff8822,transparent:true,opacity:.4+s.signal*.004})
      );
      globeScene.add(line); linkLines.push(line);
    }
  });
}

function animateGlobe() {
  requestAnimationFrame(animateGlobe);
  const canvas=document.getElementById('globe-canvas');
  if(canvas.clientWidth !== globeRenderer.domElement.width || canvas.clientHeight !== globeRenderer.domElement.height){
    globeRenderer.setSize(canvas.clientWidth, canvas.clientHeight);
    globeCamera.aspect=canvas.clientWidth/canvas.clientHeight;
    globeCamera.updateProjectionMatrix();
  }
  globeOrbit.theta+=0.002;
  globeCamera.position.set(
    3.2*Math.sin(globeOrbit.phi)*Math.sin(globeOrbit.theta),
    3.2*Math.cos(globeOrbit.phi),
    3.2*Math.sin(globeOrbit.phi)*Math.cos(globeOrbit.theta)
  );
  globeCamera.lookAt(0,0,0);
  globeRenderer.render(globeScene, globeCamera);
}

// ── Start ──────────────────────────────────────────────────────────
init().then(()=>{ initGlobe(); });
</script>
</body>
</html>"""

@app.route('/')
def index():
    return render_template_string(DASHBOARD_HTML)

if __name__ == '__main__':
    print("\n🚀 Starting Satellite Ground Control + Climate AI Server...")
    print("🌐 Dashboard: http://localhost:5000\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
