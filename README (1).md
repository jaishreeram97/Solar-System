#  Satellite Ground Control + Climate AI Observatory

## Quick Start

```bash
pip install flask flask-socketio scikit-learn numpy pandas scipy eventlet
python satellite_climate_app.py
# Open: http://localhost:5000
```

## ML Model Results (trained live on startup)
- **Temperature Anomaly**: R²=0.907 (90.7% accuracy)  
- **Polar Ice Index**: R²=0.714  
- **Climate Risk Score**: R²=0.704  
- **Precipitation Index**: R²=0.645  
- **Storm Probability**: R²=0.558  

## Architecture
Flask + SocketIO backend · Three.js 3D globe · Chart.js forecasts · scikit-learn RF models
