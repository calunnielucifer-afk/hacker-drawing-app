# 🎨 HACKER DRAWING APP v2.0

**Advanced Contour Detection System**

Created by: Stefano Luciano  
Host: stefano-tools.free.nf

## 📋 DESCRIPTION

Hacker Drawing App è un sistema avanzato per il rilevamento di contorni con interfaccia stile hacker. Include sia versione desktop che API web.

## 🚀 FEATURES

- ✅ **Ultra-advanced contour detection** con 5 livelli sensibilità
- ✅ **Border removal filter** automatico
- ✅ **Real-time countdown overlay** 
- ✅ **Hacker-style interface** con colori neon
- ✅ **Web API integration** per processing remoto
- ✅ **Local/Web processing toggle**
- ✅ **Statistics and export** functionality

## 📁 FILES

- `drawing_app.py` - Flask API web server
- `requirements.txt` - Python dependencies
- `Procfile` - Render deployment config
- `runtime.txt` - Python version specification

## 🌐 DEPLOYMENT

### Render.com (Recommended)
1. Connect this repository to Render
2. Web Service with:
   - Build: `pip install -r requirements.txt`
   - Start: `python drawing_app.py`
   - Port: 5000

### Environment Variables
```
FLASK_APP=drawing_app.py
FLASK_ENV=production
PORT=5000
```

## 🎮 DESKTOP APP

Use `drawing_app_desktop.py` for local processing with:
- 💻 Local mode (offline processing)
- 🌐 Web API mode (calls this server)
- ⏰ Countdown overlay
- 🎨 Hacker interface

## 📡 API ENDPOINTS

### POST /detect_contours
```json
{
  "image": "base64_image_data",
  "sensitivity": "medio"
}
```

### Response
```json
{
  "success": true,
  "contour_points": [...],
  "stats": {...}
}
```

## 🎨 HACKER THEME

- Colors: Green (#00ff41), Cyan, Black, Red
- Font: Courier New
- Style: Terminal/Matrix aesthetic

## 👨‍💻 AUTHOR

**Stefano Luciano**  
Computer Vision Expert  
stefano-tools.free.nf

## 📜 LICENSE

MIT License - Feel free to use and modify!
