from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import os
import sys
import time
import signal
import atexit
import threading
import psutil
import time

# LIBRERIE ULTRA AVANZATE - LIVELLO PROFESSIONALE
from skimage import filters, feature, morphology, measure, segmentation
from skimage.filters import sobel, roberts, prewitt, scharr, gabor
from skimage.feature import canny
from skimage.segmentation import watershed, active_contour

# LIBRERIE MEDICAL IMAGING RIMOSSE PER SPAZIO DISCO
# import SimpleITK as sitk  # RIMOSSO
# import itk  # RIMOSSO
# import monai  # RIMOSSO
# from monai.transforms import (  # RIMOSSO
#     Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, 
#     RandFlip, RandRotate, RandZoom, ToTensor
# )
# from monai.networks.nets import UNet  # RIMOSSO
# from monai.losses import DiceLoss  # RIMOSSO

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'drawing_app_secret_key'

# Configurazione cartelle - supporta directory personalizzata utente
def get_user_base_directory():
    """Ottiene la directory base scelta dall'utente o usa default"""
    # Priorità: 1) Variabile ambiente, 2) File config, 3) Default
    user_dir = os.environ.get('DRAWING_APP_HOME')
    
    if not user_dir:
        config_file = 'app_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    user_dir = config.get('base_directory')
            except:
                pass
    
    if not user_dir:
        user_dir = os.path.expanduser("~/hacker_drawing_app")
    
    return user_dir

# Configurazione percorsi dinamici
BASE_DIR = get_user_base_directory()
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

# Crea tutte le directory necessarie
for directory in [BASE_DIR, TEMPLATES_DIR, UPLOAD_DIR, RESULTS_DIR, CONFIG_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configurazione app
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
app.config['RESULTS_FOLDER'] = RESULTS_DIR

# Variabili globali per il monitoraggio
processing_stats = {
    'total_images': 0,
    'successful_detections': 0,
    'failed_detections': 0,
    'average_processing_time': 0.0,
    'server_start_time': time.time(),
    'active_processes': 0,
    'memory_usage': 0.0
}

# Funzione di pulizia all'uscita
def cleanup():
    """Pulisce i file temporanei all'uscita"""
    print("🧹 Pulizia file temporanei...")
    try:
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        print("✅ File temporanei rimossi")
    except Exception as e:
        print(f"❌ Errore nella pulizia: {e}")

# Registra cleanup all'uscita
atexit.register(cleanup)

# Funzione di monitoraggio sistema
def update_system_stats():
    """Aggiorna le statistiche del sistema"""
    try:
        processing_stats['memory_usage'] = psutil.virtual_memory().percent
        processing_stats['active_processes'] = len(psutil.pids())
    except:
        pass

# Funzione di contour detection avanzata
def detect_contours_advanced(image_data, sensitivity='medio', method='auto'):
    """
    Rileva i contorni nell'immagine usando metodi avanzati
    """
    try:
        processing_stats['active_processes'] += 1
        start_time = time.time()
        
        # Decodifica immagine
        image_data = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_data))
        image_array = np.array(image)
        
        # Converti in grayscale se necessario
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Configurazione sensibilità
        sensitivity_params = {
            'basso': {'blur': 9, 'thresh1': 50, 'thresh2': 150},
            'medio': {'blur': 7, 'thresh1': 100, 'thresh2': 200},
            'alto': {'blur': 5, 'thresh1': 150, 'thresh2': 250}
        }
        
        params = sensitivity_params.get(sensitivity, sensitivity_params['medio'])
        
        # Applica filtri avanzati
        blurred = cv2.GaussianBlur(gray, (params['blur'], params['blur']), 0)
        
        # Edge detection con Canny
        edges = cv2.Canny(blurred, params['thresh1'], params['thresh2'])
        
        # Trova contorni
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtra contorni piccoli
        min_area = 100
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # Crea immagine con contorni
        contour_image = image_array.copy()
        cv2.drawContours(contour_image, filtered_contours, -1, (0, 255, 0), 2)
        
        # Converti in base64
        _, buffer = cv2.imencode('.jpg', contour_image)
        contour_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Statistiche
        processing_time = time.time() - start_time
        processing_stats['total_images'] += 1
        processing_stats['successful_detections'] += 1
        processing_stats['average_processing_time'] = (
            (processing_stats['average_processing_time'] * (processing_stats['total_images'] - 1) + processing_time) 
            / processing_stats['total_images']
        )
        
        return {
            'success': True,
            'contour_image': f"data:image/jpeg;base64,{contour_image_b64}",
            'contour_points': len(filtered_contours),
            'processing_time': round(processing_time, 3),
            'stats': {
                'total_contours': len(contours),
                'filtered_contours': len(filtered_contours),
                'image_shape': image_array.shape,
                'method': method,
                'sensitivity': sensitivity
            }
        }
        
    except Exception as e:
        processing_stats['total_images'] += 1
        processing_stats['failed_detections'] += 1
        return {
            'success': False,
            'error': str(e),
            'processing_time': round(time.time() - start_time, 3) if 'start_time' in locals() else 0
        }
    finally:
        processing_stats['active_processes'] -= 1

# Route principale
@app.route('/')
def index():
    """Pagina principale dell'applicazione"""
    return render_template('index.html')

# API per contour detection
@app.route('/detect_contours', methods=['POST'])
def detect_contours_api():
    """API endpoint per contour detection"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'Nessuna immagine fornita'}), 400
        
        image_data = data['image']
        sensitivity = data.get('sensitivity', 'medio')
        method = data.get('method', 'auto')
        
        # Processa immagine
        result = detect_contours_advanced(image_data, sensitivity, method)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'error': f'Errore del server: {str(e)}'}), 500

# API per statistiche
@app.route('/stats', methods=['GET'])
def get_stats():
    """API endpoint per statistiche del sistema"""
    update_system_stats()
    uptime = time.time() - processing_stats['server_start_time']
    
    stats = {
        **processing_stats,
        'uptime_seconds': round(uptime, 2),
        'uptime_formatted': f"{int(uptime//3600)}h {int((uptime%3600)//60)}m {int(uptime%60)}s"
    }
    
    return jsonify(stats)

# API per health check
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0.0'
    })

# Route per test
@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'Hacker Drawing App API is running!',
        'endpoints': [
            '/detect_contours (POST)',
            '/stats (GET)',
            '/health (GET)',
            '/test (GET)'
        ]
    })

# Gestione errori
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File troppo grande'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint non trovato'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Errore interno del server'}), 500

# Main execution
if __name__ == '__main__':
    # Configurazione per PythonAnywhere
    if 'PYTHONANYWHERE_DOMAIN' in os.environ:
        app.config['DEBUG'] = False
        app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'pythonanywhere-secret')
        print("🚀 Running on PythonAnywhere")
    else:
        print("🏠 Running locally")
    
    # Configurazione
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '127.0.0.1')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print(f"🚀 Starting HACKER DRAWING APP on {host}:{port}")
    if 'PYTHONANYWHERE_DOMAIN' in os.environ:
        print(f"🌐 PythonAnywhere URL: https://{os.environ.get('PYTHONANYWHERE_DOMAIN')}.pythonanywhere.com")
    print(f"👨‍💻 Created by: Stefano Luciano")
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📊 Initial Stats: {processing_stats}")
    
    # Avvia server
    app.run(host=host, port=port, debug=debug)
