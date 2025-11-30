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

# LIBRERIE MEDICAL IMAGING LIVELLO GOOGLE/DEEPMIND
import SimpleITK as sitk
import itk
import monai
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity, 
    RandFlip, RandRotate, RandZoom, ToTensor
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss

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
        # Default: directory utente/Documents/DrawingApp
        user_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'DrawingApp')
    
    # Crea la directory se non esiste
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

BASE_DIRECTORY = get_user_base_directory()
UPLOAD_FOLDER = os.path.join(BASE_DIRECTORY, 'uploads')
CONTOURS_FOLDER = os.path.join(BASE_DIRECTORY, 'contours')
TEMPLATES_FOLDER = os.path.join(BASE_DIRECTORY, 'templates')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['BASE_DIRECTORY'] = BASE_DIRECTORY

# Crea le cartelle necessarie
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CONTOURS_FOLDER, exist_ok=True)
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
    
# Fix per PyInstaller - determina il path corretto per i templates
if getattr(sys, 'frozen', False):
    # Se è un eseguibile PyInstaller
    base_path = sys._MEIPASS
    template_folder = os.path.join(base_path, 'templates')
    
    # Se i templates non esistono in MEIPASS, copiali dalla directory utente
    if not os.path.exists(template_folder) or not os.listdir(template_folder):
        print(f"Templates non trovati in {template_folder}, copio dalla directory utente...")
        user_templates = os.path.join(BASE_DIRECTORY, 'templates')
        if os.path.exists(user_templates):
            import shutil
            os.makedirs(template_folder, exist_ok=True)
            for file in os.listdir(user_templates):
                src = os.path.join(user_templates, file)
                dst = os.path.join(template_folder, file)
                shutil.copy2(src, dst)
            print(f"Templates copiati da {user_templates} a {template_folder}")
        else:
            print(f"ERRORE: Templates non trovati neanche in {user_templates}")
else:
    # Se è in sviluppo - usa la directory utente configurata
    template_folder = TEMPLATES_FOLDER

app.template_folder = template_folder

# Gestione processi e pulizia
active_processes = []
active_threads = []

def cleanup_processes():
    """Pulisce tutti i processi e thread attivi"""
    print("🧹 Pulizia processi in corso...")
    
    # Termina thread attivi
    for thread in active_threads:
        if thread.is_alive():
            print(f"Terminazione thread: {thread.name}")
            # Non si può forzare terminazione thread in Python, ma li segnaliamo
            thread.stop = True
    
    # Chiudi processi figli
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            print(f"Terminazione processo figlio: {child.pid}")
            child.terminate()
            child.wait(timeout=3)
        except psutil.NoSuchProcess:
            pass
        except psutil.TimeoutExpired:
            print(f"Force kill processo: {child.pid}")
            child.kill()
    
    # Pulizia file temporanei
    try:
        temp_files = []
        for root, dirs, files in os.walk(UPLOAD_FOLDER):
            for file in files:
                if file.startswith('temp_') or file.endswith('.tmp'):
                    temp_files.append(os.path.join(root, file))
        
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
                print(f"Rimosso file temporaneo: {temp_file}")
            except:
                pass
    except:
        pass
    
    print("✅ Pulizia completata")

def signal_handler(signum, frame):
    """Gestore di segnali per pulizia elegante"""
    print(f"Ricevuto segnale {signum}, avvio pulizia...")
    cleanup_processes()
    sys.exit(0)

# Registra gestori di segnali e cleanup
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
atexit.register(cleanup_processes)

# Monitoraggio CPU
def monitor_cpu_usage():
    """Monitora l'uso della CPU durante il processing"""
    process = psutil.Process()
    cpu_percent = process.cpu_percent()
    memory_percent = process.memory_percent()
    print(f"📊 CPU: {cpu_percent:.1f}% | RAM: {memory_percent:.1f}%")
    return cpu_percent, memory_percent

# Debug: verifica che il template esista
template_path = os.path.join(template_folder, 'drawing_opera.html')
if not os.path.exists(template_path):
    print(f"❌ ERRORE: Template non trovato: {template_path}")
    print(f"Directory corrente: {os.getcwd()}")
    print(f"Directory template: {os.path.abspath(template_folder)}")
    print(f"Base path: {base_path if 'base_path' in locals() else 'N/A'}")
    print(f"Directory base utente: {BASE_DIRECTORY}")
else:
    print(f"✅ Template trovato: {template_path}")
    print(f"📁 Directory base utente: {BASE_DIRECTORY}")

@app.route('/')
def index():
    """Pagina principale stile Opera GX"""
    return render_template('drawing_opera.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content response for favicon

class ImageProcessor:
    def __init__(self):
        self.min_perimeter = 3  # Ridotto per trovare più contorni
        self.min_area = 5      # Ridotto per trovare più contorni
        self.large_perimeter_threshold = 20000  # Aumentato per accettare contorni più grandi
        self.large_area_threshold = 200000     # Aumentato per accettare contorni più grandi
    
    def decode_image(self, image_data):
        """Decodifica l'immagine base64"""
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError('Impossibile decodificare l\'immagine')
        
        return image
    
    def preprocess_image(self, image):
        """Preprocessa l'immagine per il rilevamento contorni"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cloned_image = gray.copy()
        enhanced = cv2.equalizeHist(cloned_image)
        _, binary_result = cv2.threshold(cloned_image, 127, 255, cv2.THRESH_BINARY)
        
        return gray, enhanced, binary_result
    
    def detect_contours_multi_approach(self, gray):
        """SISTEMA ULTRA AVANZATO - rilevamento precisione estrema"""
        print(f"🚀 AVVIO SISTEMA ULTRA AVANZATO su immagine {gray.shape}")
        
        # Monitoraggio CPU inizio
        cpu_start, mem_start = monitor_cpu_usage()
        
        all_contours = []
        
        # METODO 0: BASE AGGRESSIVO
        print("⚡ METODO 0 - BASE AGGRESSIVO...")
        
        # 0.1 Multiple Gaussian Blur
        blur_sizes = [(3,3), (5,5), (7,7)]
        for size in blur_sizes:
            blurred = cv2.GaussianBlur(gray, size, 0)
            
            # 0.2 Multiple Fixed Thresholds
            thresholds = list(range(30, 201, 20))  # 30, 50, 70, ..., 190
            for thresh in thresholds:
                _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                all_contours.extend(contours)
            
            # 0.3 Multiple Adaptive Thresholds
            adaptive_params = [
                (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
                (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 5),
                (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 2),
                (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 5),
                (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
                (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 5),
                (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 2),
                (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 5)
            ]
            for method, block_size, C in adaptive_params:
                try:
                    adaptive = cv2.adaptiveThreshold(blurred, 255, method, cv2.THRESH_BINARY, block_size, C)
                    contours, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    all_contours.extend(contours)
                except:
                    continue
        
        # 0.4 Multiple Canny
        canny_params = [
            (30, 100), (50, 150), (70, 200), (100, 200),
            (30, 150), (50, 200), (70, 250), (100, 250)
        ]
        for low, high in canny_params:
            edges = cv2.Canny(blurred, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        print(f"⚡ Base Aggressivo trovati: {len(all_contours)} contorni")
        
        # METODO 1: Scikit-Image Advanced
        print("🔬 METODO 1 - Scikit-Image Advanced...")
        try:
            gray_normalized = gray.astype(np.float32) / 255.0
            
            # 1.1 Multiple Edge Detection
            edges_methods = [sobel, roberts, prewitt, scharr]
            for method in edges_methods:
                try:
                    edges = method(gray_normalized)
                    
                    # Multiple thresholds
                    thresholds = [filters.threshold_otsu(edges), 
                                 filters.threshold_yen(edges),
                                 filters.threshold_li(edges),
                                 filters.threshold_mean(edges),
                                 filters.threshold_minimum(edges)]
                    
                    for thresh_func in thresholds:
                        try:
                            binary = edges > thresh_func
                            contours = self._skimage_to_opencv_contours(binary)
                            all_contours.extend(contours)
                        except:
                            continue
                except:
                    continue
            
            # 1.2 Multiple Canny
            for sigma in [0.5, 1.0, 1.5, 2.0]:
                try:
                    canny_edges = canny(gray_normalized, sigma=sigma)
                    contours = self._skimage_to_opencv_contours(canny_edges)
                    all_contours.extend(contours)
                except:
                    continue
            
            # 1.3 Gabor Filters
            for frequency in [0.1, 0.3, 0.5]:
                try:
                    real, imag = gabor(gray_normalized, frequency=frequency)
                    edges = np.sqrt(real**2 + imag**2)
                    thresh = filters.threshold_otsu(edges)
                    binary = edges > thresh
                    contours = self._skimage_to_opencv_contours(binary)
                    all_contours.extend(contours)
                except:
                    continue
            
            # 1.4 Advanced Morphology
            morph_operations = ['erosion', 'dilation', 'opening', 'closing']
            for operation in morph_operations:
                try:
                    if operation == 'erosion':
                        processed = morphology.erosion(gray_normalized, morphology.disk(2))
                    elif operation == 'dilation':
                        processed = morphology.dilation(gray_normalized, morphology.disk(2))
                    elif operation == 'opening':
                        processed = morphology.opening(gray_normalized, morphology.disk(2))
                    else:  # closing
                        processed = morphology.closing(gray_normalized, morphology.disk(2))
                    
                    edges = sobel(processed)
                    thresh = filters.threshold_otsu(edges)
                    binary = edges > thresh
                    contours = self._skimage_to_opencv_contours(binary)
                    all_contours.extend(contours)
                except:
                    continue
            
            # 1.5 Multi-Otsu
            try:
                thresholds_multi = filters.threshold_multiotsu(gray_normalized, classes=4)
                binary_multi = gray_normalized > thresholds_multi[1]  # Use middle threshold
                contours = self._skimage_to_opencv_contours(binary_multi)
                all_contours.extend(contours)
            except:
                pass
            
        except Exception as e:
            print(f"❌ Scikit-Image advanced error: {e}")
        
        print(f"🔬 TOTALE PRIMA FILTRAGGIO: {len(all_contours)}")
        
        # Filtraggio professionale
        filtered_contours = self._filter_contours_professional(all_contours)
        
        print(f"✅ CONTORNI FINALI: {len(filtered_contours)}")
        
        # Monitoraggio CPU fine
        cpu_end, mem_end = monitor_cpu_usage()
        print(f"📊 CPU: {cpu_end - cpu_start:.1f}% | RAM: {mem_end - mem_start:.1f}%")
        
        return filtered_contours
    
    def _merge_similar_contours(self, contours_list):
        """Fonde contorni che seguono la stessa direzione"""
        if not contours_list:
            return []
        
        print(f"🔀 FUSIONE di {len(contours_list)} contorni...")
        
        # Calcola direzione e posizione per ogni contorno
        contour_info = []
        for i, contour in enumerate(contours_list):
            if len(contour) < 3:
                continue
            
            area = cv2.contourArea(contour)
            # Controlla se area è un array numpy e convertilo a scalare
            if hasattr(area, '__len__') and len(area) > 1:
                area = float(area[0] if len(area) == 1 else 0)
            
            if area < 5:  # Ignora contorni troppo piccoli
                continue
            
            # Calcola bounding box e centro
            x, y, w, h = cv2.boundingRect(contour)
            center = (x + w//2, y + h//2)
            
            # Calcola direzione principale
            direction = self._get_contour_direction(contour)
            
            contour_info.append({
                'contour': contour,
                'center': center,
                'direction': direction,
                'area': area,
                'bbox': (x, y, w, h),
                'index': i
            })
        
        # Fonde contorni simili
        merged = []
        used = set()
        
        for i, current in enumerate(contour_info):
            if i in used:
                continue
            
            # Trova contorni simili da fondere
            similar_group = [current]
            used.add(i)
            
            for j, other in enumerate(contour_info):
                if j in used or i == j:
                    continue
                
                # Controlla se sono simili
                if self._are_contours_similar(current, other):
                    similar_group.append(other)
                    used.add(j)
            
            # Fonde il gruppo in un unico contorno
            if len(similar_group) > 1:
                merged_contour = self._merge_contour_group(similar_group)
                merged.append(merged_contour)
            else:
                merged.append(current['contour'])
        
        print(f"🔀 FUSIONE completata: {len(contour_info)} -> {len(merged)} contorni")
        return merged
    
    def _get_contour_direction(self, contour):
        """Calcola la direzione principale di un contorno"""
        if len(contour) < 2:
            return 0
        
        # Calcola l'angolo del bounding box
        rect = cv2.minAreaRect(contour)
        angle = rect[2]
        
        # Normalizza angolo tra 0 e 180
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        
        return angle
    
    def _are_contours_similar(self, c1, c2):
        """Verifica se due contorni sono simili (stessa direzione e vicini)"""
        # Distanza tra centri
        dist = np.sqrt((c1['center'][0] - c2['center'][0])**2 + 
                      (c1['center'][1] - c2['center'][1])**2)
        
        # Se sono troppo distanti, non sono simili
        max_dist = max(c1['bbox'][2], c1['bbox'][3], c2['bbox'][2], c2['bbox'][3]) * 2
        if dist > max_dist:
            return False
        
        # Differenza di direzione
        angle_diff = abs(c1['direction'] - c2['direction'])
        if angle_diff > 90:
            angle_diff = 180 - angle_diff
        
        # Se hanno direzioni simili (entro 30 gradi)
        return angle_diff < 30
    
    def _merge_contour_group(self, contour_group):
        """Fonde un gruppo di contorni simili in uno solo"""
        if len(contour_group) == 1:
            return contour_group[0]['contour']
        
        # Unisci tutti i punti dei contorni
        all_points = []
        for item in contour_group:
            contour = item['contour']
            try:
                for point in contour:
                    if len(point) > 0:
                        all_points.append(point[0])
            except (IndexError, TypeError) as e:
                print(f"Skip punto invalido nel merge: {e}")
                continue
        
        if not all_points:
            return contour_group[0]['contour']
        
        try:
            # Calcola il convex hull di tutti i punti
            all_points = np.array(all_points, dtype=np.float32)
            if len(all_points) < 3:
                return contour_group[0]['contour']
            
            hull = cv2.convexHull(all_points)
            return hull
        except Exception as e:
            print(f"Errore nel merge contorni: {e}")
            return contour_group[0]['contour']
    
    def _filter_merged_contours(self, contours_list):
        """Filtra i contorni fusi mantenendo quelli migliori"""
        if not contours_list:
            return []
        
        print(f"🔍 Filtraggio finale di {len(contours_list)} contorni fusi...")
        
        # Calcola metriche
        valid_contours = []
        for contour in contours_list:
            if len(contour) < 3:
                continue
            
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Controlla se area/perimeter sono array numpy e convertili a scalari
            if hasattr(area, '__len__') and len(area) > 1:
                area = float(area[0] if len(area) == 1 else 0)
            if hasattr(perimeter, '__len__') and len(perimeter) > 1:
                perimeter = float(perimeter[0] if len(perimeter) == 1 else 0)
            
            # Filtri base
            if area < 10 or perimeter < 5:
                continue
            
            # Compactness
            if perimeter == 0:
                continue
            compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
            if compactness > 1000:  # Soglia per contorni ragionevoli
                continue
            
            valid_contours.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'compactness': compactness
            })
        
        # Ordina per area (mantiene linee principali)
        valid_contours.sort(key=lambda x: x['area'], reverse=True)
        
        # Prendi i migliori (limite a 50)
        final_contours = [item['contour'] for item in valid_contours[:50]]
        
        print(f"✅ Filtraggio finale: {len(contours_list)} -> {len(final_contours)} contorni")
        return final_contours
    
    def _filter_contours_professional(self, contours_list):
        """FILTRAGGIO ULTRA-AGGRESSIVO - MANTIENE QUASI TUTTO"""
        if not contours_list:
            return []
        
        print(f"🔍 Filtraggio ULTRA-AGGRESSIVO di {len(contours_list)} contorni...")
        
        # Calcola metriche MINIME
        contour_data = []
        for i, contour in enumerate(contours_list):
            if len(contour) < 3:
                continue
                
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # FILTRI QUASINESSUNI - solo i veramente impossibili
            if area < 0.1 or perimeter < 0.5:
                continue
            
            # Calcola compactness ma con soglia ENORME
            compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
            
            # Rimuovi solo contorni MATHEMATICAMENTE impossibili
            if compactness > 10000:  # Soglia enormemente alta
                continue
            
            contour_data.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'compactness': compactness,
                'index': i
            })
        
        # Rimuovi solo duplicati OVVI
        unique_contours = self._remove_only_obvious_duplicates(contour_data)
        
        # Ordina per area (mantiene più contorni possibili)
        unique_contours.sort(key=lambda x: x['area'], reverse=True)
        
        # Prendi TUTTI (o massimo 1000 per performance)
        final_contours = [item['contour'] for item in unique_contours[:1000]]
        
        print(f"✅ Filtro ULTRA-AGGRESSIVO: {len(contours_list)} -> {len(final_contours)} contorni")
        return final_contours
    
    def _remove_only_obvious_duplicates(self, contour_data):
        """Rimuove SOLO duplicati molto evidenti"""
        if len(contour_data) <= 1:
            return [item['contour'] for item in contour_data]
        
        unique = []
        for i, current in enumerate(contour_data):
            is_duplicate = False
            
            for j, existing in enumerate(unique):
                # Confronto MOLTO permissivo
                area_diff = abs(current['area'] - existing['area']) / max(current['area'], existing['area'], 1)
                perimeter_diff = abs(current['perimeter'] - existing['perimeter']) / max(current['perimeter'], existing['perimeter'], 1)
                
                # Solo se sono IDENTICI (90% similitudine)
                if area_diff < 0.1 and perimeter_diff < 0.1:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(current)
        
        return unique
    
    def _filter_contours_simple(self, contours_list):
        """FILTRAGGIO SEMPLICE E VELOCE"""
        if not contours_list:
            return []
        
        print(f"🔍 Filtraggio SEMPLICE di {len(contours_list)} contorni...")
        
        # Filtri base solo
        filtered = []
        for contour in contours_list:
            if len(contour) < 3:
                continue
                
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # FILTRI QUASINESSUNI - solo i veramente impossibili
            if area < 0.1 or perimeter < 0.5:
                continue
                
            compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
            if compactness > 5000:  # Soglia molto alta
                continue
            
            filtered.append(contour)
        
        # Rimuovi duplicati ovvi
        unique = self._remove_obvious_duplicates_fast(filtered)
        
        # Ordina per area e prendi i migliori
        unique.sort(key=lambda c: cv2.contourArea(c), reverse=True)
        
        # Limite a 100 per performance
        final_contours = unique[:100]
        
        print(f"✅ Filtro SEMPLICE: {len(contours_list)} -> {len(final_contours)} contorni")
        return final_contours
    
    def _remove_obvious_duplicates_fast(self, contours):
        """Rimuove duplicati in modo VELOCE"""
        if len(contours) <= 1:
            return contours
        
        unique = []
        for i, current in enumerate(contours):
            is_duplicate = False
            current_area = cv2.contourArea(current)
            
            for j, existing in enumerate(unique):
                existing_area = cv2.contourArea(existing)
                
                # Confronto semplice per area
                area_diff = abs(current_area - existing_area) / max(current_area, existing_area, 1)
                if area_diff < 0.1:  # Meno del 10% di differenza
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(current)
        
        return unique
    
    def _skimage_to_opencv_contours(self, binary_mask):
        """Converte maschera binaria scikit-image in contorni OpenCV"""
        # Converti in uint8 0-255 per OpenCV
        binary_uint8 = (binary_mask * 255).astype(np.uint8)
        
        # Trova contorni con OpenCV
        contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def process_image(self, image_data):
        """Processa l'immagine per rilevare i contorni - VERSIONE CHE FUNZIONAVA"""
        try:
            # Decodifica l'immagine
            image_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Impossibile decodificare l\'immagine'}
            
            # Preprocessa l'immagine
            gray, enhanced, binary = self.preprocess_image(image)
            
            # Rileva i contorni
            contours = self.detect_contours_multi_approach(gray)
            
            # FILTRO ANTI-BROADCAST ERROR
            valid_contours = []
            for contour in contours:
                if contour is not None and len(contour) > 0:
                    try:
                        # Verifica che il contorno abbia la forma corretta
                        if contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
                            # Verifica che tutti i punti siano validi
                            valid_points = []
                            for point in contour:
                                if len(point) == 1 and len(point[0]) == 2:
                                    x, y = int(point[0][0]), int(point[0][1])
                                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                                        valid_points.append([x, y])
                            
                            if len(valid_points) >= 3:  # Almeno 3 punti per un contorno valido
                                # Converti in formato numpy corretto
                                contour_array = np.array(valid_points, dtype=np.int32).reshape(-1, 1, 2)
                                valid_contours.append(contour_array)
                    except Exception as e:
                        print(f"❌ Contorno saltato (errore): {e}")
                        continue
            
            print(f"🛡️ Contorni validi dopo filtro: {len(valid_contours)}")
            
            if not valid_contours:
                return {'error': 'Nessun contorno valido rilevato'}
            
            # Converti i contorni in punti per il frontend
            contour_points = []
            for contour in valid_contours:
                points = []
                for point in contour:
                    points.append([int(point[0][0]), int(point[0][1])])
                if points:
                    contour_points.append(points)
            
            # Crea l'immagine con i contorni
            contours_image = self.draw_contours(image, valid_contours)
            
            # Calcola le statistiche
            total_points = sum(len(contour) for contour in contour_points)
            avg_points = total_points / len(contour_points) if contour_points else 0
            
            stats = {
                'total_contours': len(contour_points),
                'total_points': total_points,
                'avg_points_per_contour': round(avg_points, 1)
            }
            
            # Converti l'immagine in base64 con validazione
            try:
                _, buffer = cv2.imencode('.jpg', contours_image)
                image_bytes = buffer.tobytes()
                contours_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Assicurati che il prefisso sia corretto
                if not contours_image_b64.startswith('data:image/'):
                    contours_image_b64 = f"data:image/jpeg;base64,{contours_image_b64}"
                
                # Rimuovi eventuali duplicati del prefisso
                if contours_image_b64.count('data:image/') > 1:
                    parts = contours_image_b64.split('data:image/')
                    contours_image_b64 = f"data:image/jpeg;base64,{parts[-1]}"
                    
            except Exception as e:
                print(f"Errore codifica immagine: {e}")
                # Immagine vuota come fallback
                contours_image_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A8A"
            
            return {
                'success': True,
                'contour_points': contour_points,
                'contours_image': contours_image_b64,
                'image_size': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                },
                'stats': stats
            }
            
        except Exception as e:
            print(f"❌ Errore nel processamento: {e}")
            return {'error': f'Errore nel processamento: {str(e)}'}
    
    def draw_contours(self, image, contours):
        """Disegna i contorni sull'immagine in modo sicuro"""
        result = image.copy()
        
        try:
            # Disegna solo contorni con dimensioni corrette
            valid_contours = []
            for contour in contours:
                if contour is not None and len(contour) > 0:
                    if contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2:
                        valid_contours.append(contour)
            
            if valid_contours:
                cv2.drawContours(result, valid_contours, -1, (0, 255, 0), 2)
                
        except Exception as e:
            print(f"❌ Errore nel disegno contorni: {e}")
            # Restituisci l'immagine originale senza contorni
            return image.copy()
        
        return result
    
    def process_image_simple(self, image_data):
        """Processa l'immagine per rilevare i contorni"""
        try:
            # Decodifica l'immagine
            image_data = base64.b64decode(image_data.split(',')[1])
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'error': 'Impossibile decodificare l\'immagine'}
            
            # Preprocessa l'immagine
            gray, enhanced, binary = self.preprocess_image(image)
            
            # Rileva i contorni
            contours = self.detect_contours_multi_approach(gray)
            
            # FILTRO ANTIOERROR: Verifica che ci siano contorni validi
            valid_contours = []
            for contour in contours:
                if contour is not None and len(contour) > 0:
                    # Verifica che il contorno abbia punti validi
                    test_mask = np.zeros(gray.shape, dtype=np.uint8)
                    try:
                        cv2.drawContours(test_mask, [contour], -1, 255, 1)
                        # Se drawContours funziona, il contorno è valido
                        valid_contours.append(contour)
                    except:
                        continue
            
            print(f"🛡️ Contorni validi dopo filtro: {len(valid_contours)}")
            
            if not valid_contours:
                return {'error': 'Nessun contorno valido rilevato'}
            
            # Converti i contorni in punti
            contour_points = []
            for contour in valid_contours:
                points = []
                for point in contour:
                    points.append([int(point[0][0]), int(point[0][1])])
                if points:  # Aggiungi solo se ci sono punti
                    contour_points.append(points)
            
            if not contour_points:
                return {'error': 'Nessun punto contorno valido'}
            
            # Crea l'immagine con i contorni
            contours_image = self.draw_contours_safely(image, valid_contours)
            
            # Calcola le statistiche
            total_points = sum(len(contour) for contour in contour_points)
            avg_points = total_points / len(contour_points) if contour_points else 0
            
            stats = {
                'total_contours': len(contour_points),
                'total_points': total_points,
                'avg_points_per_contour': round(avg_points, 1)
            }
            
            # Converti l'immagine in base64 con validazione
            try:
                _, buffer = cv2.imencode('.jpg', contours_image)
                image_bytes = buffer.tobytes()
                contours_image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                
                # Assicurati che il prefisso sia corretto
                if not contours_image_b64.startswith('data:image/'):
                    contours_image_b64 = f"data:image/jpeg;base64,{contours_image_b64}"
                
                # Rimuovi eventuali duplicati del prefisso
                if contours_image_b64.count('data:image/') > 1:
                    parts = contours_image_b64.split('data:image/')
                    contours_image_b64 = f"data:image/jpeg;base64,{parts[-1]}"
                    
            except Exception as e:
                print(f"Errore codifica immagine: {e}")
                # Immagine vuota come fallback
                contours_image_b64 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A8A"
            
            return {
                'success': True,
                'contour_points': contour_points,
                'contours_image': contours_image_b64,
                'image_size': {
                    'width': image.shape[1],
                    'height': image.shape[0]
                },
                'stats': stats
            }
            
        except Exception as e:
            print(f"❌ Errore nel processamento: {e}")
            return {'error': f'Errore nel processamento: {str(e)}'}
    
    def draw_contours_safely(self, image, contours):
        """Disegna i contorni in modo sicuro"""
        try:
            # Crea una copia dell'immagine
            result = image.copy()
            
            # Disegna i contorni solo se validi
            if contours and len(contours) > 0:
                # Filtra solo contorni con punti validi
                valid_contours = []
                for contour in contours:
                    if contour is not None and len(contour) > 0:
                        # Verifica che i punti siano validi
                        valid_points = []
                        for point in contour:
                            if len(point) > 0 and len(point[0]) >= 2:
                                x, y = int(point[0][0]), int(point[0][1])
                                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                                    valid_points.append([x, y])
                        
                        if valid_points:
                            # Converti in formato numpy corretto
                            valid_points = np.array(valid_points, dtype=np.int32).reshape(-1, 1, 2)
                            valid_contours.append(valid_points)
                
                if valid_contours:
                    cv2.drawContours(result, valid_contours, -1, (0, 255, 0), 2)
            
            return result
            
        except Exception as e:
            print(f"❌ Errore nel disegno contorni: {e}")
            # Restituisci l'immagine originale senza contorni
            return image.copy()
    
    def _filter_contours_ultra_aggressive(self, contours_list):
        """FILTRAGGIO ULTRA-AGGRESSIVO - MANTIENE QUASI TUTTO"""
        if not contours_list:
            return []
        
        print(f"🔍 Filtraggio ULTRA-AGGRESSIVO di {len(contours_list)} contorni...")
        
        # Calcola metriche MINIME
        contour_data = []
        for i, contour in enumerate(contours_list):
            if len(contour) < 3:
                continue
                
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
            
            # FILTRI QUASINESSUNI - solo i veramente impossibili
            if area < 0.1 or perimeter < 0.5:
                continue
            
            # Calcola compactness ma con soglia ENORME
            compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else float('inf')
            
            # Rimuovi solo contorni MATHEMATICAMENTE impossibili
            if compactness > 10000:
                continue
            
            # Bounding box ratio - soglia ENORME
            x, y, w, h = cv2.boundingRect(contour)
            bbox_ratio = w / h if h > 0 else 0
            
            # Rimuovi solo ratio VERAMENTE impossibili
            if bbox_ratio > 1000 or bbox_ratio < 0.001:
                continue
            
            # Aggiungi TUTTO
            contour_data.append({
                'contour': contour,
                'area': area,
                'perimeter': perimeter,
                'compactness': compactness,
                'bbox_ratio': bbox_ratio,
                'index': i
            })
        
        print(f"📊 Contorni dopo filtri minimi: {len(contour_data)}")
        
        # Rimuovi solo DUPLICATI OVVI
        unique_contours = self._remove_only_obvious_duplicates(contour_data)
        
        # Ordina per area (dal più grande)
        unique_contours.sort(key=lambda x: x['area'], reverse=True)
        
        # Prendi TUTTI (limite a 500 per non crashare)
        max_contours = min(500, len(unique_contours))
        best_contours = [item['contour'] for item in unique_contours[:max_contours]]
        
        print(f"✅ Filtro ULTRA-AGGRESSIVO: {len(contours_list)} -> {len(best_contours)} contorni")
        return best_contours
    
    def _remove_only_obvious_duplicates(self, contour_data):
        """Rimuove solo duplicati VERAMENTE ovvi (IoU > 0.9)"""
        if len(contour_data) <= 1:
            return contour_data
        
        unique = []
        threshold = 0.9  # Solo duplicati ovvi
        
        for current in contour_data:
            is_duplicate = False
            
            for existing in unique:
                similarity = self._calculate_contour_similarity(current, existing)
                if similarity > threshold:
                    # Mantieni solo quello con area più grande
                    if current['area'] > existing['area']:
                        unique.remove(existing)
                        unique.append(current)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(current)
        
        return unique
    
    def _calculate_contour_similarity(self, c1, c2):
        """Calcola la similarità tra due contorni basata su area e perimetro"""
        area_diff = abs(c1['area'] - c2['area']) / max(c1['area'], c2['area'], 1)
        perimeter_diff = abs(c1['perimeter'] - c2['perimeter']) / max(c1['perimeter'], c2['perimeter'], 1)
        
        # Similarità basata su area e perimetro
        similarity = 1 - (area_diff + perimeter_diff) / 2
        return similarity

# Inizializza il processore di immagini
processor = ImageProcessor()

@app.route('/detect_contours', methods=['POST'])
def detect_contours_route():
    """Endpoint per processare l'immagine (alias per compatibilità)"""
    return process_image_route()

@app.route('/process_image', methods=['POST'])
def process_image_route():
    """Endpoint per processare l'immagine"""
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Nessuna immagine fornita'}), 400
        
        result = processor.process_image(image_data)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Errore nell'endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_contours', methods=['POST'])
def save_contours():
    """Salva i contorni processati"""
    try:
        data = request.get_json()
        contour_points = data.get('contour_points', [])
        image_size = data.get('image_size', {})
        
        # Genera un filename unico
        timestamp = int(time.time())
        filename = f"contours_{timestamp}.json"
        filepath = os.path.join(CONTOURS_FOLDER, filename)
        
        # Salva i dati
        save_data = {
            'contour_points': contour_points,
            'image_size': image_size,
            'timestamp': timestamp
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'timestamp': timestamp
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_contours/<filename>')
def load_contours(filename):
    """Carica i contorni salvati"""
    try:
        # Validazione del filename
        if not filename.endswith('.json') or '/' in filename or '\\' in filename:
            return jsonify({'error': 'Filename non valido'}), 400
        
        filepath = os.path.join(CONTOURS_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File non trovato'}), 404
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return jsonify({
            'success': True,
            'contour_points': data.get('contour_points', []),
            'image_size': data.get('image_size', {}),
            'timestamp': data.get('timestamp', 0)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Configurazione per Render
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Solo in locale, apri il browser
    if host == '127.0.0.1':
        import webbrowser
        import threading
        import time
        
        def open_browser():
            time.sleep(1.5)
            webbrowser.open('http://localhost:5000')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    print(f"🚀 Starting HACKER DRAWING APP on {host}:{port}")
    if host != '127.0.0.1':
        print(f"🌐 Render URL: https://hacker-drawing-app.onrender.com")
    print(f"👨‍💻 Created by: Stefano Luciano")
    
    app.run(host=host, port=port, debug=debug)
