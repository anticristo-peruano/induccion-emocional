import os
import cv2 as cv
import torch
import numpy as np
import pygame
import threading
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from .architecture import ResEmoteNet
import warnings
warnings.filterwarnings('ignore')

"""
FUNCIONES DE CÁMARA
"""

# Variables generales para opencv.
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
window_name = "Inducción emocional"
font = cv.FONT_HERSHEY_SIMPLEX

# Carga del modelo predictivo (tener best.pth en functions/environment)
model_path = os.path.join(os.curdir,'functions','environment','best.pth')
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEmoteNet().to(dev)
checkpoint = torch.load(model_path, map_location=dev, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Pipeline de preprocesamiento.
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),  # Redimensiona la imagen a 64x64 píxeles
    transforms.Grayscale(num_output_channels=3),  # Convierte la imagen a escala de grises
    transforms.ToTensor(),  # Convierte la imagen en un tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliza los valores de los píxeles
])
emotions = ['happy', 'surprise', 'sad', 'anger', 'disgust', 'fear', 'neutral']
val_aro = np.array([
    [0.81, 0.51],   # Happy
    [0.40, 0.67],   # Surprise
    [-0.63, -0.27],   # Sad
    [-0.51, 0.59],  # Angry
    [-0.60, 0.35],  # Disgust
    [-0.63, 0.71],   # Fear
    [0.20, -0.20]   # Neutral
])


def predict_vector(face):
    try:
        """
        Clasifica la emoción en un frame de rostro usando el modelo predictivo.
        """
        x = preprocess(face).unsqueeze(0).to(dev)
        with torch.no_grad():
            y = model(x)
            probs = [round(score, 2) for score in F.softmax(y, dim=1).cpu().numpy().flatten()]
        return probs @ val_aro, True
    except:
        return None, False

def open_camera(duration_ms,steps):
    """
    Abre la cámara y, en tiempo real, muestra las emociones detectadas.
    """
    
    history = {
        'step':[],
        'timestamp':[],
        'vector':[]
    }

    cv.namedWindow(window_name)
    capture = cv.VideoCapture(0)

    interval = (duration_ms / 1000.0) // (steps + 1)
    t_init = time.time()
    prox_cap = t_init + interval
    caps = 0

    if not capture.isOpened():
        print("--(!)Error al abrir la cámara")
        return np.array([0,0]), history

    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv.flip(frame,1)
        now = time.time()

        if frame is None:
            print("--(!) No se capturó el frame")
            break

        # --- Visualización ---
        faces = face_cascade.detectMultiScale(cv.cvtColor(frame,cv.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- Procesamientos ---
            if caps <= steps and now >= prox_cap:
                face = Image.fromarray(frame[y:y+h, x:x+w])
                vector, flag = predict_vector(face)
                
                if flag:
                    history['step'] += [caps + 1]
                    history['timestamp'] += [round(now - t_init,2)]
                    history['vector'] += [vector]

                    print(f'[Step {caps + 1} - {round(now - t_init,2)} s] Valence: {round(vector[0],2)}, Arousal: {round(vector[1],2)}')
                    caps += 1
                    prox_cap += interval

        cv.imshow(window_name,frame)

        # --- Salidas ---
        # Presionar 'c' para salir
        if cv.waitKey(1) & 0xFF == ord('c'):
            break

        # Cerrar si ya se completaron los pasos
        if caps > steps:
            break

    capture.release()
    cv.destroyAllWindows()

    return np.mean(history['vector'],axis=0), history

"""
Función de hilo
"""

def thread_episode(songid, duration_ms, steps):
    ruta_mp3 = os.path.join(os.curdir,'functions','environment','.songs',f'{songid}.mp3')
    pygame.mixer.init()
    pygame.mixer.music.load(ruta_mp3)
    pygame.mixer.music.play()

    vector, history = open_camera(duration_ms,steps)
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

    return vector, history