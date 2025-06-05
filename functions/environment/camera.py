import os
import cv2 as cv
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from .architecture import ResEmoteNet

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

def predict_emotion(face):
    """
    Clasifica la emoción en un frame de rostro usando el modelo predictivo.
    """
    x = preprocess(face).unsqueeze(0).to(dev)
    with torch.no_grad():
        y = model(x)
        probs = [round(score, 2) for score in F.softmax(y, dim=1).cpu().numpy().flatten()]
    idx = np.argmax(probs)
    return emotions[idx], probs[idx]

def detect_and_display(frame):
    """
    Procesamiento básico de emociones en cámara.
    """
    frame = cv.flip(frame,1)
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = Image.fromarray(frame[y:y+h, x:x+w])
        emotion, prob = predict_emotion(face)

        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, f'{emotion} - {round(prob * 100, 2)}%', (x - 5, y - 15), font, 0.8, (0, 255, 0), 2)

    cv.imshow(window_name, frame)

    # TO-DO con RL: agregar un return con la emoción para combinarlo con RL.

def open_camera():
    """
    Abre la cámara y, en tiempo real, muestra las emociones detectadas.
    """
    
    cv.namedWindow(window_name)
    capture = cv.VideoCapture(0)

    if not capture.isOpened():
        print("--(!)Error al abrir la cámara")
        return

    while capture.isOpened():
        ret, frame = capture.read()
        if frame is None:
            print("--(!) No se capturó el frame")
            break

        detect_and_display(frame)

        # Presionar c para salir.
        if cv.waitKey(1) & 0xFF == ord('c'):
            break

    capture.release()
    cv.destroyAllWindows()