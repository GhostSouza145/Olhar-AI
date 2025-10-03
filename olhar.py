import cv2
from ultralytics import YOLO
import numpy as np

# Carregar modelo YOLO pré-treinado (COCO dataset: pessoas, carros, etc.)
model = YOLO("yolov8n.pt")

# Abrir webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar predição
    results = model(frame)

    # Processar resultados
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # caixas [x1, y1, x2, y2]
        confs = r.boxes.conf.cpu().numpy()  # confiança
        clss = r.boxes.cls.cpu().numpy()    # classes

        for box, conf, cls in zip(boxes, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])

            label = model.names[int(cls)]
            text = f"{label} {conf:.2f} ({area_ratio:.2%})"

            # Desenhar caixa
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Se for pessoa, analisar cor predominante
            if label == "person":
                person_crop = frame[y1:y2, x1:x2]
                if person_crop.size > 0:
                    # Converter para HSV para pegar tons de pele
                    hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
                    avg_color = np.mean(hsv.reshape(-1, 3), axis=0)
                    hue, sat, val = avg_color
                    cv2.putText(frame, f"SkinTone(H:{hue:.0f},S:{sat:.0f},V:{val:.0f})",
                                (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 200, 255), 2)

    cv2.imshow("Detecção em tempo real", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
