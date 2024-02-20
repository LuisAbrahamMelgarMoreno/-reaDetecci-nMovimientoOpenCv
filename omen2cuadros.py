import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.rectangle(frame, (0, 0), (frame.shape[1]//2, frame.shape[0]), (0, 0, 255), 2)  # Cuadro 1
    cv2.rectangle(frame, (frame.shape[1]//2, 0), (frame.shape[1], frame.shape[0]), (255, 0, 0), 2)  # Cuadro 2

    color1 = (0, 255, 0)
    color2 = (0, 255, 0)
    texto_estado1 = "Estado: No se ha detectado movimiento"
    texto_estado2 = "Estado: No se ha detectado movimiento"

    # Definir los puntos para el área de análisis de cada cuadro
    width, height = frame.shape[1], frame.shape[0]
    area_pts1 = np.array([[0, 0], [width//2, 0], [width//2, height], [0, height]])
    area_pts2 = np.array([[width//2, 0], [width, 0], [width, height], [width//2, height]])

    imAux1 = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux1 = cv2.drawContours(imAux1, [area_pts1], -1, (255), -1)
    image_area1 = cv2.bitwise_and(gray, gray, mask=imAux1)

    imAux2 = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux2 = cv2.drawContours(imAux2, [area_pts2], -1, (255), -1)
    image_area2 = cv2.bitwise_and(gray, gray, mask=imAux2)

    fgmask1 = fgbg.apply(image_area1)
    fgmask1 = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel)
    fgmask1 = cv2.dilate(fgmask1, None, iterations=2)

    fgmask2 = fgbg.apply(image_area2)
    fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel)
    fgmask2 = cv2.dilate(fgmask2, None, iterations=2)

    cnts1 = cv2.findContours(fgmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts1:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            texto_estado1 = "Estado: Alerta Movimiento Detectado!"
            color1 = (0, 0, 255)

    cnts2 = cv2.findContours(fgmask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts2:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x + width//2, y), (x + w + width//2, y + h), (0, 255, 0), 2)
            texto_estado2 = "Estado: Alerta Movimiento Detectado!"
            color2 = (0, 0, 255)

    cv2.drawContours(frame, [area_pts1], -1, color1, 2)
    cv2.putText(frame, texto_estado1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color1, 2)

    cv2.drawContours(frame, [area_pts2], -1, color2, 2)
    cv2.putText(frame, texto_estado2, (frame.shape[1]//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color2, 2)

    cv2.imshow('fgmask1', fgmask1)
    cv2.imshow('fgmask2', fgmask2)
    cv2.imshow("frame", frame)

    k = cv2.waitKey(70) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
