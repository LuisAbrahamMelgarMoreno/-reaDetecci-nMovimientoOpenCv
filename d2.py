import cv2

# Declarar area_referencia como variable global
area_referencia = None

# Función para detectar movimiento dentro de un área específica
def detectar_movimiento(frame):
    global area_referencia
    
    # Convertir el frame a escala de grises
    frame_gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Aplicar desenfoque gaussiano para eliminar ruido
    frame_gris = cv2.GaussianBlur(frame_gris, (21, 21), 0)

    # Si no hay un marco de referencia, establecerlo como el primer frame
    if area_referencia is None:
        area_referencia = frame_gris
        return frame, False

    # Calcular la diferencia absoluta entre el marco de referencia y el frame actual
    diferencia = cv2.absdiff(area_referencia, frame_gris)
    # Umbralizar la diferencia
    _, umbral = cv2.threshold(diferencia, 25, 255, cv2.THRESH_BINARY)

    # Encontrar contornos en la imagen umbralizada
    contornos, _ = cv2.findContours(umbral.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Buscar contornos dentro del área especificada
    movimiento_detectado = False
    for contorno in contornos:
        if cv2.contourArea(contorno) > 1000:  # Ajusta este valor según tu necesidad
            x, y, w, h = cv2.boundingRect(contorno)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            movimiento_detectado = True

    return frame, movimiento_detectado

# Capturar video desde la cámara
captura = cv2.VideoCapture(0)

# Obtener el tamaño del cuadro de detección de movimiento
ret, frame = captura.read()
altura, ancho, _ = frame.shape

while True:
    ret, frame = captura.read()
    if not ret:
        break

    # Definir el área de interés (por ejemplo, la mitad inferior de la pantalla)
    area_interes = frame[int(frame.shape[0]/2):, :]

    # Detectar movimiento dentro del área de interés
    frame_con_movimiento, movimiento_detectado = detectar_movimiento(area_interes)

    # Cambiar el tamaño del cuadro de la cámara para que coincida con el tamaño del cuadro de detección de movimiento
    frame = cv2.resize(frame, (ancho, altura))

    # Mostrar el frame solo si se detecta movimiento
    if movimiento_detectado:
        cv2.putText(frame_con_movimiento, "Movimiento detectado", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Camara', frame_con_movimiento)
    else:
        cv2.imshow('Camara', frame)

    # Romper el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
captura.release()
cv2.destroyAllWindows()
