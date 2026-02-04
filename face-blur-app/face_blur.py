import cv2

# Load the face detection model
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40)
    )

    # Loop through all detected faces
    for (x, y, w, h) in faces:

        face_roi = frame[y:y+h, x:x+w]

        # Apply blur
        blurred_face = cv2.GaussianBlur(
            face_roi,
            (99, 99),
            30
        )

        # Replace original face with blurred one
        frame[y:y+h, x:x+w] = blurred_face

        # (Optional) draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Blur App - Press Q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
