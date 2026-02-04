import cv2
import numpy as np
import time
import datetime

# -----------------------------
# Global variables
# -----------------------------
background = None
selected_hsv = None
frame = None
prev_time = 0

WINDOW_NAME = "Magic Cloak System (Q: Quit | B: Capture BG | Click: Pick Color)"


# -----------------------------
# Mouse callback for color pick
# -----------------------------
def pick_color(event, x, y, flags, param):
    global selected_hsv, frame

    if event == cv2.EVENT_LBUTTONDOWN and frame is not None:
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        selected_hsv = hsv_img[y, x]
        print("Selected HSV:", selected_hsv)


# -----------------------------
# Main
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not available")
    exit()

cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, pick_color)

print("Press B to capture background")
print("Click on the cloth to select its color")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # -----------------------------
    # Capture background
    # -----------------------------
    if background is None:
        cv2.putText(display, "Press B to capture background",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('b'):
            background = frame.copy()
            print("Background captured")

        if key == ord('q'):
            break

        continue

    # -----------------------------
    # If color not selected yet
    # -----------------------------
    if selected_hsv is None:
        cv2.putText(display, "Click on cloak to pick color",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            background = None
        continue

    # -----------------------------
    # Convert to HSV
    # -----------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h, s, v = selected_hsv

    # Adaptive color range
    lower = np.array([max(int(h) - 10, 0), 80, 80])
    upper = np.array([min(int(h) + 10, 180), 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    # -----------------------------
    # Mask cleaning
    # -----------------------------
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    # Soft edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    mask_inv = cv2.bitwise_not(mask)

    # -----------------------------
    # Background substitution
    # -----------------------------
    bg_part = cv2.bitwise_and(background, background, mask=mask)
    fg_part = cv2.bitwise_and(frame, frame, mask=mask_inv)

    final = cv2.add(bg_part, fg_part)

    # -----------------------------
    # FPS calculation
    # -----------------------------
    current_time = time.time()
    if prev_time == 0:
        fps = 0
    else:
        fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(final, f"FPS: {int(fps)}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    # -----------------------------
    # Cloak detection logging
    # -----------------------------
    cloak_pixels = cv2.countNonZero(mask)

    if cloak_pixels > 3000:
        with open("cloak_log.txt", "a") as f:
            f.write(f"{datetime.datetime.now()} cloak detected\n")

    # -----------------------------
    # UI text
    # -----------------------------
    cv2.putText(final, "B: Re-capture BG | Q: Quit | Click: Pick Color",
                (20, final.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, final)

    # -----------------------------
    # Key handling
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('b'):
        background = None
        print("Re-capturing background...")

cap.release()
cv2.destroyAllWindows()
