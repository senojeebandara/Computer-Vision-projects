import cv2
import numpy as np

# Start video capture
cap = cv2.VideoCapture(0)

# Define color ranges in HSV
colors = {
    "Blue": {
        "lower": np.array([100, 150, 0]),
        "upper": np.array([140, 255, 255]),
        "color": (255, 0, 0)
    },
    "Green": {
        "lower": np.array([36, 50, 70]),
        "upper": np.array([89, 255, 255]),
        "color": (0, 255, 0)
    },
    "Red": {
        "lower1": np.array([0, 120, 70]),
        "upper1": np.array([10, 255, 255]),
        "lower2": np.array([170, 120, 70]),
        "upper2": np.array([180, 255, 255]),
        "color": (0, 0, 255)
    }
}

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Counters for each color
    counts = {
        "Blue": 0,
        "Green": 0,
        "Red": 0
    }

    for color_name, params in colors.items():
        if color_name != "Red":
            mask = cv2.inRange(hsv, params["lower"], params["upper"])
        else:
            mask1 = cv2.inRange(hsv, params["lower1"], params["upper1"])
            mask2 = cv2.inRange(hsv, params["lower2"], params["upper2"])
            mask = mask1 + mask2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x, y), (x + w, y + h), params["color"], 2)
                cv2.putText(frame, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, params["color"], 2)
                counts[color_name] += 1

    # Display object counts on screen
    y_offset = 30
    for color_name in counts:
        text = f"{color_name} Objects: {counts[color_name]}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[color_name]["color"], 2)
        y_offset += 30

    cv2.imshow("Detected Colors", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
