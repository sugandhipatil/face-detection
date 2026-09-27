import cv2
from deepface import DeepFace

# Start webcam
camera = cv2.VideoCapture(0)

while True:
    # Read camera frame
    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    try:
        # Detect emotion
        result = DeepFace.analyze(
            frame,
            actions=["emotion"],
            enforce_detection=False
        )

        # Get the strongest detected emotion
        emotion = result[0]["dominant_emotion"]

        # Show emotion on screen
        cv2.putText(
            frame,
            f"Emotion: {emotion}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    except Exception as e:
        print("Detection error:", e)

    # Show camera
    cv2.imshow("Emotion Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Close everything
camera.release()
cv2.destroyAllWindows()