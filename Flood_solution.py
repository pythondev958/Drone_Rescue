from ultralytics import YOLO
import cv2

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Input video
video_path = "flood_video.mp4"
cap = cv2.VideoCapture(video_path)

# Output video settings
output_path = "flood_annotated_output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Simulated drone location (top center)
drone_point = (width // 2, 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Line from drone to person
                cv2.line(frame, drone_point, (center_x, center_y), (0, 0, 255), 2, cv2.LINE_AA)

                # Simulated lifejacket drop (orange)
                drop_y = center_y - 30
                cv2.circle(frame, (center_x, drop_y), 12, (0, 140, 255), -1)
                cv2.putText(frame, "LIFEJACKET DROPPED", (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Optional effect line
                cv2.line(frame, (center_x - 20, drop_y - 10), (center_x + 20, drop_y + 10), (255, 255, 0), 1)

    # Simulated drone icon (larger dark circle with crossbars)
    drone_radius = 20
    drone_color = (30, 30, 30)  # Dark gray
    cv2.circle(frame, drone_point, drone_radius, drone_color, -1)
    # Add drone "arms" to simulate quadcopter
    cv2.line(frame, (drone_point[0] - 25, drone_point[1]), (drone_point[0] + 25, drone_point[1]), (50, 50, 50), 2)
    cv2.line(frame, (drone_point[0], drone_point[1] - 25), (drone_point[0], drone_point[1] + 25), (50, 50, 50), 2)

    # Drone label
    cv2.putText(frame, "Drone", (drone_point[0] + 25, drone_point[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (10, 10, 10), 2)  # Black-ish

    # Show and save
    out.write(frame)
    cv2.imshow("Drone Simulation", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
