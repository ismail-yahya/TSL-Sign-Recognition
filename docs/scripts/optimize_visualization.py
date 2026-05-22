import cv2
import mediapipe as mp
import numpy as np
import os

# --- Constants: ASL Project Optimized Landmark Indices (Exactly matching data_pipeline.py) ---
# Lips (40 points): Representing mouth articulation
LIPS_IDXS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
# Upper Body Reference (3 points: Nose, L_Shoulder, R_Shoulder)
POSE_IDXS = [11, 12, 0]

def draw_styled_landmarks(image, results, draw_all=True):
    """
    Renders landmarks with a premium design.
    draw_all=True: Full Holistic (543).
    draw_all=False: Optimized Subset (85).
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    h, w, _ = image.shape
    
    if draw_all:
        # LEFT: Full MediaPipe Holistic
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(220, 220, 220), thickness=1, circle_radius=1)
            )
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    else:
        # RIGHT: Optimized 85 Points
        COLOR_HANDS = (255, 144, 30) # Vibrant Blue/Orange
        COLOR_LIPS = (0, 0, 255)      # Red
        COLOR_POSE = (0, 255, 0)      # Green

        # --- Hands (Skeleton + Points) ---
        for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=COLOR_HANDS, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(150, 150, 150), thickness=2, circle_radius=1)
                )

        # --- Lips (Points only) ---
        if results.face_landmarks:
            for idx in LIPS_IDXS:
                lm = results.face_landmarks.landmark[idx]
                cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 3, COLOR_LIPS, -1)

        # --- Pose (Connections + Points for the 3 points) ---
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            # Indices: 11=L_SH, 12=R_SH, 0=NOSE
            pts = {idx: (int(landmarks[idx].x * w), int(landmarks[idx].y * h)) for idx in POSE_IDXS}
            # Draw triangle/structure between them
            cv2.line(image, pts[11], pts[12], COLOR_POSE, 2)
            cv2.line(image, pts[11], pts[0], COLOR_POSE, 2)
            cv2.line(image, pts[12], pts[0], COLOR_POSE, 2)
            for pt in pts.values():
                cv2.circle(image, pt, 7, COLOR_POSE, -1)

def add_header(img, title):
    """Adds a clean white header. Title characters cleaned for OpenCV putText."""
    header_h = 80
    header = np.full((header_h, img.shape[1], 3), (255, 255, 255), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.0
    # OpenCV putText doesn't support Turkish chars natively, using ASCII equivalents for clarity
    title_clean = title.replace('ü', 'u').replace('ü', 'u').replace('ş', 's').replace('ı', 'i')
    text_size = cv2.getTextSize(title_clean, font, font_scale, 2)[0]
    text_x = (header.shape[1] - text_size[0]) // 2
    text_y = (header.shape[0] + text_size[1]) // 2
    cv2.putText(header, title_clean, (text_x, text_y), font, font_scale, (30, 30, 30), 2)
    return np.vstack((header, img))

def main():
    # --- Exact Image Path from USER ---
    image_path = "docs/scripts/user_sign_image.jpeg" 
    
    if not os.path.exists(image_path):
        # Fallback to any user_sign_image in root
        import glob
        alt = glob.glob("user_sign_image*.jpeg")
        if alt: image_path = alt[0]
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load {image_path}")
        return

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(static_image_mode=True, model_complexity=2) as holistic:
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # 1. Left View: 543 Points (Original background)
        left_side = image.copy()
        draw_styled_landmarks(left_side, results, draw_all=True)
        left_side = add_header(left_side, "Tum Ham Noktalar (543)")

        # 2. Right View: 85 Points (White background)
        right_side = np.full(image.shape, (255, 255, 255), dtype=np.uint8)
        draw_styled_landmarks(right_side, results, draw_all=False)
        right_side = add_header(right_side, "Optimize Edilmis Noktalar (85)")

        # Combine & Save
        final_viz = np.hstack((left_side, right_side))
        output_name = "sekil_3_optimization.png"
        cv2.imwrite(output_name, final_viz)
        print(f"Success! side-by-side comparison saved to: {os.path.abspath(output_name)}")

if __name__ == "__main__":
    main()

