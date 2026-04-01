import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # --- 1. إعداد المسارات وتحميل الصورة ---
    # نستخدم نفس الصورة المرجعية للمستخدم
    image_path = "user_sign_image_1774967112825.png"
    if not os.path.exists(image_path):
        import glob
        alt = glob.glob("user_sign_image*.png")
        if alt: image_path = alt[0]
        else:
            print("Error: Reference image not found.")
            return

    image = cv2.imread(image_path)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --- 2. استخراج النقاط الهيكلية (Pose Landmarks) ---
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            print("Error: No pose landmarks detected.")
            return

        # فهرس النقاط المهمة للتوضيح (الجزء العلوي من الجسم)
        # 0: الأنف، 11: الكتف الأيسر، 12: الكتف الأيمن، 13: المرفق الأيسر، 14: المرفق الأيمن، 15: الرسغ الأيسر، 16: الرسغ الأيمن
        indices = [0, 11, 12, 13, 14, 15, 16]
        raw_landmarks = []
        for idx in indices:
            lm = results.pose_landmarks.landmark[idx]
            # تحويل الإحداثيات النسبية (0-1) إلى إحداثيات بكسل خام (Raw)
            raw_landmarks.append([lm.x * w, lm.y * h])
        
        raw_landmarks = np.array(raw_landmarks)

    # --- 3. عملية التطبيع المكاني (Spatial Normalization) ---
    # أ. استخراج المراجع (الأكتاف)
    # ملاحظة: الفهرس 1 و 2 في قائمتنا المصغرة هما الكتف الأيسر (11) والأيمن (12)
    left_sh = raw_landmarks[1]
    right_sh = raw_landmarks[2]

    # ب. المركزية (Centering): جعل منتصف الأكتاف هو نقطة الأصل (0,0)
    # المعادلة: x_centered = x_raw - midpoint(shoulders)
    mid_shoulder = (left_sh + right_sh) / 2
    centered_data = raw_landmarks - mid_shoulder

    # ج. تغيير المقياس (Scaling): القسمة على المسافة بين الأكتاف لتوحيد الحجم
    # المعادلة: x_normalized = x_centered / distance(shoulders)
    shoulder_dist = np.linalg.norm(left_sh - right_sh)
    if shoulder_dist == 0: shoulder_dist = 1.0 # تجنب القسمة على صفر
    normalized_data = centered_data / shoulder_dist

    # --- 4. الرسم البياني للمقارنة باستخدام Matplotlib ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    plt.subplots_adjust(wspace=0.3)

    # المخطط الأول: البيانات الخام (Ham Veri)
    ax1.scatter(raw_landmarks[:, 0], raw_landmarks[:, 1], color='red', s=50, label='Raw Points')
    # عكس المحور Y لأن إحداثيات الصورة تبدأ من الأعلى (0) إلى الأسفل
    ax1.set_ylim(h, 0)
    ax1.set_xlim(0, w)
    ax1.set_title('Ham Veri (Raw Pixel Coordinates)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('X (Pixels)', fontsize=12)
    ax1.set_ylabel('Y (Pixels)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # إضافة مسميات للنقاط للتوضيح
    labels = ['Nose', 'L_SH', 'R_SH', 'L_ELB', 'R_ELB', 'L_WRI', 'R_WRI']
    for i, txt in enumerate(labels):
        ax1.annotate(txt, (raw_landmarks[i, 0], raw_landmarks[i, 1]), xytext=(5, 5), textcoords='offset points')

    # المخطط الثاني: البيانات المطبعة (Normalize Edilmiş Veri)
    ax2.scatter(normalized_data[:, 0], normalized_data[:, 1], color='blue', s=60, label='Normalized Points', zorder=5)
    ax2.axhline(0, color='black', linewidth=1, alpha=0.7) # خط الأصل الأفقي
    ax2.axvline(0, color='black', linewidth=1, alpha=0.7) # خط الأصل العمودي
    
    # عكس المحور Y للمخطط المطبع ليتوافق مع هيكلية الصورة
    ax2.set_ylim(1.5, -1.5) 
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_title('Normalize Edilmis Veri (Centered on Shoulders)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Normalized X (Unit-less)', fontsize=12)
    ax2.set_ylabel('Normalized Y (Unit-less)', fontsize=12)
    ax2.grid(True, linestyle=':', alpha=0.8)

    for i, txt in enumerate(labels):
        ax2.annotate(txt, (normalized_data[i, 0], normalized_data[i, 1]), xytext=(5, 5), textcoords='offset points')

    # شرح المعادلات في الرسم البياني
    explanation = (
        "Normalization Formulas:\n"
        "1. Centering: P' = P - MidPoint(Shoulders)\n"
        "2. Scaling: P'' = P' / Distance(L_Shoulder, R_Shoulder)\n\n"
        "Goal: Achieve invariance to user position and distance from camera."
    )
    plt.figtext(0.5, 0.01, explanation, wrap=True, horizontalalignment='center', 
                fontsize=11, bbox=dict(facecolor='white', alpha=0.5))

    # حفظ النتيجة
    output_path = "sekil_4_normalization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Success: Visualization saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
