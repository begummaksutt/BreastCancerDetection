

import os
import cv2
import gradio as gr
from ultralytics import YOLO

# =======================
# MODEL YOLU
# =======================
MODEL_PATH = "/Users/begumaksut/Desktop/meme kanseri projesi/best (7).pt"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} bulunamadı. Dosya yolunu kontrol et!")

# Modeli yükle
model = YOLO(MODEL_PATH, task="detect")

# =======================
# PREDICT FONKSİYONU
# =======================
def predict(image):
    results = model.predict(source=image, conf=0.25, task="detect")
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

    risk_percentage = 0
    message = "🟢 Normal hücre tespit edildi! Görüntü normal görünüyor, risk yok."

    # En yüksek riskli hücreyi bulmak için
    risk_order = {"kötü huylu hücre": 3, "kotu huylu hucre": 3, "iyi huylu hücre": 2, "normal hücre": 1}
    current_risk_level = 0

    if len(results[0].boxes) > 0:
        for cls_index in results[0].boxes.cls:
            class_name = model.names[int(cls_index)].lower().strip().replace("ü", "u")  # Türkçe karakter ve boşluk temizleme
            if class_name in risk_order and risk_order[class_name] > current_risk_level:
                current_risk_level = risk_order[class_name]
                if class_name in ["kötü huylu hücre", "kotu huylu hucre"]:
                    risk_percentage = 90
                    message = "🚨 KIRMIZI ALARM! Kötü huylu hücre tespit edildi! Acilen uzman doktora başvurun. Erken müdahale hayat kurtarır!"
                elif class_name == "iyi huylu hücre":
                    risk_percentage = 30
                    message = "🟡 İyi huylu hücre bulundu! Şu an için rahatlayabilirsiniz, ama kontrollerinizi aksatmayın."
                elif class_name == "normal hücre":
                    risk_percentage = 5
                    message = "🟢 Normal hücre tespit edildi! Görüntü normal görünüyor, risk yok."

    risk_info = f"Risk Yüzdesi: %{risk_percentage}\nMesaj: {message}"
    return annotated_frame, risk_info


# =======================
# GRADIO ARAYÜZÜ
# =======================
app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Resim yükle veya kamera kullan"),
    outputs=[
        gr.Image(type="pil", label="Tahmin Sonucu"),
        gr.Textbox(label="Risk Analizi", lines=4)
    ],
    live=False,
    title="Meme Kanseri Tespiti - YOLOv8",
    description="best (5).pt modeliyle hücre tespiti ve sınıflandırma"
)

if __name__ == "__main__":
    app.launch()
