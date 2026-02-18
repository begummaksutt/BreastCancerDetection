import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# =======================
# MODEL YOLU
# =======================
MODEL_PATH = "/Users/begumaksut/Desktop/meme kanseri projesi/best (7).pt"
model = YOLO(MODEL_PATH, task="detect")

# =======================
# Sınıf isimleri ve risk mesajları
# =======================
class_names = {
    "malignant": "Kötü Huylu Hücre",
    "benign": "İyi Huylu Hücre",
    "normal": "Normal Hücre"
}

risk_messages = {
    "Kötü Huylu Hücre": ("90%", "⚠️ Yüksek risk: Kötü huylu hücre tespit edildi. Lütfen uzman bir doktora başvurun.", "#FF4C4C"),
    "İyi Huylu Hücre": ("40%", "ℹ️ İyi huylu hücre tespit edildi. Düzenli kontroller önerilir.", "#FFB300"),
    "Normal Hücre": ("5%", "✅ Normal hücre tespit edildi. Endişelenecek bir durum yoktur.", "#4CAF50")
}

# Etiket renkleri (çerçeve ve yazı arka planı)
label_colors = {
    "Kötü Huylu Hücre": "#FF4C4C",
    "İyi Huylu Hücre": "#FFB300",
    "Normal Hücre": "#4CAF50"
}

# =======================
# PREDICT FONKSİYONU
# =======================
def predict(image):
    results = model(image)

    if hasattr(image, "convert"):
        img = image.convert("RGB")
    else:
        img = Image.fromarray(image)

    draw = ImageDraw.Draw(img)

    # UTF-8 destekli yazı tipi
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 20)
    except:
        font = ImageFont.load_default()

    display_label = "Belirsiz"
    risk_percentage = "0%"
    risk_message = "Analiz yapılmadı."
    bg_color = "#FFFFFF"

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            label = r.names[cls_id]

            display_label = class_names.get(label, label)

            # Kutu koordinatları
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Çerçeve ve etiket rengi
            label_color = label_colors.get(display_label, "#FF4C4C")
            draw.rectangle([x1, y1, x2, y2], outline=label_color, width=3)

            # Etiket boyutu
            bbox = draw.textbbox((0,0), display_label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

            # Etiket arka planı ve yazı
            draw.rectangle([x1, y1 - text_h - 5, x1 + text_w, y1], fill=label_color)
            draw.text((x1 + 2, y1 - text_h - 5), display_label, fill="white", font=font)

            # Risk bilgisi ve arka plan rengi
            risk_percentage, risk_message, bg_color = risk_messages.get(display_label, ("0%", "Analiz yapılmadı.", "#FFFFFF"))

    # HTML ile renkli kutu oluştur
    result_html = f'<div style="background-color:{bg_color}; padding:10px; border-radius:5px; color:white;">Tahmin Sonucu: {display_label}</div>'
    risk_html = f'<div style="background-color:{bg_color}; padding:10px; border-radius:5px; color:white;">Risk Yüzdesi: {risk_percentage}<br>{risk_message}</div>'

    return img, result_html, risk_html

# =======================
# GRADIO ARAYÜZÜ
# =======================
with gr.Blocks() as demo:
    gr.Markdown("## 🧬 Meme Kanseri Hücre Analizi")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Görüntü Yükle")
            predict_button = gr.Button("Analiz Et")
        with gr.Column():
            image_output = gr.Image(type="pil", label="Analiz Sonucu")
            result_output = gr.HTML(label="Tahmin Sonucu")
            risk_output = gr.HTML(label="Risk Analizi")

    predict_button.click(fn=predict, inputs=image_input, outputs=[image_output, result_output, risk_output])

demo.launch()
