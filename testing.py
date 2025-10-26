import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

# Konfigurasi
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 3  # background, occupied, vacant

# Load backbone ResNet50
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

backbone = resnet50(pretrained=True)
backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
backbone.out_channels = 2048

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
    aspect_ratios=((0.25, 0.5, 1.0, 2.0, 3.0),))
# anchor_generator = AnchorGenerator(sizes=((64, 128, 256),),
#     aspect_ratios=((0.25, 0.5, 1.0, 2.0, 3.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                   num_classes=num_classes,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)

# Load trained weights
model.load_state_dict(torch.load("fasterrcnn_epoch25.pth", map_location=device))
model.to(device)
model.eval()
def preprocess_brightness_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.5):
    # Konversi ke YCrCb untuk manipulasi brightness (luminance)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # CLAHE untuk channel luminance (Y)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    y_clahe = clahe.apply(y)

    # Gabungkan kembali dan konversi ke BGR
    ycrcb_clahe = cv2.merge((y_clahe, cr, cb))
    image_clahe = cv2.cvtColor(ycrcb_clahe, cv2.COLOR_YCrCb2BGR)

    # Gamma correction untuk menggelapkan highlight berlebih
    inv_gamma = 2.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                     for i in np.arange(256)]).astype("uint8")
    image_gamma = cv2.LUT(image_clahe, table)
    return image_gamma

# Fungsi untuk testing gambar
def test_image(image_path, threshold=0.5):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    txtWidth = int(width/32)
    txtHeight = int(height/24)
    if image is None:
        raise ValueError(f"Gambar tidak ditemukan: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_brightness_contrast(image)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    

    with torch.no_grad():
        outputs = model(image_tensor)

    pred_boxes = outputs[0]['boxes'].cpu().numpy()
    pred_scores = outputs[0]['scores'].cpu().numpy()
    pred_labels = outputs[0]['labels'].cpu().numpy()

    print(f"Jumlah prediksi: {len(pred_boxes)}")
    print("Skor prediksi:", pred_scores)

    # Gambar hasil
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Filter label sesuai threshold
    filtered_labels = [label for score, label in zip(pred_scores, pred_labels) if score >= threshold]

    # Hitung jumlah occupied dan vacant
    num_occupied = sum(1 for l in filtered_labels if l == 1)
    num_vacant = sum(1 for l in filtered_labels if l == 2)
    
    status_text = f"occupied = {num_occupied}"
    status_text2 = f"vacant = {num_vacant}"
    cv2.putText(image_bgr, status_text, (txtWidth, txtHeight),  # posisi teks
                cv2.FONT_HERSHEY_SIMPLEX, int(txtWidth/30), (0, 0, 255), int(txtWidth/15))  # warna putih
    cv2.putText(image_bgr, status_text2, (txtWidth, txtHeight*2),  # posisi teks
                cv2.FONT_HERSHEY_SIMPLEX, int(txtWidth/30), (0, 255, 0), int(txtWidth/15))  # warna putih

    # Gambar kotak dan label seperti biasa
    for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.astype(int)
            color = (0, 255, 0) if label == 2 else (0, 0, 255)
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, int(txtWidth/15))
            label_text = f"{'vacant' if label == 2 else 'occupied'}: {score:.2f}"
            cv2.putText(image_bgr, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, int(txtWidth/30))



    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction for {os.path.basename(image_path)}")
    plt.axis('off')
    plt.show()

# Jalankan jika sebagai skrip utama
if __name__ == "__main__":
    test_image("WhatsApp Image 2025-05-04 at 14.32.41.jpeg", threshold=0.65)
    test_image("VideoCapture_20250525-084655.jpg", threshold=0.65)

