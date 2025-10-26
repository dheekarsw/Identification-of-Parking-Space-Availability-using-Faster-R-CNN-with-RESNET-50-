import cv2
import numpy as np

def preprocess_brightness_contrast(image, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.5):
    """
    Menerapkan CLAHE dan Gamma Correction untuk mengatasi overexposure akibat cahaya terang.
    
    Parameters:
        image: np.ndarray
            Gambar dalam format BGR (dari cv2.imread)
        clip_limit: float
            Nilai limit untuk kontras pada CLAHE
        tile_grid_size: tuple
            Ukuran grid untuk lokal histogram equalization
        gamma: float
            Faktor gamma (gamma < 1 untuk mencerahkan, gamma > 1 untuk menggelapkan)
    
    Returns:
        preprocessed: np.ndarray
            Gambar yang telah diproses
    """
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
    inv_gamma = 3.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255
                     for i in np.arange(256)]).astype("uint8")
    image_gamma = cv2.LUT(image_clahe, table)

    return image_gamma
# Load gambar dari file
# image = cv2.imread('2012-09-14_13_16_21_jpg.rf.d3d49d8f1e18ab3b5fa30b6397e94f4b.jpg')
image = cv2.imread('2013-04-16_09_30_03_jpg.rf.fc26bbe17b3904c723f504c8d3b2e171.jpg')
# image = cv2.imread('2012-09-21_11_35_24_jpg.rf.d5f8c080ed001fa09c4d57056e091e71.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Terapkan preprocessing
processed_image = preprocess_brightness_contrast(image)

# Tampilkan hasil
cv2.imshow("Original", image)
cv2.imshow("Processed", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
