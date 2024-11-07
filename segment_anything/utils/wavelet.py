import numpy as np
from PIL import Image
import pywt
import argparse
import os
import cv2

def wavelet_transform(image):
    # image = np.array(image)

    LL, (LH, HL, HH) = pywt.dwt2(image, 'Haar')

    LL = (LL - LL.min()) / (LL.max() - LL.min()) * 255
    LL = Image.fromarray(LL.astype(np.uint8))

    LH = (LH - LH.min()) / (LH.max() - LH.min()) * 255
    HL = (HL - HL.min()) / (HL.max() - HL.min()) * 255
    HH = (HH - HH.min()) / (HH.max() - HH.min()) * 255

    merge1 = HH + HL + LH
    h, w = image.shape[-2], image.shape[-1]
    merge1 = cv2.resize(merge1, (h, w))
    merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min()) * 255

    merge1 = Image.fromarray(merge1.astype(np.uint8))
    return merge1