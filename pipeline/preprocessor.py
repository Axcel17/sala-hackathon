# pipeline/preprocessor.py
"""
Preprocessing de imágenes para cada modelo.

REGLA CRÍTICA: cada modelo fue entrenado con un preprocess_input
específico. Usar el preprocessing incorrecto produce predicciones
incorrectas aunque el modelo esté perfectamente entrenado.

  M1  (MobileNetV3Small)  → mobilenet_v3.preprocess_input   → escala a [-1, 1]
  M3  (EfficientNetB3)    → efficientnet.preprocess_input    → normalización por canal
  M4  (EfficientNetB3)    → efficientnet.preprocess_input    → normalización por canal
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

from pipeline.config import IMG_SIZE


def _cargar_imagen(imagen_input) -> np.ndarray:
    """
    Acepta ruta de archivo (str) o bytes de imagen.
    Devuelve array RGB de shape (224, 224, 3).
    """
    if isinstance(imagen_input, (str, bytes)) and isinstance(imagen_input, str):
        # Es una ruta de archivo
        img = tf.keras.utils.load_img(imagen_input, target_size=IMG_SIZE)
        return tf.keras.utils.img_to_array(img)

    elif isinstance(imagen_input, bytes):
        # Son bytes crudos (ej: desde WhatsApp o upload web)
        img_tensor = tf.image.decode_image(imagen_input, channels=3, expand_animations=False)
        img_tensor = tf.image.resize(img_tensor, IMG_SIZE)
        return img_tensor.numpy()

    elif isinstance(imagen_input, np.ndarray):
        # Ya es un array — solo redimensionar si es necesario
        if imagen_input.shape[:2] != IMG_SIZE:
            img_tensor = tf.image.resize(imagen_input, IMG_SIZE)
            return img_tensor.numpy()
        return imagen_input.astype(np.float32)

    else:
        raise ValueError(
            f"Tipo de imagen no soportado: {type(imagen_input)}. "
            "Se acepta: str (ruta), bytes, o np.ndarray."
        )


def preprocess_para_m1(imagen_input) -> np.ndarray:
    """
    Preprocessing para M1 (MobileNetV3Small).
    Devuelve array de shape (1, 224, 224, 3) listo para predict().
    """
    arr = _cargar_imagen(imagen_input)
    arr = mobilenet_preprocess(arr)          # escala a [-1, 1]
    return np.expand_dims(arr, axis=0)       # (1, 224, 224, 3)


def preprocess_para_efficientnet(imagen_input) -> np.ndarray:
    """
    Preprocessing para M3 y M4 (EfficientNetB3).
    Devuelve array de shape (1, 224, 224, 3) listo para predict().
    """
    arr = _cargar_imagen(imagen_input)
    arr = efficientnet_preprocess(arr)       # normalización por canal ImageNet
    return np.expand_dims(arr, axis=0)       # (1, 224, 224, 3)
