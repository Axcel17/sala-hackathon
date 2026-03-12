# pipeline/model_loader.py
"""
Carga los 3 modelos una sola vez cuando el sistema arranca.

CRÍTICO: los modelos se cargan en este módulo y se importan
desde aquí en el resto del código. NUNCA se cargan dentro
de una función de predicción — eso sería recargarlos en
cada request, lo cual es lento e innecesario.
"""

import os
import tensorflow as tf
import keras           # Keras 3 standalone — para M3 y M4 (guardados con TF 2.19 / Keras 3)
import tf_keras        # Legacy TF Keras    — para M1  (guardado con TF 2.15 / Keras 2)
from pipeline.config import MODEL_PATHS

# M1 fue guardado con TF 2.15 (Keras 2.x) → se carga con tf_keras
# M3 y M4 fueron guardados con TF 2.19 (Keras 3) → se cargan con keras standalone
_KERAS2_MODELS = {"m1"}


def _verificar_modelos():
    """Verifica que los archivos existen antes de intentar cargarlos."""
    faltantes = []
    for nombre, ruta in MODEL_PATHS.items():
        if not os.path.exists(ruta):
            faltantes.append(f"  {nombre}: {ruta}")

    if faltantes:
        raise FileNotFoundError(
            "Los siguientes modelos no se encontraron:\n"
            + "\n".join(faltantes)
            + "\n\nVerifica que MODEL_PATHS en config.py apunta a las rutas correctas."
        )


def cargar_modelos() -> dict:
    """
    Carga M1, M3 y M4 en memoria y los devuelve en un dict.

    Returns:
        {
            "m1": tf_keras.Model,  MobileNetV3Small  (Keras 2.x format)
            "m3": keras.Model,     EfficientNetB3 nutrientes (Keras 3 format)
            "m4": keras.Model,     EfficientNetB3 enfermedades (Keras 3 format)
        }
    """
    _verificar_modelos()

    print("Cargando modelos...")

    modelos = {}
    for nombre, ruta in MODEL_PATHS.items():
        print(f"  [{nombre}] {os.path.basename(ruta)}...", end=" ", flush=True)
        if nombre in _KERAS2_MODELS:
            modelos[nombre] = tf_keras.models.load_model(ruta)   # Keras 2.x
        else:
            modelos[nombre] = keras.models.load_model(ruta)       # Keras 3
        print("OK")

    print("Modelos listos.\n")
    return modelos


# ─────────────────────────────────────────────────────
# Instancia global — se carga una vez al importar el módulo
# En app.py se hace: from pipeline.model_loader import MODELOS
# ─────────────────────────────────────────────────────

MODELOS = cargar_modelos()
