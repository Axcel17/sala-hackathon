# pipeline/config.py
"""
Configuración central del pipeline.
Todo lo que puede cambiar vive aquí — nada hardcodeado en el resto del código.
"""

import os

# ─────────────────────────────────────────────────────
# RUTAS DE MODELOS
# ─────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

MODEL_PATHS = {
    "m1": os.path.join(MODELS_DIR, "m1_validador.keras"),
    "m3": os.path.join(MODELS_DIR, "m3_nutrientes.keras"),
    "m4": os.path.join(MODELS_DIR, "m4_enfermedades.keras"),
}

# M1 — binario, sigmoid
# 0 = banana_leaf, 1 = non_banana_leaf
CLASES_M1 = {
    0: "banana_leaf",
    1: "non_banana_leaf",
}

# M4 — enfermedades (orden alfabético)
CLASES_M4 = {
    0: "Cordana",
    1: "Fusarium",
    2: "Sano",
    3: "SigatokaNegra",
}

# M3 — nutrientes (orden alfabético)
CLASES_M3 = {
    0: "Boron",
    1: "Calcium",
    2: "Iron",
    3: "Magnesium",
    4: "Manganese",
    5: "Potassium",
    6: "Sano",
    7: "Sulphur",
}

# ─────────────────────────────────────────────────────
# TAMAÑO DE IMAGEN
# Mismo para los 3 modelos — todos entrenados con 224x224
# ─────────────────────────────────────────────────────

IMG_SIZE = (224, 224)

# ─────────────────────────────────────────────────────
# UMBRALES DE CONFIANZA
#
# Si la confianza del modelo está por debajo del umbral,
# el sistema reporta "baja confianza" para que el LLM
# lo comunique al agricultor.
#
# M1: umbral más alto — no queremos pasar imágenes dudosas
# M4/M3: umbral más bajo — preferimos dar un diagnóstico
#         con advertencia que no dar ninguno
# ─────────────────────────────────────────────────────

UMBRALES = {
    "m1": 0.70,
    "m4": 0.55,
    "m3": 0.55,
}

# ─────────────────────────────────────────────────────
# FLUJOS POSIBLES
# Constantes para evitar strings sueltos en el código
# ─────────────────────────────────────────────────────

FLUJO_NO_BANANA  = "NO_BANANA"
FLUJO_ENFERMO    = "ENFERMO"
FLUJO_DEFICIT    = "DEFICIT"
FLUJO_SANO       = "SANO"

# ─────────────────────────────────────────────────────
# CLASES QUE ACTIVAN CADA FLUJO
# ─────────────────────────────────────────────────────

CLASES_ENFERMEDAD = {"Cordana", "Fusarium", "SigatokaNegra"}
CLASES_DEFICIT    = {"Boron", "Calcium", "Iron", "Magnesium", "Manganese", "Potassium", "Sulphur"}
CLASE_SANO        = "Sano"
