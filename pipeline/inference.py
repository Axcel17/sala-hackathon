# pipeline/inference.py
"""
Lógica de decisión del pipeline completo.

Expone una sola función pública: run_pipeline(imagen)
El resto del sistema solo necesita llamar a esta función.

Flujo:
    imagen
      ↓
    M1: ¿Es hoja de banano?
      ├── NO  → PipelineResult(flujo=NO_BANANA, llm_needed=False)
      └── SÍ
            ↓
          M4: ¿Tiene enfermedad?
            ├── Cordana / Fusarium / SigatokaNegra
            │     → PipelineResult(flujo=ENFERMO, llm_needed=True)
            └── Sano
                  ↓
                M3: ¿Tiene déficit?
                  ├── Boron / Calcium / ... / Sulphur
                  │     → PipelineResult(flujo=DEFICIT, llm_needed=True)
                  └── Sano
                        → PipelineResult(flujo=SANO, llm_needed=False)
"""

import numpy as np
from typing import Union

from pipeline.config import (
    CLASES_M1, CLASES_M3, CLASES_M4,
    UMBRALES, FLUJO_NO_BANANA, FLUJO_ENFERMO, FLUJO_DEFICIT, FLUJO_SANO,
    CLASES_ENFERMEDAD, CLASES_DEFICIT, CLASE_SANO,
)
from pipeline.model_loader import MODELOS
from pipeline.preprocessor import preprocess_para_m1, preprocess_para_efficientnet
from pipeline.schemas import ModelPrediction, PipelineResult


# ─────────────────────────────────────────────────────
# FUNCIONES INTERNAS DE PREDICCIÓN
# ─────────────────────────────────────────────────────

def _predecir_m1(imagen_input) -> ModelPrediction:
    """
    M1 — Validador de imagen.
    Modelo binario con salida sigmoid.
    Output del modelo: P(non_banana_leaf)
    """
    arr  = preprocess_para_m1(imagen_input)
    prob_non_banana = float(MODELOS["m1"].predict(arr, verbose=0)[0][0])
    prob_banana     = 1.0 - prob_non_banana

    # La clase ganadora es banana si prob_banana > 0.5
    if prob_banana > 0.5:
        clase     = "banana_leaf"
        confianza = prob_banana
    else:
        clase     = "non_banana_leaf"
        confianza = prob_non_banana

    return ModelPrediction(
        clase     = clase,
        confianza = confianza,
        baja_confianza = confianza < UMBRALES["m1"],
        probabilidades = {
            "banana_leaf":     round(prob_banana, 4),
            "non_banana_leaf": round(prob_non_banana, 4),
        },
    )


def _predecir_m4(imagen_input) -> ModelPrediction:
    """
    M4 — Clasificador de enfermedades.
    Modelo multiclase con salida softmax (4 clases).
    """
    arr   = preprocess_para_efficientnet(imagen_input)
    probs = MODELOS["m4"].predict(arr, verbose=0)[0]   # shape (4,)

    idx_ganador = int(np.argmax(probs))
    clase       = CLASES_M4[idx_ganador]
    confianza   = float(probs[idx_ganador])

    return ModelPrediction(
        clase     = clase,
        confianza = confianza,
        baja_confianza = confianza < UMBRALES["m4"],
        probabilidades = {
            CLASES_M4[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
    )


def _predecir_m3(imagen_input) -> ModelPrediction:
    """
    M3 — Clasificador de deficiencias nutricionales.
    Modelo multiclase con salida softmax (8 clases).
    """
    arr   = preprocess_para_efficientnet(imagen_input)
    probs = MODELOS["m3"].predict(arr, verbose=0)[0]   # shape (8,)

    idx_ganador = int(np.argmax(probs))
    clase       = CLASES_M3[idx_ganador]
    confianza   = float(probs[idx_ganador])

    return ModelPrediction(
        clase     = clase,
        confianza = confianza,
        baja_confianza = confianza < UMBRALES["m3"],
        probabilidades = {
            CLASES_M3[i]: round(float(p), 4)
            for i, p in enumerate(probs)
        },
    )


# ─────────────────────────────────────────────────────
# FUNCIÓN PÚBLICA — única entrada al pipeline
# ─────────────────────────────────────────────────────

def run_pipeline(imagen_input: Union[str, bytes, "np.ndarray"]) -> PipelineResult:
    """
    Corre el pipeline completo sobre una imagen.

    Args:
        imagen_input: ruta (str), bytes, o np.ndarray

    Returns:
        PipelineResult con el diagnóstico y toda la trazabilidad
    """
    predicciones = {}

    # ── M1: ¿Es hoja de banano? ──────────────────────
    pred_m1 = _predecir_m1(imagen_input)
    predicciones["m1"] = pred_m1

    if pred_m1.clase == "non_banana_leaf":
        return PipelineResult(
            flujo        = FLUJO_NO_BANANA,
            llm_needed   = False,
            predicciones = predicciones,
        )

    # ── M4: ¿Tiene enfermedad? ───────────────────────
    pred_m4 = _predecir_m4(imagen_input)
    predicciones["m4"] = pred_m4

    if pred_m4.clase in CLASES_ENFERMEDAD:
        return PipelineResult(
            flujo          = FLUJO_ENFERMO,
            llm_needed     = True,
            diagnostico    = pred_m4.clase,
            confianza      = pred_m4.confianza,
            baja_confianza = pred_m4.baja_confianza,
            categoria      = "enfermedad",
            predicciones   = predicciones,
        )

    # M4 dijo Sano → correr M3
    # ── M3: ¿Tiene déficit nutricional? ──────────────
    pred_m3 = _predecir_m3(imagen_input)
    predicciones["m3"] = pred_m3

    if pred_m3.clase in CLASES_DEFICIT:
        return PipelineResult(
            flujo          = FLUJO_DEFICIT,
            llm_needed     = True,
            diagnostico    = pred_m3.clase,
            confianza      = pred_m3.confianza,
            baja_confianza = pred_m3.baja_confianza,
            categoria      = "deficiencia",
            predicciones   = predicciones,
        )

    # M3 también dijo Sano → planta completamente sana
    return PipelineResult(
        flujo        = FLUJO_SANO,
        llm_needed   = False,
        predicciones = predicciones,
    )
