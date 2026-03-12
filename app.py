# app.py
"""
API REST de BananaVision.
Expone el pipeline completo + Gemini como un solo endpoint.

⚙️ PARÁMETRO — Para producción: agregar autenticación, rate limiting,
logging estructurado, y manejo de errores más granular.
"""

import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from pipeline.inference import run_pipeline
from pipeline.config import FLUJO_SANO, FLUJO_NO_BANANA
from llm import generar_recomendacion, generar_recomendacion_con_imagen, obtener_respuesta_estatica

app = FastAPI(
    title="BananaVision AI",
    description="Diagnóstico de enfermedades y deficiencias nutricionales en banano",
    version="1.0.0"
)

# ⚙️ PARÁMETRO — En producción restringir origins a tu dominio del frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# SCHEMAS DE RESPUESTA
# ─────────────────────────────────────────────────────

class DiagnosticoResponse(BaseModel):
    # Resultado del pipeline
    flujo: str                    # NO_BANANA | ENFERMO | DEFICIT | SANO
    diagnostico: Optional[str]    # Nombre de la enfermedad o nutriente
    confianza_pct: Optional[float]
    baja_confianza: bool
    categoria: Optional[str]      # "enfermedad" | "deficiencia" | None

    # Recomendación generada por Gemini
    recomendacion: str

    # Trazabilidad — qué modelos corrieron y con qué confianza
    modelos_corridos: dict


# ─────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "sistema": "BananaVision AI v1.0"}


@app.post("/diagnostico", response_model=DiagnosticoResponse)
async def diagnosticar(
    imagen: UploadFile = File(...),
    usar_vision_gemini: bool = False   # ⚙️ PARÁMETRO: True = Gemini también ve la foto
):
    """
    Endpoint principal. Recibe una imagen y devuelve el diagnóstico completo.

    - Corre el pipeline M1 → M4 → M3 en cascada
    - Si llm_needed=True: llama a Gemini para generar la recomendación
    - Si llm_needed=False: devuelve mensaje estático
    """

    # Validar que es una imagen
    if not imagen.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen (jpg, png, etc.)"
        )

    # Leer bytes de la imagen
    imagen_bytes = await imagen.read()

    # Correr pipeline de modelos
    resultado = run_pipeline(imagen_bytes)

    # Generar recomendación
    if resultado.llm_needed:
        try:
            if usar_vision_gemini:
                recomendacion = generar_recomendacion_con_imagen(resultado, imagen_bytes)
            else:
                recomendacion = generar_recomendacion(resultado)
        except Exception as e:
            import traceback
            print(f"[GEMINI ERROR] {e}")
            traceback.print_exc()
            recomendacion = obtener_respuesta_estatica(resultado.flujo)
    else:
        recomendacion = obtener_respuesta_estatica(resultado.flujo)

    # Serializar trazabilidad de modelos
    modelos_corridos = {
        modelo_id: {
            "clase": pred.clase,
            "confianza_pct": round(pred.confianza * 100, 1),
            "baja_confianza": pred.baja_confianza,
        }
        for modelo_id, pred in resultado.predicciones.items()
    }

    ctx = resultado.to_llm_context()

    return DiagnosticoResponse(
        flujo=resultado.flujo,
        diagnostico=ctx["diagnostico"],
        confianza_pct=ctx["confianza_pct"],
        baja_confianza=ctx["baja_confianza"],
        categoria=ctx["categoria"],
        recomendacion=recomendacion,
        modelos_corridos=modelos_corridos,
    )


@app.get("/modelos/estado")
def estado_modelos():
    """
    Verifica que los 3 modelos están cargados en memoria.
    Útil para health checks del frontend.
    """
    from pipeline.model_loader import MODELOS
    return {
        "modelos_cargados": list(MODELOS.keys()),
        "total": len(MODELOS),
        "listos": len(MODELOS) == 3,
    }
