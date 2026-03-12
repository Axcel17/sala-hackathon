# main.py
"""
Servidor FastAPI que expone el pipeline de diagnóstico de hojas de banano.

Endpoints:
    GET  /health   → estado del servidor y modelos
    POST /predict  → recibe imagen como archivo (multipart), devuelve diagnóstico + recomendación RAG
"""

import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import run_pipeline, _predecir_m1, _predecir_m3, _predecir_m4
from llm import obtener_recomendacion_fija, obtener_detalle_fijo, obtener_respuesta_estatica

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bananavision")

app = FastAPI(title="BananaVision API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# SCHEMAS
# ─────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    flujo: str
    llm_needed: bool
    diagnostico: str | None = None
    confianza: float | None = None
    confianza_pct: float | None = None
    baja_confianza: bool = False
    categoria: str | None = None
    recomendacion: str = ""
    detalle: dict | None = None


class ModelDirectResponse(BaseModel):
    clase: str
    confianza: float
    confianza_pct: float
    baja_confianza: bool
    probabilidades: dict[str, float]


# ─────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Verifica que el servidor y los modelos están listos."""
    return {"status": "ok", "modelos": ["m1", "m3", "m4"]}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen como archivo y devuelve el diagnóstico del pipeline
    más una recomendación generada por el RAG (OpenAI GPT-4o mini + documentos agrícolas).
    """
    image_bytes = await file.read()
    log.info(f"📥 Imagen recibida — {file.filename} ({len(image_bytes) / 1024:.1f} KB)")

    try:
        resultado = run_pipeline(image_bytes)
    except Exception as e:
        log.error(f"❌ Error en pipeline: {e}")
        raise HTTPException(status_code=500, detail=f"Error en el pipeline: {str(e)}")

    confianza_pct = round(resultado.confianza * 100, 1) if resultado.confianza else None
    baja = " ⚠️  baja confianza" if resultado.baja_confianza else ""
    diag = f"→ {resultado.diagnostico} ({confianza_pct}%){baja}" if resultado.diagnostico else ""
    log.info(f"✅ Resultado: flujo={resultado.flujo}  {diag}")

    # Respuesta fija basada en los documentos de DOCUMENTOS RAG
    if resultado.llm_needed and resultado.diagnostico:
        log.info(f"📄 Generando recomendación fija para '{resultado.diagnostico}'...")
        recomendacion = obtener_recomendacion_fija(resultado.diagnostico)
        detalle = obtener_detalle_fijo(resultado.diagnostico)
    else:
        recomendacion = obtener_respuesta_estatica(resultado.flujo)
        detalle = None

    return PredictResponse(
        flujo          = resultado.flujo,
        llm_needed     = resultado.llm_needed,
        diagnostico    = resultado.diagnostico,
        confianza      = resultado.confianza,
        confianza_pct  = confianza_pct,
        baja_confianza = resultado.baja_confianza,
        categoria      = resultado.categoria,
        recomendacion  = recomendacion,
        detalle        = detalle,
    )


@app.post("/predict/enfermedades", response_model=ModelDirectResponse)
async def predict_enfermedades(file: UploadFile = File(...)):
    """
    M1 → M4. Primero valida que sea hoja de banano y luego clasifica enfermedades.
    Clases M4: Cordana, Fusarium, Sano, SigatokaNegra.
    Retorna 400 si la imagen no es una hoja de banano.
    """
    image_bytes = await file.read()
    log.info(f"📥 [M1→M4] Imagen recibida — {file.filename} ({len(image_bytes) / 1024:.1f} KB)")

    try:
        pred_m1 = _predecir_m1(image_bytes)
    except Exception as e:
        log.error(f"❌ Error en M1: {e}")
        raise HTTPException(status_code=500, detail=f"Error en M1: {str(e)}")

    if pred_m1.clase == "non_banana_leaf":
        log.info(f"🚫 [M1] No es hoja de banano ({pred_m1.confianza * 100:.1f}%)")
        raise HTTPException(status_code=400, detail="La imagen no es una hoja de banano")

    log.info(f"✅ [M1] Hoja de banano ({pred_m1.confianza * 100:.1f}%) → corriendo M4...")

    try:
        pred = _predecir_m4(image_bytes)
    except Exception as e:
        log.error(f"❌ Error en M4: {e}")
        raise HTTPException(status_code=500, detail=f"Error en M4: {str(e)}")

    log.info(f"✅ [M4] {pred.clase} ({pred.confianza * 100:.1f}%)")
    return ModelDirectResponse(
        clase          = pred.clase,
        confianza      = pred.confianza,
        confianza_pct  = round(pred.confianza * 100, 1),
        baja_confianza = pred.baja_confianza,
        probabilidades = pred.probabilidades,
    )


@app.post("/predict/nutrientes", response_model=ModelDirectResponse)
async def predict_nutrientes(file: UploadFile = File(...)):
    """
    M1 → M3. Primero valida que sea hoja de banano y luego clasifica deficiencias nutricionales.
    Clases M3: Boron, Calcium, Iron, Magnesium, Manganese, Potassium, Sano, Sulphur.
    Retorna 400 si la imagen no es una hoja de banano.
    """
    image_bytes = await file.read()
    log.info(f"📥 [M1→M3] Imagen recibida — {file.filename} ({len(image_bytes) / 1024:.1f} KB)")

    try:
        pred_m1 = _predecir_m1(image_bytes)
    except Exception as e:
        log.error(f"❌ Error en M1: {e}")
        raise HTTPException(status_code=500, detail=f"Error en M1: {str(e)}")

    if pred_m1.clase == "non_banana_leaf":
        log.info(f"🚫 [M1] No es hoja de banano ({pred_m1.confianza * 100:.1f}%)")
        raise HTTPException(status_code=400, detail="La imagen no es una hoja de banano")

    log.info(f"✅ [M1] Hoja de banano ({pred_m1.confianza * 100:.1f}%) → corriendo M3...")

    try:
        pred = _predecir_m3(image_bytes)
    except Exception as e:
        log.error(f"❌ Error en M3: {e}")
        raise HTTPException(status_code=500, detail=f"Error en M3: {str(e)}")

    log.info(f"✅ [M3] {pred.clase} ({pred.confianza * 100:.1f}%)")
    return ModelDirectResponse(
        clase          = pred.clase,
        confianza      = pred.confianza,
        confianza_pct  = round(pred.confianza * 100, 1),
        baja_confianza = pred.baja_confianza,
        probabilidades = pred.probabilidades,
    )
