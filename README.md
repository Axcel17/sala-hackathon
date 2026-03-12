# BananaVision AI

Sistema de diagnóstico de enfermedades y deficiencias nutricionales en banano mediante visión computacional y LLM.

## Cómo funciona

```
Imagen → M1 (¿es banano?) → M4 (¿enfermedad?) → M3 (¿déficit nutricional?) → GPT-4o mini → Recomendación
```

- **M1** — MobileNetV3Small: valida que la imagen sea hoja de banano
- **M4** — EfficientNetB3: detecta enfermedades (Cordana, Fusarium, Sigatoka Negra)
- **M3** — EfficientNetB3: detecta deficiencias nutricionales (Calcio, Potasio, Hierro, etc.)
- **GPT-4o mini**: genera recomendación en español simple para el agricultor, usando los documentos de `DOCUMENTOS RAG/`

## Requisitos

- Python 3.12
- Los modelos `.keras` en la carpeta `models/` (via Git LFS)
- API key de OpenAI

## Instalación

```bash
# 1. Clonar el repositorio
git clone git@github.com:Axcel17/sala-hackathon.git
cd sala-hackathon

# 2. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Configurar API Key

```bash
export OPENAI_API_KEY="sk-..."
```

Para no tener que exportarla cada vez, agrégala a tu `~/.bashrc` o `~/.zshrc`:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

## Ejecutar el servidor

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/diagnostico` | Diagnóstico completo desde imagen |
| `GET` | `/modelos/estado` | Verifica que los 3 modelos están cargados |

### POST /diagnostico

**Parámetros:**
- `imagen` (form-data): archivo de imagen JPG/PNG
- `usar_vision_gemini` (query, opcional): `false` por defecto

**Ejemplo con curl:**
```bash
curl -X POST 'http://localhost:8000/diagnostico' \
  -F 'imagen=@foto_hoja.jpg;type=image/jpeg'
```

**Respuesta:**
```json
{
  "flujo": "ENFERMO",
  "diagnostico": "Fusarium",
  "confianza_pct": 96.5,
  "baja_confianza": false,
  "categoria": "enfermedad",
  "recomendacion": "Es urgente que contactes a Agrocalidad...",
  "modelos_corridos": {
    "m1": { "clase": "banana_leaf", "confianza_pct": 99.8, "baja_confianza": false },
    "m4": { "clase": "Fusarium", "confianza_pct": 96.5, "baja_confianza": false }
  }
}
```

**Flujos posibles:**

| `flujo` | Significado |
|---------|-------------|
| `NO_BANANA` | La imagen no es una hoja de banano |
| `ENFERMO` | Se detectó una enfermedad |
| `DEFICIT` | Se detectó deficiencia nutricional |
| `SANO` | La planta está sana |

## Documentación interactiva

Con el servidor corriendo, abre: [http://localhost:8000/docs](http://localhost:8000/docs)

## Estructura del proyecto

```
├── app.py                  # API REST (FastAPI)
├── requirements.txt
├── pipeline/
│   ├── inference.py        # Orquesta M1 → M4 → M3
│   ├── model_loader.py     # Carga los modelos al arrancar
│   ├── config.py           # Clases, rutas, umbrales
│   ├── preprocessor.py     # Preprocesamiento de imágenes
│   └── schemas.py          # PipelineResult, ModelPrediction
├── llm/
│   ├── gemini_client.py    # Cliente OpenAI GPT-4o mini
│   ├── base_conocimiento.py # Carga documentos RAG
│   └── respuestas_estaticas.py # Respuestas para SANO y NO_BANANA
├── DOCUMENTOS RAG/         # Base de conocimiento agrícola (.md)
│   ├── fusarium.md
│   ├── black-sigatoka.md
│   ├── cordana.md
│   ├── potassium.md
│   └── ...
└── models/
    ├── m1_validador.keras
    ├── m3_nutrientes.keras
    └── m4_enfermedades.keras
```
