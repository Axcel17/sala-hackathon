---
title: Bananavision Api
emoji: 🦀
colorFrom: red
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# BananaVision API

Sistema de diagnóstico automático de hojas de banano mediante visión computacional. Detecta enfermedades y deficiencias nutricionales a partir de una fotografía, y entrega recomendaciones agronómicas accionables.

---

## El problema

Los agricultores de banano pierden cosechas enteras por no detectar a tiempo enfermedades como Sigatoka Negra o Fusarium, o déficits de nutrientes que reducen el rendimiento. El diagnóstico experto es costoso, lento y no siempre accesible en zonas rurales.

## La solución

BananaVision permite tomar una foto de la hoja con cualquier dispositivo y obtener en segundos un diagnóstico preciso con pasos concretos de tratamiento — sin necesidad de un agrónomo en campo.

---

## Cómo funciona

El sistema corre tres modelos de deep learning en cascada:

```
Foto de hoja
     │
     ▼
  M1 — MobileNetV3
  ¿Es hoja de banano?
     │
     ├─ NO  →  "Imagen no válida"
     │
     └─ SÍ
          │
          ▼
       M4 — EfficientNetB3
       ¿Tiene enfermedad?
          │
          ├─ Cordana / Fusarium / Sigatoka Negra
          │       → Diagnóstico + recomendación
          │
          └─ Sano
               │
               ▼
            M3 — EfficientNetB3
            ¿Tiene déficit nutricional?
               │
               ├─ Boro / Calcio / Hierro / Magnesio /
               │  Manganeso / Potasio / Azufre
               │       → Diagnóstico + recomendación
               │
               └─ Sano  →  "Planta saludable"
```

Cada diagnóstico incluye:

- Clase detectada y nivel de confianza
- Advertencia si la confianza es baja
- Recomendación con acciones ordenadas por urgencia
- Productos sugeridos y dónde conseguirlos

---

## Modelos

| Modelo | Arquitectura     | Clases | Tarea                           |
| ------ | ---------------- | ------ | ------------------------------- |
| M1     | MobileNetV3Small | 2      | Validar que sea hoja de banano  |
| M4     | EfficientNetB3   | 4      | Detectar enfermedades           |
| M3     | EfficientNetB3   | 8      | Detectar déficits nutricionales |

**Enfermedades detectadas (M4):** Cordana, Fusarium, Sigatoka Negra, Sano

**Déficits detectados (M3):** Boro, Calcio, Hierro, Magnesio, Manganeso, Potasio, Azufre, Sano

---

## API

### `GET /health`

Verifica que el servidor y los tres modelos estén listos.

```json
{ "status": "ok", "modelos": ["m1", "m3", "m4"] }
```

---

### `POST /predict`

Pipeline completo: M1 → M4 → M3. Devuelve diagnóstico y recomendación.

**Request:** `multipart/form-data` con campo `file` (imagen JPG/PNG)

```json
{
  "flujo": "ENFERMO",
  "llm_needed": true,
  "diagnostico": "SigatokaNegra",
  "confianza": 0.91,
  "confianza_pct": 91.0,
  "baja_confianza": false,
  "categoria": "enfermedad",
  "recomendacion": "Aplicar fungicida sistémico...",
  "detalle": { ... }
}
```

---

### `POST /predict/enfermedades`

M1 → M4. Solo clasifica enfermedades. Devuelve probabilidades de las 4 clases.

```json
{
  "clase": "Fusarium",
  "confianza": 0.87,
  "confianza_pct": 87.0,
  "baja_confianza": false,
  "probabilidades": {
    "Cordana": 0.04,
    "Fusarium": 0.87,
    "Sano": 0.05,
    "SigatokaNegra": 0.04
  }
}
```

---

### `POST /predict/nutrientes`

M1 → M3. Solo clasifica déficits nutricionales. Devuelve probabilidades de las 8 clases.

```json
{
  "clase": "Potassium",
  "confianza": 0.78,
  "confianza_pct": 78.0,
  "baja_confianza": false,
  "probabilidades": {
    "Boron": 0.01,
    "Calcium": 0.03,
    "Iron": 0.02,
    "Magnesium": 0.05,
    "Manganese": 0.04,
    "Potassium": 0.78,
    "Sano": 0.06,
    "Sulphur": 0.01
  }
}
```

---

## Stack técnico

- **Deep Learning:** TensorFlow / Keras — MobileNetV3 + EfficientNetB3
- **API:** FastAPI + Uvicorn
- **Base de conocimiento:** 11 documentos agronómicos estructurados (Sigatoka, Fusarium, Cordana, micronutrientes)
- **Deploy:** Docker — HuggingFace Spaces

---

## Correr localmente

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Documentación interactiva disponible en `http://localhost:8000/docs`

---

## App

[https://github.com/Jmuniz27/sala_ai_app](BananaVisionAI)

---

## Demo

<video src="Video/SalaDemo.mp4" controls width="100%"></video>
