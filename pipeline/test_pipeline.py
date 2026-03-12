# test_pipeline.py
"""
Script de verificación del pipeline completo.
Corre esto ANTES de conectar el LLM para confirmar
que los modelos cargan y predicen correctamente.

Uso:
    python test_pipeline.py --imagen /ruta/imagen.jpg
    python test_pipeline.py --test_dir /ruta/carpeta/  # prueba múltiples imágenes
"""

import argparse
import os
import sys
import json
import time

# Allow running this script directly from the pipeline/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline
from pipeline.config import FLUJO_NO_BANANA, FLUJO_ENFERMO, FLUJO_DEFICIT, FLUJO_SANO


def probar_imagen(ruta: str, verbose: bool = True) -> dict:
    """Corre el pipeline sobre una imagen y devuelve el resultado."""
    inicio = time.time()
    resultado = run_pipeline(ruta)
    elapsed = time.time() - inicio

    if verbose:
        print(resultado)
        print(f"  Tiempo total   : {elapsed * 1000:.0f}ms")
        print(f"  Modelos corridos: {list(resultado.predicciones.keys())}")
        print()

    return {
        "imagen":    os.path.basename(ruta),
        "flujo":     resultado.flujo,
        "diagnostico": resultado.diagnostico,
        "confianza": round(resultado.confianza * 100, 1) if resultado.confianza else None,
        "tiempo_ms": round(elapsed * 1000),
        "llm_needed": resultado.llm_needed,
    }


def probar_directorio(ruta_dir: str):
    """
    Prueba todas las imágenes de un directorio.
    Útil para verificar el pipeline con múltiples casos.
    """
    extensiones = {'.jpg', '.jpeg', '.png', '.bmp'}
    imagenes = [
        os.path.join(ruta_dir, f)
        for f in os.listdir(ruta_dir)
        if os.path.splitext(f)[1].lower() in extensiones
    ]

    if not imagenes:
        print(f"No se encontraron imágenes en: {ruta_dir}")
        return

    print(f"Probando {len(imagenes)} imágenes...\n")

    resultados = []
    conteo_flujos = {
        FLUJO_NO_BANANA: 0,
        FLUJO_ENFERMO:   0,
        FLUJO_DEFICIT:   0,
        FLUJO_SANO:      0,
    }

    for ruta in sorted(imagenes):
        r = probar_imagen(ruta, verbose=True)
        resultados.append(r)
        conteo_flujos[r["flujo"]] = conteo_flujos.get(r["flujo"], 0) + 1

    # Resumen
    tiempos = [r["tiempo_ms"] for r in resultados]
    print("=" * 55)
    print("  RESUMEN")
    print("=" * 55)
    print(f"  Total imágenes  : {len(resultados)}")
    print(f"  Tiempo promedio : {sum(tiempos) / len(tiempos):.0f}ms")
    print(f"  Tiempo máximo   : {max(tiempos)}ms")
    print(f"  Tiempo mínimo   : {min(tiempos)}ms")
    print()
    print("  Distribución de flujos:")
    for flujo, n in conteo_flujos.items():
        if n > 0:
            pct = n / len(resultados) * 100
            print(f"    {flujo:<15}: {n:>3} ({pct:.0f}%)")
    print("=" * 55)


def verificar_contexto_llm(ruta: str):
    """
    Muestra exactamente qué va a recibir el LLM.
    Útil para verificar antes de conectar el RAG.
    """
    resultado = run_pipeline(ruta)

    print("\n" + "=" * 55)
    print("  CONTEXTO QUE RECIBIRÁ EL LLM")
    print("=" * 55)

    if not resultado.llm_needed:
        print(f"  LLM no necesario — flujo: {resultado.flujo}")
        print("  Se usará mensaje estático.")
    else:
        ctx = resultado.to_llm_context()
        print(json.dumps(ctx, indent=2, ensure_ascii=False))

        print()
        print("  Probabilidades del modelo diagnóstico:")
        modelo_key = "m4" if resultado.categoria == "enfermedad" else "m3"
        pred = resultado.predicciones.get(modelo_key)
        if pred:
            for clase, prob in sorted(pred.probabilidades.items(),
                                      key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 30)
                print(f"    {clase:<20}: {prob*100:5.1f}%  {bar}")

    print("=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verificar pipeline BananaVision")
    parser.add_argument("--imagen",   type=str, help="Ruta a una imagen")
    parser.add_argument("--test_dir", type=str, help="Directorio con imágenes de prueba")
    parser.add_argument("--llm_ctx",  type=str, help="Ver contexto LLM para una imagen")
    args = parser.parse_args()

    if args.imagen:
        probar_imagen(args.imagen, verbose=True)

    elif args.test_dir:
        probar_directorio(args.test_dir)

    elif args.llm_ctx:
        verificar_contexto_llm(args.llm_ctx)

    else:
        # Sin argumentos: prueba rápida de que los modelos cargan
        print("Pipeline inicializado correctamente.")
        print("Modelos cargados: M1, M3, M4")
        print("\nUso:")
        print("  python test_pipeline.py --imagen /ruta/imagen.jpg")
        print("  python test_pipeline.py --test_dir /ruta/carpeta/")
        print("  python test_pipeline.py --llm_ctx /ruta/imagen.jpg")
