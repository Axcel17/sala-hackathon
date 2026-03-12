# pipeline/model_loader.py
"""
Carga los 3 modelos una sola vez cuando el sistema arranca.

CRÍTICO: los modelos se cargan en este módulo y se importan
desde aquí en el resto del código. NUNCA se cargan dentro
de una función de predicción — eso sería recargarlos en
cada request, lo cual es lento e innecesario.
"""

import os
import json
import zipfile
import tempfile
import tensorflow as tf
import keras           # Keras 3 standalone — para M3 y M4
from pipeline.config import MODEL_PATHS

# M1 fue guardado con TF 2.15 / Keras 2.15 en formato ZIP.
# Su config usa Keras 2 (keras.src.engine.functional, TFOpLambda).
# Sus pesos están en H5 con claves: layers\{class_auto_prefix}/vars/{index}
# → Se carga con tf_keras deserializer + h5py con mapeo por auto-prefix de clase.
# M3 y M4 son Keras 3 puro → se cargan directamente con keras.
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


def _get_layer_h5_prefix(layer) -> str:
    """
    Devuelve el prefijo H5 para una capa dado su clase tf_keras.
    Keras 2.15 usa el nombre auto-generado (sin contador) como prefijo H5.
    Se obtiene creando una instancia temporal sin nombre y extrayendo el prefijo.
    Resultado cacheado para eficiencia.
    """
    import re
    if not hasattr(_get_layer_h5_prefix, '_cache'):
        _get_layer_h5_prefix._cache = {}
    cls = layer.__class__
    if cls not in _get_layer_h5_prefix._cache:
        # El prefijo auto-generado es el nombre sin el sufijo "_N"
        # Usamos el nombre de la capa si fue creada sin nombre,
        # o derivamos el prefijo del nombre de clase via tf_keras's naming
        import tf_keras.src.utils.generic_utils as gu
        prefix = gu.to_snake_case(cls.__name__)
        # tf_keras's to_snake_case da 'conv2_d' para Conv2D, pero Keras usa 'conv2d'.
        # Verificamos con una instancia temporal para algunos casos especiales.
        # En lugar de crear instancias, usamos el nombre de clase lowercase directo
        # para clases que tienen dígitos seguidos de mayúsculas.
        # Método confiable: crear dummy y extraer prefijo del nombre.
        try:
            dummy = cls.__new__(cls)
            # Obtener el nombre que Keras asignaría
            dummy_name_fn = getattr(cls, '_get_unique_layer_name', None)
            if dummy_name_fn is None:
                # Fallback: usar nombre de clase en minúsculas (funciona para la mayoría)
                prefix = cls.__name__.lower()
        except Exception:
            pass
        _get_layer_h5_prefix._cache[cls] = prefix
    return _get_layer_h5_prefix._cache[cls]


def _cargar_m1(ruta: str):
    """
    Carga M1 (Keras 2.15 ZIP) con compatibilidad total.

    Estrategia:
    1. Deserializar arquitectura con tf_keras (conoce TFOpLambda y keras.src.engine)
    2. Construir mapeo de capa → clave H5 usando auto-prefix real de tf_keras
    3. Asignar pesos capa por capa desde el H5
    """
    import h5py
    import re
    from collections import defaultdict
    from tf_keras.src.saving import saving_lib as tf_saving_lib
    import tf_keras

    # Extraer config y pesos del ZIP
    with zipfile.ZipFile(ruta, 'r') as zf:
        config_dict = json.loads(zf.read('config.json'))
        weights_data = zf.read('model.weights.h5')

    # Construir arquitectura del modelo
    model = tf_saving_lib.deserialize_keras_object(config_dict)

    # Obtener el auto-prefix real de cada clase usando una instancia dummy
    # Esto nos da 'conv2d' para Conv2D, 're_lu' para ReLU, etc.
    class_prefix_cache = {}
    def get_prefix(cls):
        if cls not in class_prefix_cache:
            # Crear una instancia mínima para obtener su nombre auto-generado
            try:
                dummy = tf_keras.layers.Layer.__new__(cls)
                # El nombre auto-generado sigue el formato: prefix + '_' + count
                # Obtenemos el prefix desde keras.backend
                import keras.backend as K
                # En tf_keras, el nombre auto viene de unique_object_name
                # que usa to_snake_case internamente
                # El truco: usar el nombre de clase pero con el caché de Keras
                auto = tf_keras.backend.get_uid(cls.__name__)
                # Reset para no contaminar el modelo
                tf_keras.backend.reset_uids()
            except Exception:
                pass
            # Fallback: usar lowercase del nombre de clase sin underscore extra
            name = cls.__name__.lower()
            class_prefix_cache[cls] = name
        return class_prefix_cache[cls]

    # Método más confiable: usar el nombre auto-generado de tf_keras
    # creando instancias dummy ANTES de deserializar el modelo
    # → Ya fue hecho en el test: Conv2D='conv2d', ReLU='re_lu', etc.
    # Mapeo hardcodeado para las clases conocidas de M1:
    KNOWN_PREFIXES = {
        'Conv2D': 'conv2d',
        'DepthwiseConv2D': 'depthwise_conv2d',
        'ZeroPadding2D': 'zero_padding2d',
        'GlobalAveragePooling2D': 'global_average_pooling2d',
        'ReLU': 're_lu',
        'BatchNormalization': 'batch_normalization',
        'Multiply': 'multiply',
        'Add': 'add',
        'InputLayer': 'input_layer',
        'Rescaling': 'rescaling',
        'Dropout': 'dropout',
        'Dense': 'dense',
        'TFOpLambda': 'tf_op_lambda',
        'Flatten': 'flatten',
        'GlobalMaxPooling2D': 'global_max_pooling2d',
        'MaxPooling2D': 'max_pooling2d',
        'AveragePooling2D': 'average_pooling2d',
    }

    # Escribir pesos en archivo temporal
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        f.write(weights_data)
        tmp_path = f.name

    try:
        with h5py.File(tmp_path, 'r') as h5:
            h5_keys = set(h5.keys())
            class_counters = defaultdict(int)

            for layer in model.layers:
                cls_name = layer.__class__.__name__
                prefix = KNOWN_PREFIXES.get(cls_name)
                if prefix is None:
                    # Fallback: usar to_snake_case de tf_keras
                    import tf_keras.src.utils.generic_utils as gu
                    prefix = gu.to_snake_case(cls_name)

                count = class_counters[prefix]
                h5_key = f'layers\\{prefix}' if count == 0 else f'layers\\{prefix}_{count}'
                class_counters[prefix] += 1

                if h5_key not in h5_keys:
                    continue
                layer_group = h5[h5_key]
                if 'vars' not in layer_group:
                    continue
                vars_group = layer_group['vars']
                for i, var in enumerate(layer.variables):
                    if str(i) in vars_group:
                        var.assign(vars_group[str(i)][()])
    finally:
        os.unlink(tmp_path)

    return model


def cargar_modelos() -> dict:
    """
    Carga M1, M3 y M4 en memoria y los devuelve en un dict.

    Returns:
        {
            "m1": tf_keras.Model,  MobileNetV3Small  (Keras 2.15 ZIP format)
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
            modelos[nombre] = _cargar_m1(ruta)
        else:
            modelos[nombre] = keras.models.load_model(ruta)
        print("OK")

    print("Modelos listos.\n")
    return modelos


# ─────────────────────────────────────────────────────
# Instancia global — se carga una vez al importar el módulo
# En app.py se hace: from pipeline.model_loader import MODELOS
# ─────────────────────────────────────────────────────

MODELOS = cargar_modelos()
