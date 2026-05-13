"""
mnist.py — Descarga y preparación del dataset MNIST
=====================================================
Usa únicamente urllib, gzip, struct y numpy (todos permitidos/built-in).
Los archivos se guardan en .mnist_cache/ junto al proyecto para que
solo se descarguen una vez.

Retorna listas de (vector_784_float32, etiqueta_int):
  entrenamiento : 60 000 muestras
  prueba        :  10 000 muestras
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  cargar_mnist
# ══════════════════════════════════════════════════════════════════════════════

def cargar_mnist() -> Tuple[
    List[Tuple[np.ndarray, int]],
    List[Tuple[np.ndarray, int]],
]:
    """
    Descarga y prepara el dataset MNIST sin depender de paquetes externos.
    Usa únicamente urllib, gzip, struct y numpy (todos permitidos/built-in).

    Los archivos se guardan en  .mnist_cache/  junto al script para que
    solo se descarguen una vez.

    Retorna
    -------
    entrenamiento : list of (vector_784_float32, etiqueta_int)  — 60 000 muestras
    prueba        : list of (vector_784_float32, etiqueta_int)  —  10 000 muestras
    """
    import gzip
    import os
    import struct
    import urllib.request

    # Mirror de Google; estable y sin necesidad de paquetes externos
    BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    ARCHIVOS = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".mnist_cache")
    os.makedirs(cache_dir, exist_ok=True)

    def _descargar(clave: str) -> str:
        nombre = ARCHIVOS[clave]
        local  = os.path.join(cache_dir, nombre)
        if not os.path.exists(local):
            url = BASE_URL + nombre
            print(f"[MNIST] Descargando {nombre} ...")
            urllib.request.urlretrieve(url, local)
            print(f"[MNIST] Guardado en {local}")
        return local

    def _leer_imagenes(ruta: str) -> np.ndarray:
        with gzip.open(ruta, "rb") as f:
            _magic, n, filas, cols = struct.unpack(">IIII", f.read(16))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, filas * cols).astype(np.float32) / 255.0

    def _leer_etiquetas(ruta: str) -> np.ndarray:
        with gzip.open(ruta, "rb") as f:
            _magic, n = struct.unpack(">II", f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.astype(int)

    print("[MNIST] Cargando dataset...")
    X_train = _leer_imagenes(_descargar("train_images"))
    y_train = _leer_etiquetas(_descargar("train_labels"))
    X_test  = _leer_imagenes(_descargar("test_images"))
    y_test  = _leer_etiquetas(_descargar("test_labels"))

    entrenamiento = [(X_train[i], int(y_train[i])) for i in range(len(X_train))]
    prueba        = [(X_test[i],  int(y_test[i]))  for i in range(len(X_test))]

    print(
        f"[MNIST] Entrenamiento: {len(entrenamiento)} muestras  |  "
        f"Prueba: {len(prueba)} muestras"
    )
    return entrenamiento, prueba
