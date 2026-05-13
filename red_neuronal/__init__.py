"""
red_neuronal — Paquete MLP desde cero para reconocimiento de dígitos
=====================================================================
Estructura del paquete:
  activaciones.py  →  sigmoid, TipoNodo
  conexion.py      →  Conexion  (enlace pesado entre nodos)
  nodo.py          →  Nodo      (neurona individual)
  capa.py          →  Capa      (forward + propagar_error)
  red.py           →  RedNeuronal  (MLP 784→64→10 completo)
  mnist.py         →  cargar_mnist (descarga y prepara el dataset)

Uso rápido
----------
  from red_neuronal import RedNeuronal, cargar_mnist

  red = RedNeuronal(tasa_aprendizaje=0.1, semilla=42)
  train, test = cargar_mnist()
  red.entrenar_numpy(train, epocas=10)
  digito, confianzas = red.predecir_con_confianza(vector_784)
"""

from .activaciones import sigmoid, TipoNodo
from .conexion import Conexion
from .nodo import Nodo
from .capa import Capa
from .red import RedNeuronal
from .mnist import cargar_mnist

__all__ = [
    "sigmoid",
    "TipoNodo",
    "Conexion",
    "Nodo",
    "Capa",
    "RedNeuronal",
    "cargar_mnist",
]
