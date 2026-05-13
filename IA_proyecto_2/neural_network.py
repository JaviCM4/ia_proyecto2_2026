"""
neural_network.py — Módulo de compatibilidad
=============================================
Todo el código de la Red Neuronal fue reorganizado en el paquete red_neuronal/.
Este archivo re-exporta todo para que app.py y entrenar_mnist.py sigan
funcionando sin ningún cambio.

  red_neuronal/activaciones.py  →  sigmoid, TipoNodo
  red_neuronal/conexion.py      →  Conexion
  red_neuronal/nodo.py          →  Nodo
  red_neuronal/capa.py          →  Capa
  red_neuronal/red.py           →  RedNeuronal
  red_neuronal/mnist.py         →  cargar_mnist
"""

from red_neuronal import sigmoid, TipoNodo, Conexion, Nodo, Capa, RedNeuronal, cargar_mnist

__all__ = ["sigmoid", "TipoNodo", "Conexion", "Nodo", "Capa", "RedNeuronal", "cargar_mnist"]
