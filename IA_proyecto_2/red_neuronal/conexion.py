"""
conexion.py — Enlace dirigido y pesado entre dos nodos
=======================================================
Cada Conexion une un nodo origen con un nodo destino y almacena:
  - peso  : valor del peso sináptico (inicializado aleatoriamente)
  - error : error que viajó por esta conexión durante backprop
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Solo se usa para las anotaciones de tipo; no genera importación circular
    from .nodo import Nodo


# ══════════════════════════════════════════════════════════════════════════════
#  Conexion
# ══════════════════════════════════════════════════════════════════════════════

class Conexion:
    """
    Enlace dirigido y pesado entre dos nodos.

    Atributos
    ---------
    nodo_origen  : referencia directa al nodo fuente
    nodo_destino : referencia directa al nodo destino
    peso         : float inicializado en U(-0.5, 0.5)
    error        : error que viajó POR esta conexión específica (backprop)
    """

    def __init__(self, nodo_origen: Nodo, nodo_destino: Nodo) -> None:
        self.nodo_origen:  Nodo  = nodo_origen
        self.nodo_destino: Nodo  = nodo_destino
        self.peso:         float = random.uniform(-0.5, 0.5)
        self.error:        float = 0.0

    def __repr__(self) -> str:
        return (
            f"Conexion(IN={self.nodo_origen.id} → OUT={self.nodo_destino.id}  "
            f"w={self.peso:.4f}  err={self.error:.4f})"
        )
