"""
activaciones.py — Función de activación y enumeración de tipos de nodo
=======================================================================
Contiene los bloques más básicos compartidos por toda la red:
  - sigmoid : función de activación con recorte numérico
  - TipoNodo: etiqueta que identifica a qué capa pertenece cada nodo
"""

from __future__ import annotations

import math
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
#  Función de activación
# ══════════════════════════════════════════════════════════════════════════════

def sigmoid(z: float) -> float:
    """σ(z) = 1 / (1 + e^-z)  —  con recorte para estabilidad numérica."""
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(-z))


# ══════════════════════════════════════════════════════════════════════════════
#  TipoNodo
# ══════════════════════════════════════════════════════════════════════════════

class TipoNodo(Enum):
    INPUT  = "INPUT"
    HIDDEN = "HIDDEN"
    OUTPUT = "OUTPUT"
