"""
nodo.py — Neurona individual de la red
=======================================
Cada Nodo almacena su activación (valor), la suma ponderada previa (z)
y el error acumulado durante backprop, además de las listas de conexiones
de entrada y salida.
"""

from __future__ import annotations

from typing import List

from .activaciones import TipoNodo
from .conexion import Conexion


# ══════════════════════════════════════════════════════════════════════════════
#  Nodo
# ══════════════════════════════════════════════════════════════════════════════

class Nodo:
    """
    Neurona individual de la red.

    Atributos
    ---------
    id                 : índice dentro de su capa
    tipo               : TipoNodo (INPUT | HIDDEN | OUTPUT)
    valor              : a = σ(z)  →  resultado DESPUÉS de sigmoid;
                         reutilizado en backprop sin recalcular
    z                  : sumatoria Σ(W×O) ANTES de sigmoid
    error              : suma de todos los errores que llegaron al nodo
    conexiones_entrada : List[Conexion] cuyo nodo_destino es éste
    conexiones_salida  : List[Conexion] cuyo nodo_origen  es éste
    """

    def __init__(self, id: int, tipo: TipoNodo) -> None:
        self.id:    int      = id
        self.tipo:  TipoNodo = tipo
        self.valor: float    = 0.0
        self.z:     float    = 0.0
        self.error: float    = 0.0
        self.conexiones_entrada: List[Conexion] = []
        self.conexiones_salida:  List[Conexion] = []

    def __repr__(self) -> str:
        return (
            f"Nodo({self.tipo.value}[{self.id}]  "
            f"val={self.valor:.4f}  z={self.z:.4f}  err={self.error:.4f})"
        )
