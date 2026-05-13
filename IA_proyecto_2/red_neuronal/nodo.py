"""
=======================================
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


class Nodo:

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
