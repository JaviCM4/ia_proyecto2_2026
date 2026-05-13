"""
capa.py — Colección de nodos del mismo tipo (INPUT, HIDDEN u OUTPUT)
=====================================================================
La clase Capa agrupa nodos y ejecuta:
  - forward()        → propagación hacia adelante (z y sigmoid)
  - propagar_error() → distribución de errores hacia atrás (backprop)
"""

from __future__ import annotations

from typing import List

from .activaciones import TipoNodo, sigmoid
from .nodo import Nodo


# ══════════════════════════════════════════════════════════════════════════════
#  Capa
# ══════════════════════════════════════════════════════════════════════════════

class Capa:
    """
    Colección de nodos del mismo tipo (INPUT, HIDDEN u OUTPUT).

    Métodos principales
    -------------------
    forward()        → calcula z y aplica sigmoid en cada nodo
    propagar_error() → distribuye el error hacia atrás por los pesos
    """

    def __init__(self, tipo: TipoNodo, num_nodos: int) -> None:
        self.tipo:  TipoNodo   = tipo
        self.nodos: List[Nodo] = [Nodo(i, tipo) for i in range(num_nodos)]

    # ── Forward Propagation ───────────────────────────────────────────────────

    def forward(self) -> None:
        """
        Para cada nodo (excepto INPUT):
          z     = Σ (valor_nodo_anterior × peso_conexion)
          valor = σ(z)
        Ambos se almacenan en el nodo para reutilizarlos en backprop.
        """
        if self.tipo == TipoNodo.INPUT:
            return  # valores INPUT son asignados externamente

        for nodo in self.nodos:
            nodo.z     = sum(c.nodo_origen.valor * c.peso
                             for c in nodo.conexiones_entrada)
            nodo.valor = sigmoid(nodo.z)

    # ── Backpropagation (repartir error hacia atrás) ──────────────────────────

    def propagar_error(self) -> None:
        """
        Para cada nodo de ESTA capa, distribuye su contribución de error hacia
        atrás usando la fracción de peso de cada conexión de salida:

          error_conexion = error_nodo_destino × (peso / Σ|pesos_salida|)
          error_nodo     = Σ error_conexion   (acumulado sobre todas las salidas)

        El error calculado se guarda en la Conexion y en el Nodo origen.

        Nota: se usa |peso| en el denominador para garantizar que la suma
        sea siempre positiva (estabilidad numérica con pesos de signo mixto).
        """
        for nodo in self.nodos:
            if not nodo.conexiones_salida:
                continue

            suma_abs_pesos = sum(abs(c.peso) for c in nodo.conexiones_salida)
            if suma_abs_pesos < 1e-12:
                continue

            nodo.error = 0.0
            for conexion in nodo.conexiones_salida:
                # Error que viajó POR esta conexión específica
                conexion.error = (
                    conexion.nodo_destino.error
                    * (conexion.peso / suma_abs_pesos)
                )
                nodo.error += conexion.error

    def __repr__(self) -> str:
        return f"Capa({self.tipo.value}, {len(self.nodos)} nodos)"
