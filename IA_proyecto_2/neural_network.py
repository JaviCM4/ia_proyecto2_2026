"""
Red Neuronal Artificial (MLP) desde cero — Reconocimiento de dígitos 0-9
=========================================================================
Curso  : Inteligencia Artificial 1 — USAC / CUNOC
Librerías permitidas : numpy, math, random
Prohibido            : TensorFlow, PyTorch, Keras, scikit-learn

Arquitectura
------------
  Input  : 784 nodos  (imagen 28×28 normalizada  [0.0 – 1.0])
  Hidden :  64 nodos  (activación Sigmoid)
  Output :  10 nodos  (activación Sigmoid, uno por dígito 0-9)
"""

from __future__ import annotations

import math
import random
from enum import Enum
from typing import List, Tuple, Optional

import numpy as np


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


# ══════════════════════════════════════════════════════════════════════════════
#  RedNeuronal  (MLP 784 → 64 → 10)
# ══════════════════════════════════════════════════════════════════════════════

class RedNeuronal:
    """
    Perceptrón Multicapa (MLP) para reconocimiento de dígitos escritos a mano.

    Arquitectura
    ------------
      Input  : 784 nodos  (imagen 28×28 normalizada)
      Hidden :  64 nodos  (Sigmoid)
      Output :  10 nodos  (Sigmoid, uno por dígito 0-9)

    Parámetros
    ----------
    tasa_aprendizaje : η, por defecto 0.1
    semilla          : semilla aleatoria para reproducibilidad (opcional)
    """

    def __init__(
        self,
        tasa_aprendizaje: float = 0.1,
        semilla: Optional[int] = None,
    ) -> None:

        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)

        self.tasa_aprendizaje: float      = tasa_aprendizaje
        self.historial_loss:   List[float] = []
        self.historial_acc:    List[float] = []

        # ── Crear las tres capas ──────────────────────────────────────────────
        self.capa_input  = Capa(TipoNodo.INPUT,  784)
        self.capa_hidden = Capa(TipoNodo.HIDDEN,  64)
        self.capa_output = Capa(TipoNodo.OUTPUT,  10)

        # ── Conectar fully-connected: INPUT→HIDDEN y HIDDEN→OUTPUT ────────────
        self._conectar(self.capa_input,  self.capa_hidden)
        self._conectar(self.capa_hidden, self.capa_output)

    # ══════════════════════════════════════════════════════════════════════════
    #  Construcción de topología
    # ══════════════════════════════════════════════════════════════════════════

    def _conectar(self, origen: Capa, destino: Capa) -> None:
        """Crea conexiones fully-connected entre dos capas adyacentes."""
        for nodo_o in origen.nodos:
            for nodo_d in destino.nodos:
                c = Conexion(nodo_o, nodo_d)
                nodo_o.conexiones_salida.append(c)
                nodo_d.conexiones_entrada.append(c)

    # ══════════════════════════════════════════════════════════════════════════
    #  Utilidades internas
    # ══════════════════════════════════════════════════════════════════════════

    def _set_input(self, vector: np.ndarray) -> None:
        """Carga el vector de 784 valores en los nodos de entrada."""
        for i, nodo in enumerate(self.capa_input.nodos):
            nodo.valor = float(vector[i])
            nodo.z     = float(vector[i])   # z = valor directo en capa INPUT

    def _reset_errores(self) -> None:
        """Reinicia errores de nodos y conexiones antes de cada muestra."""
        for capa in (self.capa_input, self.capa_hidden, self.capa_output):
            for nodo in capa.nodos:
                nodo.error = 0.0
                for c in nodo.conexiones_salida:
                    c.error = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    #  Forward Propagation
    # ══════════════════════════════════════════════════════════════════════════

    def _forward(self, vector: np.ndarray) -> None:
        """Propagación hacia adelante: INPUT → HIDDEN → OUTPUT."""
        self._set_input(vector)
        self.capa_hidden.forward()
        self.capa_output.forward()

    # ══════════════════════════════════════════════════════════════════════════
    #  Error en capa Output (MSE simplificado)
    # ══════════════════════════════════════════════════════════════════════════

    def _calcular_error_output(self, etiqueta: int) -> float:
        """
        Para cada nodo output:
          esperado   = 1.0 si el índice coincide con la etiqueta, sino 0.0
          error_nodo = esperado − obtenido          ← guardado en nodo.error
          loss      += ½ × (esperado − obtenido)²  ← MSE simplificado

        Retorna el loss total de la muestra.
        """
        loss = 0.0
        for i, nodo in enumerate(self.capa_output.nodos):
            esperado   = 1.0 if i == etiqueta else 0.0
            error      = esperado - nodo.valor
            nodo.error = error
            loss      += 0.5 * error * error
        return loss

    # ══════════════════════════════════════════════════════════════════════════
    #  Backpropagation
    # ══════════════════════════════════════════════════════════════════════════

    def _backprop(self) -> None:
        """
        Paso 1 — capa_hidden.propagar_error():
            Usa los errores de los nodos OUTPUT (ya calculados).
            → Almacena error en cada conexión HID→OUT.
            → Acumula error total en cada nodo HIDDEN.

        Paso 2 — capa_input.propagar_error():
            Usa los errores de los nodos HIDDEN (recién calculados).
            → Almacena error en cada conexión IN→HID.
            → Acumula error total en cada nodo INPUT (informativo).
        """
        self.capa_hidden.propagar_error()   # usa errores de OUTPUT
        self.capa_input.propagar_error()    # usa errores de HIDDEN

    # ══════════════════════════════════════════════════════════════════════════
    #  Gradiente Descendente — actualización de pesos
    # ══════════════════════════════════════════════════════════════════════════

    def _actualizar_pesos(self) -> None:
        """
        Para cada conexión:

          ΔW = η × ej × σ(z_destino) × (1 − σ(z_destino)) × valor_origen
          W  ← W + ΔW

        donde:
          ej      = nodo_destino.error  (error directo del nodo, no distribuido)
          Para HID→OUT : ej = esperado − obtenido       (calculado en output)
          Para IN→HID  : ej = nodo_hidden.error         (acumulado en backprop)

        Usar nodo_destino.error (no c.error) es crítico:
        c.error lleva el error ya multiplicado por (peso/Σ|pesos|),
        lo que divide el gradiente y destruye la señal de aprendizaje.
        """
        # ── Conexiones HIDDEN → OUTPUT ────────────────────────────────────────
        for nodo_h in self.capa_hidden.nodos:
            for c in nodo_h.conexiones_salida:
                nodo_o = c.nodo_destino
                delta  = (
                    nodo_o.error                           # ej: error del nodo OUTPUT
                    * nodo_o.valor * (1.0 - nodo_o.valor)  # σ'(z) = σ(z)·(1−σ(z))
                    * nodo_h.valor                         # Oi: activación origen
                )
                c.peso += self.tasa_aprendizaje * delta

        # ── Conexiones INPUT → HIDDEN ─────────────────────────────────────────
        for nodo_i in self.capa_input.nodos:
            for c in nodo_i.conexiones_salida:
                nodo_h = c.nodo_destino
                delta  = (
                    nodo_h.error                           # ej: error acumulado del nodo HIDDEN
                    * nodo_h.valor * (1.0 - nodo_h.valor)  # σ'(z)
                    * nodo_i.valor                         # Oi: valor del píxel
                )
                c.peso += self.tasa_aprendizaje * delta

    # ══════════════════════════════════════════════════════════════════════════
    #  Modo Debug
    # ══════════════════════════════════════════════════════════════════════════

    def _debug_iteracion(self, etiqueta: int) -> None:
        """
        Imprime z, sigmoid(z), error y delta_peso de un nodo por capa,
        con 4 decimales de precisión.

        Llamar DESPUÉS de _backprop() y ANTES de _actualizar_pesos() para
        observar los valores exactos que se usarán en la actualización.
        """
        sep = "═" * 65
        print(f"\n{sep}")
        print("  MODO DEBUG — PASO A PASO  (1 muestra, antes de actualizar W)")
        print(f"{sep}")

        # ── CAPA INPUT: nodo central de la imagen (índice 392) ────────────────
        n_in = self.capa_input.nodos[392]
        print(f"\n▶ [INPUT]  Nodo 392")
        print(f"   valor (píxel normalizado) = {n_in.valor:.4f}")

        # ── CAPA HIDDEN: nodo 0 ───────────────────────────────────────────────
        # conexiones_entrada de HID[0] están ordenadas por nodo origen (IN[0..783])
        n_h   = self.capa_hidden.nodos[0]
        c_i2h = n_h.conexiones_entrada[392]   # conexión IN[392] → HID[0]

        delta_i2h = (
            n_h.error              # error acumulado del nodo hidden
            * n_h.valor * (1.0 - n_h.valor)
            * c_i2h.nodo_origen.valor
        )
        print(f"\n▶ [HIDDEN] Nodo 0")
        print(f"   z                      = {n_h.z:.4f}")
        print(f"   sigmoid(z)             = {n_h.valor:.4f}")
        print(f"   error acumulado        = {n_h.error:.4f}")
        print(f"   ── Conexión IN[392] → HID[0] ──────────────────────────")
        print(f"      peso               = {c_i2h.peso:.4f}")
        print(f"      error_conexion     = {c_i2h.error:.4f}")
        print(f"      ΔW (pre-update)    = {self.tasa_aprendizaje * delta_i2h:.4f}")

        # ── CAPA OUTPUT: nodo == etiqueta ─────────────────────────────────────
        # conexiones_entrada de OUT[etiqueta] están ordenadas por nodo origen (HID[0..63])
        n_o   = self.capa_output.nodos[etiqueta]
        c_h2o = n_o.conexiones_entrada[0]     # conexión HID[0] → OUT[etiqueta]

        delta_h2o = (
            n_o.error              # error directo del nodo output
            * n_o.valor * (1.0 - n_o.valor)
            * c_h2o.nodo_origen.valor
        )
        print(f"\n▶ [OUTPUT] Nodo {etiqueta}  (dígito esperado)")
        print(f"   z                      = {n_o.z:.4f}")
        print(f"   sigmoid(z)             = {n_o.valor:.4f}")
        print(f"   error (esperado−obt)   = {n_o.error:.4f}")
        print(f"   ── Conexión HID[0] → OUT[{etiqueta}] ─────────────────────")
        print(f"      peso               = {c_h2o.peso:.4f}")
        print(f"      error_conexion     = {c_h2o.error:.4f}")
        print(f"      ΔW (pre-update)    = {self.tasa_aprendizaje * delta_h2o:.4f}")

        print(f"\n{sep}\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  Entrenar
    # ══════════════════════════════════════════════════════════════════════════

    def entrenar(
        self,
        datos:          List[Tuple[np.ndarray, int]],
        epocas:         int  = 10,
        debug_primera:  bool = False,
        verbose:        bool = True,
    ) -> List[float]:
        """
        Entrena la red mediante Gradiente Descendente Estocástico (SGD).

        Parámetros
        ----------
        datos          : lista de (vector_784_float32, etiqueta_int)
                         vector normalizado [0.0, 1.0], etiqueta ∈ {0..9}
        epocas         : número de épocas de entrenamiento
        debug_primera  : si True → imprime debug en la primera muestra de
                         la primera época (antes de actualizar pesos)
        verbose        : si True → muestra loss y accuracy al final de cada época

        Retorna
        -------
        historial_loss : lista con el loss promedio de cada época
        """
        for epoca in range(1, epocas + 1):
            loss_total = 0.0
            correctos  = 0

            for idx, (vector, etiqueta) in enumerate(datos):

                # 0. Reiniciar errores acumulados
                self._reset_errores()

                # 1. Forward propagation
                self._forward(vector)

                # 2. Calcular error en capa output + loss de la muestra
                loss = self._calcular_error_output(etiqueta)
                loss_total += loss

                # 3. Backpropagation (repartir errores)
                self._backprop()

                # 4. Modo debug: solo primera muestra de la primera época
                if debug_primera and epoca == 1 and idx == 0:
                    self._debug_iteracion(etiqueta)

                # 5. Actualizar pesos (gradiente descendente)
                self._actualizar_pesos()

                # Accuracy acumulada (con valores post-forward)
                salidas = [n.valor for n in self.capa_output.nodos]
                if int(np.argmax(salidas)) == etiqueta:
                    correctos += 1

            loss_prom = loss_total / len(datos)
            accuracy  = correctos  / len(datos) * 100.0
            self.historial_loss.append(loss_prom)
            self.historial_acc.append(accuracy)

            if verbose:
                print(
                    f"Época {epoca:>4}/{epocas}  "
                    f"Loss={loss_prom:.6f}  "
                    f"Accuracy={accuracy:.2f}%"
                )

        return self.historial_loss

    # ══════════════════════════════════════════════════════════════════════════
    #  Entrenar rápido (numpy vectorizado) — misma matemática, 100-1000× más veloz
    # ══════════════════════════════════════════════════════════════════════════

    def entrenar_numpy(
        self,
        datos:   List[Tuple[np.ndarray, int]],
        epocas:  int  = 10,
        verbose: bool = True,
    ) -> List[float]:
        """
        Idéntica matemática que entrenar(), pero usa matrices numpy en vez de
        iterar sobre objetos Conexion. Velocidad ~100-1000× mayor.

        Matemática implementada (igual que la versión por objetos):
        ─────────────────────────────────────────────────────────────
        Forward:
          z1 = W_ih.T @ x        (784→64)
          a1 = σ(z1)
          z2 = W_ho.T @ a1       (64→10)
          a2 = σ(z2)

        Error output:
          e2 = y_onehot − a2
          loss += ½ ‖e2‖²

        Actualización HID→OUT:
          ΔW_ho = η × outer(a1, e2 * a2*(1−a2))

        Error hidden (fórmula del profesor: distribuir por fracción de peso):
          col_abs_sum = Σ_h |W_ho[h,o]|   para cada o
          W_norm[h,o] = W_ho[h,o] / col_abs_sum[o]
          e1[h]       = Σ_o W_norm[h,o] × e2[o]   = W_norm @ e2

        Actualización IN→HID:
          ΔW_ih = η × outer(x, e1 * a1*(1−a1))

        Al terminar sincroniza los pesos a los objetos Conexion para que
        predecir(), _debug_iteracion() y guardar_pesos() sigan funcionando.

        Parámetros
        ----------
        datos   : lista de (vector_784_float32, etiqueta_int)
        epocas  : número de épocas
        verbose : imprime loss y accuracy al final de cada época

        Retorna
        -------
        historial_loss acumulado
        """

        def _sig(z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

        # ── Extraer pesos a matrices numpy ────────────────────────────────────
        W_ih = np.array(
            [[c.peso for c in n.conexiones_salida] for n in self.capa_input.nodos],
            dtype=np.float64,
        )   # (784, 64)
        W_ho = np.array(
            [[c.peso for c in n.conexiones_salida] for n in self.capa_hidden.nodos],
            dtype=np.float64,
        )   # (64, 10)

        y_cache = np.eye(10, dtype=np.float64)  # one-hot lookup

        # ── Bucle de entrenamiento ────────────────────────────────────────────
        for epoca in range(1, epocas + 1):
            loss_total = 0.0
            correctos  = 0

            for vector, etiqueta in datos:
                x  = vector.astype(np.float64)          # (784,)
                yv = y_cache[etiqueta]                   # (10,) one-hot

                # Forward
                a1 = _sig(W_ih.T @ x)                   # (64,)
                a2 = _sig(W_ho.T @ a1)                  # (10,)

                # Error en output
                e2 = yv - a2                             # (10,)
                loss_total += 0.5 * float(np.dot(e2, e2))

                # Actualización HID→OUT
                grad_o  = e2 * a2 * (1.0 - a2)          # (10,)  δ_output

                # Error en hidden (fórmula del profesor)
                col_abs = np.sum(np.abs(W_ho), axis=0) + 1e-12  # (10,)
                e1      = (W_ho / col_abs) @ e2          # (64,)
                
                W_ho   += self.tasa_aprendizaje * np.outer(a1, grad_o)

                # Actualización IN→HID
                grad_h  = e1 * a1 * (1.0 - a1)          # (64,)  δ_hidden
                W_ih   += self.tasa_aprendizaje * np.outer(x, grad_h)

                if int(np.argmax(a2)) == etiqueta:
                    correctos += 1

            loss_prom = loss_total / len(datos)
            accuracy  = correctos  / len(datos) * 100.0
            self.historial_loss.append(loss_prom)
            self.historial_acc.append(accuracy)

            if verbose:
                print(
                    f"Época {epoca:>4}/{epocas}  "
                    f"Loss={loss_prom:.6f}  "
                    f"Accuracy={accuracy:.2f}%"
                )

        # ── Sincronizar pesos de vuelta a los objetos Conexion ────────────────
        for i, nodo in enumerate(self.capa_input.nodos):
            for j, c in enumerate(nodo.conexiones_salida):
                c.peso = float(W_ih[i, j])

        for i, nodo in enumerate(self.capa_hidden.nodos):
            for j, c in enumerate(nodo.conexiones_salida):
                c.peso = float(W_ho[i, j])

        return self.historial_loss

    # ══════════════════════════════════════════════════════════════════════════
    #  Predecir
    # ══════════════════════════════════════════════════════════════════════════

    def predecir(self, vector_784: np.ndarray) -> int:
        """
        Realiza inferencia y retorna el dígito con mayor activación.

        Parámetros
        ----------
        vector_784 : ndarray de 784 floats en [0.0, 1.0]

        Retorna
        -------
        int : dígito predicho (0-9)
        """
        self._reset_errores()
        self._forward(vector_784)
        salidas = [n.valor for n in self.capa_output.nodos]
        return int(np.argmax(salidas))

    def predecir_con_confianza(
        self,
        vector_784: np.ndarray,
    ) -> Tuple[int, List[float]]:
        """
        Retorna (dígito_predicho, lista_de_10_activaciones).
        Útil para mostrar la barra de confianza en la interfaz web.
        """
        self._reset_errores()
        self._forward(vector_784)
        salidas = [n.valor for n in self.capa_output.nodos]
        return int(np.argmax(salidas)), salidas

    # ══════════════════════════════════════════════════════════════════════════
    #  Persistencia de pesos
    # ══════════════════════════════════════════════════════════════════════════

    def guardar_pesos(self, ruta: str) -> None:
        """
        Guarda los pesos en formato .npz de numpy.
          pesos_ih : shape (784, 64)   — INPUT → HIDDEN
          pesos_ho : shape ( 64, 10)   — HIDDEN → OUTPUT
        """
        pesos_ih = np.array(
            [[c.peso for c in n.conexiones_salida]
             for n in self.capa_input.nodos],
            dtype=np.float64,
        )
        pesos_ho = np.array(
            [[c.peso for c in n.conexiones_salida]
             for n in self.capa_hidden.nodos],
            dtype=np.float64,
        )
        np.savez(ruta, pesos_ih=pesos_ih, pesos_ho=pesos_ho)
        print(f"[RedNeuronal] Pesos guardados → {ruta}.npz")

    def cargar_pesos(self, ruta: str) -> None:
        """Carga pesos guardados previamente con guardar_pesos()."""
        path  = ruta if ruta.endswith(".npz") else ruta + ".npz"
        datos = np.load(path)
        pesos_ih: np.ndarray = datos["pesos_ih"]   # (784, 64)
        pesos_ho: np.ndarray = datos["pesos_ho"]   # ( 64, 10)

        for i, nodo in enumerate(self.capa_input.nodos):
            for j, c in enumerate(nodo.conexiones_salida):
                c.peso = float(pesos_ih[i, j])

        for i, nodo in enumerate(self.capa_hidden.nodos):
            for j, c in enumerate(nodo.conexiones_salida):
                c.peso = float(pesos_ho[i, j])

        print(f"[RedNeuronal] Pesos cargados ← {path}")

    def __repr__(self) -> str:
        return (
            f"RedNeuronal(784→64→10  "
            f"η={self.tasa_aprendizaje}  "
            f"épocas_entrenadas={len(self.historial_loss)})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Utilidad: carga de MNIST
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

    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mnist_cache")
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


# ══════════════════════════════════════════════════════════════════════════════
#  Punto de entrada — demostración y verificación rápida
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── Inicializando Red Neuronal 784 → 64 → 10 ──────────────────")
    red = RedNeuronal(tasa_aprendizaje=0.1, semilla=42)
    print(red)

    train, test = cargar_mnist()

    # Entrenar 3 épocas sobre las primeras 10 000 muestras
    # debug_primera=True → imprime paso a paso la primera muestra
    red.entrenar(train[:10_000], epocas=3, debug_primera=True)
    red.guardar_pesos("pesos_red")

    # Evaluar sobre 2 000 muestras de prueba
    correctos = sum(
        1 for v, e in test[:2_000]
        if red.predecir(v) == e
    )
    print(f"\nAccuracy en test (2 000 muestras): {correctos / 2000 * 100:.2f}%")
