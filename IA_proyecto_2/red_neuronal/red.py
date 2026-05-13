from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

from .activaciones import TipoNodo
from .capa import Capa
from .conexion import Conexion

class RedNeuronal:

    def __init__(
        self,
        tasa_aprendizaje: float = 0.1,
        semilla: Optional[int] = None,
    ) -> None:

        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)

        self.tasa_aprendizaje: float       = tasa_aprendizaje
        self.historial_loss:   List[float] = []
        self.historial_acc:    List[float] = []

        # -- Crear las tres capas ----------------
        self.capa_input  = Capa(TipoNodo.INPUT,  784)
        self.capa_hidden = Capa(TipoNodo.HIDDEN,  64)
        self.capa_output = Capa(TipoNodo.OUTPUT,  10)

        # -- Conectar ----------------
        self._conectar(self.capa_input,  self.capa_hidden)
        self._conectar(self.capa_hidden, self.capa_output)

    # Construcción de topología
    def _conectar(self, origen: Capa, destino: Capa) -> None:
        """Crea conexiones fully-connected entre dos capas adyacentes."""
        for nodo_o in origen.nodos:
            for nodo_d in destino.nodos:
                c = Conexion(nodo_o, nodo_d)
                nodo_o.conexiones_salida.append(c)
                nodo_d.conexiones_entrada.append(c)

    # Utilidades internas
    def _set_input(self, vector: np.ndarray) -> None:
        """Carga el vector de 784 valores en los nodos de entrada."""
        for i, nodo in enumerate(self.capa_input.nodos):
            nodo.valor = float(vector[i])
            nodo.z     = float(vector[i])

    def _reset_errores(self) -> None:
        """Reinicia errores de nodos y conexiones antes de cada muestra."""
        for capa in (self.capa_input, self.capa_hidden, self.capa_output):
            for nodo in capa.nodos:
                nodo.error = 0.0
                for c in nodo.conexiones_salida:
                    c.error = 0.0

    # Forward Propagation
    def _forward(self, vector: np.ndarray) -> None:
        """Propagación hacia adelante: INPUT → HIDDEN → OUTPUT."""
        self._set_input(vector)
        self.capa_hidden.forward()
        self.capa_output.forward()

    # Error en capa Output
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

    # Backpropagation — repartir errores hacia atrás
    def _backprop(self) -> None:
        self.capa_hidden.propagar_error()   # usa errores de OUTPUT
        self.capa_input.propagar_error()    # usa errores de HIDDEN
    
    # Gradiente Descendente — actualización de pesos
    def _actualizar_pesos(self) -> None:
        # -- Conexiones HIDDEN -> OUTPUT ------------------------------------
        for nodo_h in self.capa_hidden.nodos:
            for c in nodo_h.conexiones_salida:
                nodo_o = c.nodo_destino
                delta  = (
                    nodo_o.error                           # ej: error del nodo OUTPUT
                    * nodo_o.valor * (1.0 - nodo_o.valor)  # σ'(z) = σ(z)·(1−σ(z))
                    * nodo_h.valor                         # Oi: activación origen
                )
                c.peso += self.tasa_aprendizaje * delta

        # -- Conexiones INPUT -> HIDDEN ------------------------------------
        for nodo_i in self.capa_input.nodos:
            for c in nodo_i.conexiones_salida:
                nodo_h = c.nodo_destino
                delta  = (
                    nodo_h.error                           # ej: error acumulado del nodo HIDDEN
                    * nodo_h.valor * (1.0 - nodo_h.valor)  # σ'(z)
                    * nodo_i.valor                         # Oi: valor del píxel
                )
                c.peso += self.tasa_aprendizaje * delta

    # Modo Debug
    def _debug_iteracion(self, etiqueta: int) -> None:
        sep = "═" * 65
        print(f"\n{sep}")
        print("  MODO DEBUG — PASO A PASO  (1 muestra, antes de actualizar W)")
        print(f"{sep}")

        # -- CAPA INPUT: nodo central de la imagen (índice 392) -------------------
        n_in = self.capa_input.nodos[392]
        print(f"\n▶ [INPUT]  Nodo 392")
        print(f"   valor (píxel normalizado) = {n_in.valor:.4f}")

        # -- CAPA HIDDEN: nodo 0 -------------------------------------------------
        n_h   = self.capa_hidden.nodos[0]
        c_i2h = n_h.conexiones_entrada[392]   # conexión IN[392] → HID[0]

        delta_i2h = (
            n_h.error
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

        # -- CAPA OUTPUT: nodo == etiqueta ----------------------------------------
        n_o   = self.capa_output.nodos[etiqueta]
        c_h2o = n_o.conexiones_entrada[0]     # conexión HID[0] → OUT[etiqueta]

        delta_h2o = (
            n_o.error
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

    # Entrenar
    def entrenar(
        self,
        datos:          List[Tuple[np.ndarray, int]],
        epocas:         int  = 10,
        debug_primera:  bool = False,
        verbose:        bool = True,
    ) -> List[float]:
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

                # Accuracy acumulada
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

    # Entrenar rápido con numpy
    def entrenar_numpy(
        self,
        datos:   List[Tuple[np.ndarray, int]],
        epocas:  int  = 10,
        verbose: bool = True,
    ) -> List[float]:

        def _sig(z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

        # -- Extraer pesos a matrices numpy ------------------------------------
        W_ih = np.array(
            [[c.peso for c in n.conexiones_salida] for n in self.capa_input.nodos],
            dtype=np.float64,
        )   # (784, 64)
        W_ho = np.array(
            [[c.peso for c in n.conexiones_salida] for n in self.capa_hidden.nodos],
            dtype=np.float64,
        )   # (64, 10)

        y_cache = np.eye(10, dtype=np.float64)  # one-hot lookup

        # -- Bucle de entrenamiento ---------------------------------------------
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

                # Actualización HID -> OUT
                grad_o  = e2 * a2 * (1.0 - a2)          # (10,)  δ_output

                # Error en hidden (fórmula del profesor)
                col_abs = np.sum(np.abs(W_ho), axis=0) + 1e-12  # (10,)
                e1      = (W_ho / col_abs) @ e2          # (64,)

                W_ho   += self.tasa_aprendizaje * np.outer(a1, grad_o)

                # Actualización IN -> HID
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

        # -- Sincronizar pesos de vuelta a los objetos Conexion -----------------
        for i, nodo in enumerate(self.capa_input.nodos):
            for j, c in enumerate(nodo.conexiones_salida):
                c.peso = float(W_ih[i, j])

        for i, nodo in enumerate(self.capa_hidden.nodos):
            for j, c in enumerate(nodo.conexiones_salida):
                c.peso = float(W_ho[i, j])

        return self.historial_loss

    # Predecir
    def predecir(self, vector_784: np.ndarray) -> int:
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

    # Persistencia de pesos
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
