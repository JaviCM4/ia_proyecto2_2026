"""
app.py — Interfaz web Flask para la Red Neuronal de reconocimiento de dígitos
==============================================================================
Ejecutar : python app.py
Acceder  : http://localhost:5000

Rutas
-----
GET  /                      → página principal
GET  /api/status            → estado actual de la red
POST /api/train/start       → inicia entrenamiento en hilo separado
GET  /api/train/stream      → SSE: progreso de entrenamiento en tiempo real
POST /api/predict           → predice dígito desde imagen base64 (webcam)
GET  /api/network           → pesos y activaciones para visualización
"""

from __future__ import annotations

import base64
import datetime
import json
import os
import queue
import threading
import traceback
from typing import List, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from neural_network import RedNeuronal, cargar_mnist

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Bitácora de matrices de pesos
# ─────────────────────────────────────────────────────────────────────────────

_ARCHIVO_BITACORA = "bitacora_matrices.log"
_EPOCAS_BITACORA  = {1, 50, 100}

def _guardar_bitacora_matrices(red: RedNeuronal, epoca: int, total_epocas: int) -> None:
    sep  = "=" * 72
    sep2 = "-" * 72
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    W_ho = [[c.peso for c in n.conexiones_salida] for n in red.capa_hidden.nodos]
    W_ih_top10 = [[c.peso for c in red.capa_input.nodos[i].conexiones_salida] for i in range(10)]

    loss_actual = red.historial_loss[-1] if red.historial_loss else float('nan')
    acc_actual  = red.historial_acc[-1]  if red.historial_acc  else float('nan')

    mode = "a" if epoca > 1 else "w"
    with open(_ARCHIVO_BITACORA, mode, encoding="utf-8") as f:
        f.write(f"\n{sep}\n")
        f.write(f"  EPOCA {epoca:>3} / {total_epocas}  |  {ts}\n")
        f.write(f"  Loss: {loss_actual:.8f}   Acc train: {acc_actual:.2f}%\n")
        f.write(f"{sep}\n\n")

        f.write("MATRIZ HIDDEN->OUTPUT  (64 filas x 10 columnas)\n")
        f.write("Fila = nodo hidden j  |  Columna = nodo output k (digito 0-9)\n")
        f.write(sep2 + "\n")
        f.write("j\t" + "\t".join(f"k={k}" for k in range(10)) + "\n")
        for j, fila in enumerate(W_ho):
            f.write(f"{j}\t" + "\t".join(f"{w:.8f}" for w in fila) + "\n")

        f.write("\n")
        f.write("MATRIZ INPUT->HIDDEN  (primeras 10 filas: pixels i=0..9  x 64 columnas)\n")
        f.write("Fila = pixel i  |  Columna = nodo hidden j (0-63)\n")
        f.write("NOTA: la matriz completa tiene 784 x 64 = 50 176 entradas.\n")
        f.write("      Solo se registran las primeras 10 filas como evidencia de cambio.\n")
        f.write(sep2 + "\n")
        f.write("i\t" + "\t".join(f"j={j}" for j in range(64)) + "\n")
        for i, fila in enumerate(W_ih_top10):
            f.write(f"{i}\t" + "\t".join(f"{w:.8f}" for w in fila) + "\n")

        f.write("\n")
    print(f"  [Bitacora] Epoca {epoca} -> {_ARCHIVO_BITACORA}")

# ─────────────────────────────────────────────────────────────────────────────
#  Estado global  (thread-safe mediante lock)
# ─────────────────────────────────────────────────────────────────────────────

class _State:
    def __init__(self) -> None:
        self.lock             = threading.Lock()
        self.red              = RedNeuronal(tasa_aprendizaje=0.1, semilla=42)
        self.mnist            : Optional[tuple]      = None
        self.training_active  : bool                 = False
        self.sse_clients      : List[queue.Queue]    = []
        self.last_activations : dict                 = {}
        self.last_trace       : dict                 = {}
        self.pesos_path       = "pesos"
        self.edu_computed     : bool                 = False

S = _State()

# Cargar pesos si ya existen
if os.path.exists(S.pesos_path + ".npz"):
    S.red.cargar_pesos(S.pesos_path)
    print(f"[App] Pesos cargados desde {S.pesos_path}.npz")
else:
    print("[App] Sin pesos previos — usa el panel de entrenamiento.")


# ─────────────────────────────────────────────────────────────────────────────
#  Broadcast SSE
# ─────────────────────────────────────────────────────────────────────────────

def _broadcast(data: dict) -> None:
    """Envía un evento JSON a todos los clientes SSE conectados."""
    msg = json.dumps(data)
    for q in list(S.sse_clients):
        try:
            q.put_nowait(msg)
        except queue.Full:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  Worker de entrenamiento  (hilo daemon)
# ─────────────────────────────────────────────────────────────────────────────

def _training_worker(epocas: int, tasa: float) -> None:
    try:
        # Nueva red desde cero con la tasa solicitada
        with S.lock:
            S.red = RedNeuronal(tasa_aprendizaje=tasa, semilla=42)

        # Cargar MNIST una sola vez (queda cacheado en S.mnist)
        if S.mnist is None:
            _broadcast({"type": "status", "msg": "Descargando/cargando MNIST..."})
            S.mnist = cargar_mnist()

        train, test = S.mnist
        _broadcast({
            "type": "status",
            "msg": f"Entrenando {epocas} épocas · {len(train)} muestras",
        })

        for epoca in range(1, epocas + 1):
            with S.lock:
                S.red.entrenar_numpy(train, epocas=1, verbose=False)
                loss = S.red.historial_loss[-1]
                acc  = S.red.historial_acc[-1]
                # Pasar muestra fija (test[0]) por forward para que el canvas
                # muestre cómo la red procesa SIEMPRE el mismo dígito con los
                # pesos actuales — hace la animación visualmente significativa.
                display_vec, _ = test[0]
                S.red._forward(display_vec)
                S.last_activations = {
                    "input" : [round(float(v), 4) for v in display_vec],
                    "hidden": [round(float(n.valor), 4) for n in S.red.capa_hidden.nodos],
                    "output": [round(float(n.valor), 4) for n in S.red.capa_output.nodos],
                }

            _broadcast({
                "type"    : "epoch",
                "epoca"   : epoca,
                "total"   : epocas,
                "loss"    : round(loss, 6),
                "accuracy": round(acc, 2),
            })

            # Bitácora en épocas clave
            if epoca in _EPOCAS_BITACORA:
                with S.lock:
                    _guardar_bitacora_matrices(S.red, epoca, epocas)

        # Evaluar en test y guardar pesos
        with S.lock:
            correctos = sum(1 for v, e in test if S.red.predecir(v) == e)
            acc_test  = round(correctos / len(test) * 100, 2)
            S.red.guardar_pesos(S.pesos_path)

        _broadcast({"type": "done", "acc_test": acc_test})

    except Exception as exc:
        _broadcast({"type": "error", "msg": str(exc), "trace": traceback.format_exc()})
    finally:
        with S.lock:
            S.training_active = False


# ─────────────────────────────────────────────────────────────────────────────
#  Preprocesado de cámara
# ─────────────────────────────────────────────────────────────────────────────

def procesar_frame(frame):
    # 1. Escala de grises
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Blur para reducir ruido antes de binarizar
    gris = cv2.GaussianBlur(gris, (5, 5), 0)

    # 3. Threshold de Otsu (encuentra el umbral óptimo automáticamente)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Encontrar contorno principal
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_grandes = [c for c in contornos if cv2.contourArea(c) > 500]

    if contornos_grandes:
        pts = np.vstack(contornos_grandes)
        x, y, w, h = cv2.boundingRect(pts)
        pad = int(max(w, h) * 0.3)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(binaria.shape[1], x + w + pad)
        y2 = min(binaria.shape[0], y + h + pad)
        recorte = binaria[y1:y2, x1:x2]
    else:
        recorte = binaria

    # Centrar usando momentos (igual que MNIST)
    M = cv2.moments(recorte)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        h_r, w_r = recorte.shape
        dx = w_r//2 - cx
        dy = h_r//2 - cy
        M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
        recorte = cv2.warpAffine(recorte, M_translate, (w_r, h_r))

    # 5. Resize a 28x28
    imagen_28 = cv2.resize(recorte, (28, 28), interpolation=cv2.INTER_AREA)

    # Adelgazar trazo para parecerse a MNIST
    kernel_erode = np.ones((2, 2), np.uint8)
    imagen_28 = cv2.erode(imagen_28, kernel_erode, iterations=1)

    # 6. Normalizar
    vector = (imagen_28.flatten() / 255.0).astype(np.float32)
    print(f"[procesar_frame] Píxeles activos >0.5: {(vector > 0.5).sum()}")
    return vector


# ─────────────────────────────────────────────────────────────────────────────
#  Rutas
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    pesos_exist = os.path.exists(S.pesos_path + ".npz")
    return render_template(
        "index.html",
        pesos_exist=pesos_exist,
        epocas_entrenadas=len(S.red.historial_loss),
        last_loss=round(S.red.historial_loss[-1], 6) if S.red.historial_loss else None,
        last_acc=round(S.red.historial_acc[-1], 2)  if S.red.historial_acc  else None,
    )


@app.route("/api/status")
def api_status():
    with S.lock:
        return jsonify({
            "training_active"  : S.training_active,
            "epocas_entrenadas": len(S.red.historial_loss),
            "pesos_exist"      : os.path.exists(S.pesos_path + ".npz"),
            "last_loss"        : round(S.red.historial_loss[-1], 6) if S.red.historial_loss else None,
            "last_acc"         : round(S.red.historial_acc[-1],  2) if S.red.historial_acc  else None,
        })


@app.route("/api/train/start", methods=["POST"])
def train_start():
    with S.lock:
        if S.training_active:
            return jsonify({"error": "Ya hay un entrenamiento en curso"}), 400
        S.training_active = True

    data   = request.get_json(force=True) or {}
    epocas = max(1, int(data.get("epocas", 10)))
    tasa   = float(data.get("tasa", 0.1))

    t = threading.Thread(
        target=_training_worker,
        args=(epocas, tasa),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "epocas": epocas, "tasa": tasa})


@app.route("/api/train/stream")
def train_stream():
    """SSE: emite eventos de progreso mientras dura el entrenamiento."""
    q = queue.Queue(maxsize=500)
    S.sse_clients.append(q)

    def generate():
        try:
            while True:
                try:
                    msg = q.get(timeout=25)
                    yield f"data: {msg}\n\n"
                    d = json.loads(msg)
                    if d.get("type") in ("done", "error"):
                        break
                except queue.Empty:
                    yield ": heartbeat\n\n"   # mantener conexión viva
        finally:
            try:
                S.sse_clients.remove(q)
            except ValueError:
                pass

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control"    : "no-cache",
            "X-Accel-Buffering": "no",
            "Connection"       : "keep-alive",
        },
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Recibe imagen base64 de la webcam, la procesa y retorna la predicción."""
    data    = request.get_json(force=True) or {}
    img_b64 = data.get("image", "")

    if not img_b64:
        return jsonify({"error": "No se recibió imagen"}), 400

    # Decodificar base64 → numpy → OpenCV
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    try:
        img_bytes = base64.b64decode(img_b64)
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("imdecode devolvió None")
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return jsonify({"error": f"Error al decodificar imagen: {e}"}), 400

    # ── Preprocesado usando procesar_frame() (guarda debug_1..4 en cada llamada)
    vector = procesar_frame(img_bgr)

    # Preview: leer debug_4_28x28.png ya guardado por procesar_frame()
    img_28_preview = cv2.imread("debug_4_28x28.png", cv2.IMREAD_GRAYSCALE)
    if img_28_preview is None:
        img_28_preview = np.zeros((28, 28), dtype=np.uint8)

    with S.lock:
        digito, activaciones = S.red.predecir_con_confianza(vector)

        # Calcular errores para traza (sin actualizar pesos)
        loss_trace = S.red._calcular_error_output(digito)
        S.red._backprop()

        hidden_act = [round(float(n.valor), 4) for n in S.red.capa_hidden.nodos]

        hidden_trace = [
            {
                "j"      : j,
                "z"      : round(float(n.z),     6),
                "sigma_z": round(float(n.valor),  6),
                "error"  : round(float(n.error),  6),
            }
            for j, n in enumerate(S.red.capa_hidden.nodos)
        ]
        output_trace = [
            {
                "k"       : k,
                "z"       : round(float(n.z),    6),
                "sigma_z" : round(float(n.valor), 6),
                "esperado": 1.0 if k == digito else 0.0,
                "error"   : round(float(n.error), 6),
            }
            for k, n in enumerate(S.red.capa_output.nodos)
        ]

        S.last_activations = {
            "input" : vector.tolist(),
            "hidden": hidden_act,
            "output": [round(float(a), 4) for a in activaciones],
        }
        S.last_trace = {
            "digito": int(digito),
            "inputs": [round(float(v), 6) for v in vector],
            "hidden": hidden_trace,
            "output": output_trace,
            "loss"  : round(float(loss_trace), 8),
        }

    # Preview 28×28 como PNG base64
    _, buf  = cv2.imencode(".png", img_28_preview)
    preview_b64 = base64.b64encode(buf.tobytes()).decode()

    print(f"[DEBUG] Shape: {vector.shape}")
    print(f"[DEBUG] Min: {vector.min():.4f}  Max: {vector.max():.4f}")
    print(f"[DEBUG] Píxeles activos >0.5: {(vector > 0.5).sum()}")
    print(f"[DEBUG] Píxeles activos >0.1: {(vector > 0.1).sum()}")
    print(f"[DEBUG] Predicción: {digito}")
    print(f"[DEBUG] Activaciones output: {[round(float(a),3) for a in activaciones]}")

    return jsonify({
        "digito"      : int(digito),
        "activaciones": [round(float(a), 4) for a in activaciones],
        "preview"     : "data:image/png;base64," + preview_b64,
    })


@app.route("/api/network")
def api_network():
    """Retorna pesos y activaciones muestreados para dibujar la red."""
    N_IN  = 28   # nodos input representativos
    N_HID = 32   # nodos hidden representativos

    in_idx  = np.linspace(0, 783, N_IN,  dtype=int).tolist()
    hid_idx = np.linspace(0,  63, N_HID, dtype=int).tolist()

    with S.lock:
        # Pesos INPUT→HIDDEN (muestra N_IN × N_HID)
        W_ih_s = [
            [float(S.red.capa_input.nodos[i].conexiones_salida[j].peso)
             for j in hid_idx]
            for i in in_idx
        ]
        # Pesos HIDDEN→OUTPUT (muestra N_HID × 10)
        W_ho_s = [
            [float(S.red.capa_hidden.nodos[j].conexiones_salida[k].peso)
             for k in range(10)]
            for j in hid_idx
        ]

    act = S.last_activations
    input_act  = [float(act["input"][i])  for i in in_idx]  if "input"  in act else [0.0] * N_IN
    hidden_act = [float(act["hidden"][j]) for j in hid_idx] if "hidden" in act else [0.0] * N_HID
    output_act = [float(v) for v in act.get("output", [0.0] * 10)]

    return jsonify({
        "in_idx"    : in_idx,
        "hid_idx"   : hid_idx,
        "W_ih"      : W_ih_s,
        "W_ho"      : W_ho_s,
        "input_act" : input_act,
        "hidden_act": hidden_act,
        "output_act": output_act,
    })


@app.route("/api/debug7")
def api_debug7():
    """
    Carga el primer dígito '7' del set de prueba de MNIST,
    lo pasa por la red (sin modificar pesos) y devuelve la traza completa.
    """
    # Cargar MNIST si no está en caché
    if S.mnist is None:
        try:
            S.mnist = cargar_mnist()
        except Exception as exc:
            return jsonify({"error": f"No se pudo cargar MNIST: {exc}"}), 500

    _, test = S.mnist

    # Buscar el primer ejemplo con etiqueta == 7
    sample_vec = None
    for vec, label in test:
        if label == 7:
            sample_vec = vec
            break

    if sample_vec is None:
        return jsonify({"error": "No se encontró ningún dígito 7 en MNIST test"}), 404

    with S.lock:
        S.red._reset_errores()
        S.red._forward(sample_vec)
        salidas = [n.valor for n in S.red.capa_output.nodos]
        digito  = int(np.argmax(salidas))

        loss_trace = S.red._calcular_error_output(7)   # objetivo siempre es 7
        S.red._backprop()

        hidden_trace = [
            {
                "j"      : j,
                "z"      : round(float(n.z),    6),
                "sigma_z": round(float(n.valor), 6),
                "error"  : round(float(n.error), 6),
            }
            for j, n in enumerate(S.red.capa_hidden.nodos)
        ]
        output_trace = [
            {
                "k"       : k,
                "z"       : round(float(n.z),    6),
                "sigma_z" : round(float(n.valor), 6),
                "esperado": 1.0 if k == 7 else 0.0,
                "error"   : round(float(n.error), 6),
            }
            for k, n in enumerate(S.red.capa_output.nodos)
        ]

        # Guardar en last_trace para que /api/export_trace también lo use
        S.last_trace = {
            "digito": digito,
            "inputs": [round(float(v), 6) for v in sample_vec],
            "hidden": hidden_trace,
            "output": output_trace,
            "loss"  : round(float(loss_trace), 8),
        }

    return jsonify(S.last_trace)


@app.route("/api/trace")
def api_trace():
    """Retorna la traza completa del último forward pass (inputs, hidden, output, loss)."""
    with S.lock:
        if not S.last_trace:
            return jsonify({"error": "Sin predicción disponible. Captura una imagen primero."}), 404
        return jsonify(S.last_trace)


@app.route("/api/export_trace")
def api_export_trace():
    """Genera un TXT tabulado con toda la traza para importar en Excel."""
    with S.lock:
        if not S.last_trace:
            return jsonify({"error": "Sin traza disponible. Realiza una predicción primero."}), 404
        trace    = dict(S.last_trace)
        pesos_ih = [
            (i, j, float(c.peso))
            for i, nodo in enumerate(S.red.capa_input.nodos)
            for j, c     in enumerate(nodo.conexiones_salida)
        ]
        pesos_ho = [
            (j, k, float(c.peso))
            for j, nodo in enumerate(S.red.capa_hidden.nodos)
            for k, c     in enumerate(nodo.conexiones_salida)
        ]

    lines = [
        "TRAZA RED NEURONAL — PROPAGACIÓN HACIA ADELANTE",
        f"Dígito predicho : {trace['digito']}",
        f"Loss total (½Σe²): {trace['loss']:.8f}",
        "Arquitectura    : 784 → 64 → 10  |  Activación: Sigmoid  |  Optimizador: SGD",
        "=" * 70,
        "",
        "SECCIÓN 1: ENTRADAS (784 píxeles normalizados  [0.0 – 1.0])",
        "Pixel_idx\tFila\tColumna\tValor_normalizado",
    ]
    for idx, val in enumerate(trace["inputs"]):
        lines.append(f"{idx}\t{idx // 28}\t{idx % 28}\t{val:.6f}")

    lines += [
        "",
        f"SECCIÓN 2: PESOS INPUT→HIDDEN  ({len(pesos_ih)} conexiones  784×64)",
        "i_input\tj_hidden\tPeso",
    ]
    for i, j, w in pesos_ih:
        lines.append(f"{i}\t{j}\t{w:.8f}")

    lines += [
        "",
        "SECCIÓN 3: CAPA HIDDEN (64 nodos)   z = Σ(x_i × w_ij)   σ(z) = 1 / (1 + e^−z)",
        "j\tz_suma_ponderada\tsigma_z_activacion\terror_backprop",
    ]
    for n in trace["hidden"]:
        lines.append(f"{n['j']}\t{n['z']:.6f}\t{n['sigma_z']:.6f}\t{n['error']:.6f}")

    lines += [
        "",
        f"SECCIÓN 4: PESOS HIDDEN→OUTPUT  ({len(pesos_ho)} conexiones  64×10)",
        "j_hidden\tk_output\tPeso",
    ]
    for j, k, w in pesos_ho:
        lines.append(f"{j}\t{k}\t{w:.8f}")

    lines += [
        "",
        "SECCIÓN 5: CAPA OUTPUT (10 nodos)   dígitos 0-9",
        "k_digito\tz_suma_ponderada\tsigma_z_activacion\tesperado\terror(esp−obt)\t0.5×error²",
    ]
    for n in trace["output"]:
        half_e2 = 0.5 * n["error"] * n["error"]
        lines.append(
            f"{n['k']}\t{n['z']:.6f}\t{n['sigma_z']:.6f}"
            f"\t{n['esperado']:.1f}\t{n['error']:.6f}\t{half_e2:.8f}"
        )

    lines += [
        "",
        f"PÉRDIDA TOTAL = ½ × Σ(esperado − σ(z))² = {trace['loss']:.8f}",
        "(Constantes: no hay sesgos/bias en esta implementación)",
    ]

    return Response(
        "\n".join(lines),
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=traza_red_neuronal.txt"},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Modo Educativo  (paso a paso)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/edu/sample")
def api_edu_sample():
    """
    Modo educativo: carga una muestra de MNIST, ejecuta forward+backprop
    sobre la red actual (sin actualizar pesos) y retorna todos los valores
    intermedios para la vista paso a paso.
    """
    import random as _rnd
    digit_req = request.args.get("digito", type=int, default=None)

    if S.mnist is None:
        try:
            S.mnist = cargar_mnist()
        except Exception as exc:
            return jsonify({"error": f"No se pudo cargar MNIST: {exc}"}), 500

    _, test = S.mnist

    if digit_req is not None and 0 <= digit_req <= 9:
        candidates = [v for v, l in test if l == digit_req]
        if not candidates:
            return jsonify({"error": f"Dígito {digit_req} no encontrado en test"}), 404
        sample_vec = candidates[0]
        label = digit_req
    else:
        sample_vec, label = _rnd.choice(test)

    with S.lock:
        S.red._reset_errores()
        S.red._forward(sample_vec)
        loss = S.red._calcular_error_output(label)
        S.red._backprop()
        S.edu_computed = True

        hidden_data = [
            {
                "j"          : j,
                "z"          : round(float(n.z),                    6),
                "sigma_z"    : round(float(n.valor),                 6),
                "sigma_prime": round(float(n.valor * (1 - n.valor)), 6),
                "error"      : round(float(n.error),                 6),
            }
            for j, n in enumerate(S.red.capa_hidden.nodos)
        ]
        output_data = [
            {
                "k"          : k,
                "z"          : round(float(n.z),                    6),
                "sigma_z"    : round(float(n.valor),                 6),
                "sigma_prime": round(float(n.valor * (1 - n.valor)), 6),
                "esperado"   : 1.0 if k == label else 0.0,
                "error"      : round(float(n.error),                 6),
                "half_e2"    : round(0.5 * float(n.error) ** 2,      8),
            }
            for k, n in enumerate(S.red.capa_output.nodos)
        ]

        # Actualizar last_activations para que el canvas de red refleje esta muestra
        S.last_activations = {
            "input" : [round(float(v), 6) for v in sample_vec],
            "hidden": [round(float(n.valor), 4) for n in S.red.capa_hidden.nodos],
            "output": [round(float(n.valor), 4) for n in S.red.capa_output.nodos],
        }

    return jsonify({
        "label" : int(label),
        "inputs": [round(float(v), 6) for v in sample_vec],
        "hidden": hidden_data,
        "output": output_data,
        "loss"  : round(float(loss), 8),
        "tasa"  : S.red.tasa_aprendizaje,
    })


@app.route("/api/edu/apply", methods=["POST"])
def api_edu_apply():
    """Aplica la actualización de pesos pendiente del modo educativo (un paso SGD)."""
    with S.lock:
        if not S.edu_computed:
            return jsonify({"error": "No hay cálculo pendiente. Llama /api/edu/sample primero."}), 400
        S.red._actualizar_pesos()
        S.edu_computed = False
        ep = len(S.red.historial_loss)
    return jsonify({"ok": True, "epocas_entrenadas": ep})


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[App] Servidor Flask → http://localhost:5000")
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
