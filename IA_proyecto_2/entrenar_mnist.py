"""
entrenar_mnist.py — Entrenamiento completo de la RedNeuronal con MNIST
======================================================================
- Descarga MNIST automáticamente (solo urllib + gzip + numpy)
- Normaliza las imágenes /255.0
- Entrena 10 épocas usando el método vectorizado numpy (fast path)
- Imprime época / loss / accuracy / tiempo por época
- Guarda los pesos en pesos.npz al terminar
- Evalúa accuracy final en el conjunto de prueba
"""

import time
import datetime
import numpy as np
from neural_network import RedNeuronal, cargar_mnist

# ── Configuración ─────────────────────────────────────────────────────────────
TASA_APRENDIZAJE = 0.01
EPOCAS           = 100
ARCHIVO_PESOS    = "pesos"        # se guarda como pesos.npz
SEMILLA          = 42
ARCHIVO_BITACORA = "bitacora_matrices.log"
EPOCAS_BITACORA  = {1, 50, 100}   # épocas en las que se guarda la bitácora


# ── Función de bitácora de matrices ──────────────────────────────────────────
def guardar_bitacora_matrices(red: RedNeuronal, epoca: int) -> None:
    """
    Guarda en bitacora_matrices.log:
      1. Matriz HIDDEN->OUTPUT completa  (64 x 10 = 640 pesos)
      2. Primeras 10 filas de INPUT->HIDDEN  (pixeles 0-9 x 64 columnas)
    Se llama exactamente en las epocas indicadas en EPOCAS_BITACORA.
    """
    sep  = "=" * 72
    sep2 = "-" * 72
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # HIDDEN->OUTPUT: 64 nodos hidden, cada uno con 10 conexiones de salida
    W_ho = [
        [c.peso for c in nodo.conexiones_salida]
        for nodo in red.capa_hidden.nodos
    ]   # lista 64 x 10

    # INPUT->HIDDEN: solo las primeras 10 filas (pixels i=0..9), 64 columnas
    W_ih_top10 = [
        [c.peso for c in red.capa_input.nodos[i].conexiones_salida]
        for i in range(10)
    ]   # lista 10 x 64

    loss_actual = red.historial_loss[-1] if red.historial_loss else float('nan')
    acc_actual  = red.historial_acc[-1]  if red.historial_acc  else float('nan')

    mode = "a" if epoca > 1 else "w"   # sobreescribir en epoca 1, agregar en las demas
    with open(ARCHIVO_BITACORA, mode, encoding="utf-8") as f:

        f.write(f"\n{sep}\n")
        f.write(f"  EPOCA {epoca:>3} / {EPOCAS}  |  {ts}\n")
        f.write(f"  Loss: {loss_actual:.8f}   Acc train: {acc_actual:.2f}%\n")
        f.write(f"{sep}\n\n")

        # Matriz HIDDEN->OUTPUT (64 x 10)
        f.write("MATRIZ HIDDEN->OUTPUT  (64 filas x 10 columnas)\n")
        f.write("Fila = nodo hidden j  |  Columna = nodo output k (digito 0-9)\n")
        f.write(sep2 + "\n")
        header = "j\t" + "\t".join(f"k={k}" for k in range(10))
        f.write(header + "\n")
        for j, fila in enumerate(W_ho):
            vals = "\t".join(f"{w:.8f}" for w in fila)
            f.write(f"{j}\t{vals}\n")

        f.write("\n")

        # INPUT->HIDDEN: primeras 10 filas (pixels 0-9 x 64 columnas)
        f.write("MATRIZ INPUT->HIDDEN  (primeras 10 filas: pixels i=0..9  x 64 columnas)\n")
        f.write("Fila = pixel i  |  Columna = nodo hidden j (0-63)\n")
        f.write("NOTA: la matriz completa tiene 784 x 64 = 50 176 entradas.\n")
        f.write("      Solo se registran las primeras 10 filas como evidencia de cambio.\n")
        f.write(sep2 + "\n")
        header2 = "i\t" + "\t".join(f"j={j}" for j in range(64))
        f.write(header2 + "\n")
        for i, fila in enumerate(W_ih_top10):
            vals = "\t".join(f"{w:.8f}" for w in fila)
            f.write(f"{i}\t{vals}\n")

        f.write("\n")

    print(f"  [Bitacora] Epoca {epoca} guardada -> {ARCHIVO_BITACORA}")

# ── Cargar dataset ────────────────────────────────────────────────────────────
train, test = cargar_mnist()      # ya normalizado /255.0 por cargar_mnist()

print(f"\nConfiguración:")
print(f"  Tasa de aprendizaje : {TASA_APRENDIZAJE}")
print(f"  Épocas              : {EPOCAS}")
print(f"  Muestras train      : {len(train)}")
print(f"  Muestras test       : {len(test)}")
print()

# ── Instanciar red ────────────────────────────────────────────────────────────
red = RedNeuronal(tasa_aprendizaje=TASA_APRENDIZAJE, semilla=SEMILLA)

# ── Entrenamiento época a época con el fast-path numpy ────────────────────────
# entrenar_numpy() usa operaciones matriciales numpy en vez de iterar sobre
# objetos Conexion, logrando la misma matemática 100-1000× más rápido.
print("=" * 65)
print(f"  {'Época':>6}  {'Loss':>12}  {'Acc train':>10}  {'Tiempo':>8}")
print("=" * 65)

t_global = time.time()

for epoca in range(1, EPOCAS + 1):
    t0 = time.time()
    red.entrenar_numpy(train, epocas=1, verbose=False)
    elapsed = time.time() - t0

    loss_prom = red.historial_loss[-1]
    acc_train = red.historial_acc[-1]
    print(f"  {epoca:>6}/{EPOCAS}  {loss_prom:>12.6f}  {acc_train:>9.2f}%  {elapsed:>7.1f}s")

    if epoca in EPOCAS_BITACORA:
        guardar_bitacora_matrices(red, epoca)

print("=" * 65)
print(f"\nTiempo total de entrenamiento: {time.time() - t_global:.1f}s")

# ── Guardar pesos ─────────────────────────────────────────────────────────────
red.guardar_pesos(ARCHIVO_PESOS)

# ── Evaluación en conjunto de prueba ─────────────────────────────────────────
print("\nEvaluando en conjunto de prueba...")
correctos_test = 0
for vector, etiqueta in test:
    if red.predecir(vector) == etiqueta:
        correctos_test += 1

acc_test = correctos_test / len(test) * 100.0
print(f"\n{'=' * 65}")
print(f"  Accuracy final en TEST ({len(test)} muestras): {acc_test:.2f}%")
print(f"{'=' * 65}")

# ── Desglose por dígito ───────────────────────────────────────────────────────
print("\nAccuracy por dígito:")
por_digito = {i: [0, 0] for i in range(10)}   # [correctos, total]
for vector, etiqueta in test:
    pred = red.predecir(vector)
    por_digito[etiqueta][1] += 1
    if pred == etiqueta:
        por_digito[etiqueta][0] += 1

for digito, (cor, tot) in por_digito.items():
    barra = "█" * int(cor / tot * 30)
    print(f"  Dígito {digito}: {cor:>4}/{tot}  ({cor/tot*100:>5.1f}%)  {barra}")
