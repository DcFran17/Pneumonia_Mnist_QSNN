# QSNN — Pneumonia Classification from Chest X-Rays Using a Quantum Spiking Neural Network

Este repositorio contiene el notebook principal del proyecto **QSNN** enfocado en la clasificación de neumonía a partir de radiografías de tórax utilizando redes neuronales de impulsos cuánticos (QSNN) basadas en la neurona **Alpha** con decaimiento exponencial.

## Archivos del repositorio

* **Pneumonia_QSNN_Alpha.ipynb** — Notebook principal del experimento. Contiene la arquitectura, configuración, flujo de entrenamiento y evaluación del modelo utilizando el dataset PneumoniaMNIST.

## De qué trata el notebook

El trabajo aborda la clasificación binaria de neumonía (Casos Normales vs. Neumonía) en imágenes de radiografías de tórax de $28 \times 28$ píxeles (*PneumoniaMNIST*). Para ello, se implementa una arquitectura de Red Neuronal de Impulsos Cuántica (*Quantum Spiking Neural Network* - QSNN) basada en la neurona tipo **Alpha**.

La característica central de la neurona Alpha es la modelación del potencial de membrana y la fuga de señal (*leak*) a través de un factor de decaimiento exponencial $e^{-\tau / T_1}$. La red mapea las características de las imágenes de entrada a parámetros de rotación cuántica ($\theta$) y constantes temporales ($\tau$), procesando la información a lo largo de $25$ pasos temporales y utilizando un gradiente sustituto (*surrogate gradient*) para habilitar la retropropagación (*backpropagation*).

Además, el flujo de trabajo compensa el desbalance de clases del dataset mediante una función de pérdida *Cross-Entropy* ponderada y evalúa el rendimiento del modelo a través de múltiples métricas de clasificación.

## Contenido de cada sección del notebook

1. **Imports and configuration** — Carga las librerías necesarias (PyTorch, Kagglehub, Scikit-Learn, Matplotlib, etc.), fija la semilla de reproducibilidad (`SEED = 1524206349`) y define los hiperparámetros del sistema ($784$ entradas, $1000$ neuronas ocultas, $2$ salidas, $25$ pasos temporales, $5$ épocas, *batch size* de $200$).
2. **Alpha Neuron** — Implementa la clase `AlphaSurrogate` (paso *forward* y derivada para *backpropagation* con gradiente sustituto) y la clase `Alpha` (módulo de PyTorch que administra la memoria y el umbral de disparo de la neurona con decaimiento $e^{-\tau / T_1}$).
3. **Loading and preparing PneumoniaMNIST** — Descarga automáticamente el dataset `PneumoniaMNIST` desde Kagglehub, define la clase `PneumoniaDataset` para normalizar los valores de píxel al rango $[0, 1]$ ($3\,882$ imágenes de entrenamiento y $624$ de prueba) y calcula los pesos para corregir el desbalance de clases.
4. **QSNN Architecture** — Define la red `QSNN`, la cual utiliza capas lineales para generar los ángulos de rotación ($\theta$) y tiempos de decaimiento ($\tau$) en dos capas de neuronas Alpha (capa oculta de $1000$ neuronas y capa de salida de $2$ neuronas).
5. **Training and evaluation** — Configura el optimizador Adam y entrena la red acumulando la pérdida durante los $25$ pasos temporales de cada muestra. En la evaluación, la clase predicha se determina contando el número total de *spikes* generados por cada neurona de salida.
6. **Experimental results** — Muestra la tabla de métricas finales en el conjunto de prueba (Exactitud/Accuracy del **84.13%**, F1-Score ponderado de **0.8424**, Precision de **0.8446** y Recall de **0.8413**) y la matriz de confusión correspondiente ($193$ aciertos en Normal y $382$ en Neumonía).
7. **Loss Evolution** — Grafica la curva de evolución de la función de pérdida durante las $5$ épocas de entrenamiento, mostrando una reducción sostenida de aproximadamente $16.81$ a $15.48$.
