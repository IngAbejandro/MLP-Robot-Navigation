# Navegación Reactiva de un Robot Móvil mediante MLP (Edge AI) 🤖

El modelo actúa como un controlador inteligente que procesa señales ultrasónicas para determinar la acción cinemática óptima, diseñado bajo restricciones computacionales para su viabilidad en sistemas embebidos (Edge AI).

##Descripción del Dataset
Se utilizó el **Wall-Following Robot Navigation Data Set** (UCI Machine Learning Repository).
* **Robot:** SCITOS G5.
* **Sensores:** 24 sensores de ultrasonido dispuestos circularmente en el chasis.
* **Muestras:** 5,456 lecturas.
* **Clases (Acciones de Control):** `Move-Forward`, `Slight-Right-Turn`, `Sharp-Right-Turn`, `Slight-Left-Turn`.

##Fases del Proyecto

### 1. Análisis Exploratorio de Datos (EDA) y Acondicionamiento
* **Auditoría de Señales:** Se identificó la reflexión acústica especular (outliers) típica de los sensores ultrasónicos mediante diagramas de caja (Boxplots).
* **Desbalance de Clases:** Se detectó una tendencia del 40% hacia la clase `Move-Forward`, propia de un comportamiento de seguimiento de pared.
* **Acondicionamiento:** Se aplicó estandarización Z-score ($z = \frac{x - \mu}{\sigma}$) para garantizar la estabilidad numérica durante el cálculo de gradientes.

### 2. Arquitectura Computacional (PyTorch)
Se diseñó una arquitectura de red neuronal profunda optimizada para baja latencia:
* **Capa de Entrada:** 24 neuronas (Características de los sensores).
* **Capas Ocultas:** 64 y 32 neuronas con activación `ReLU` (Filtro de ruido y extracción de características no lineales).
* **Capa de Salida:** 4 neuronas con activación `Softmax` implícita en la función de pérdida `CrossEntropyLoss`.

### 3. Entrenamiento y Sintonización
* **Optimizador:** Adam (Learning Rate: 0.001).
* **Batch Size:** 64.
* **Resultados:** Se alcanzó una precisión de validación superior al **91%**. El análisis de las curvas de pérdida demuestra la necesidad de aplicar *Early Stopping* alrededor de la época 40 para evitar el sobreajuste (*overfitting*) al ruido acústico del entorno de entrenamiento.

### 4. Análisis de Viabilidad para Edge AI
* **Fronteras de Decisión:** La Matriz de Confusión demuestra que el modelo es robusto, presentando dudas únicamente en las transiciones físicas naturales (ej. avanzar recto vs. girar levemente a la derecha al acercarse gradualmente a un obstáculo).
* **Carga Computacional:** El modelo cuenta con **3,812 parámetros entrenables**.
* **Conclusión de Hardware:** Con un peso aproximado de 15 KB, el controlador neuronal es altamente viable para ser desplegado en la memoria SRAM de microcontroladores de bajo costo (ej. ESP32, STM32) para inferencia en tiempo real sin conexión a la nube.


##Requisitos para ejecutar
\`\`\`bash
pip install torch pandas matplotlib seaborn scikit-learn
\`\`\`
