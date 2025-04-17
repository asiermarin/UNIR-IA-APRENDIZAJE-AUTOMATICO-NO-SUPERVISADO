# Tema 3 - Diferentes implementaciones de K-Means

---

### 1. Introducción y objetivos

1.1. El algoritmo K-Means es una herramienta esencial para el aprendizaje no supervisado por su **simplicidad y eficacia** en la creación de clústeres.

1.2. Este tema explora varias variantes del algoritmo K-Means:
   - **Lloyd’s**
   - **MacQueen’s**
   - **Hartigan-Wong**
   - **Elkan’s**
   - **Soft K-Means / Fuzzy K-Means**

1.3. Objetivos del tema:
   - Comprender la **teoría y fundamentos matemáticos** de cada variante.
   - Desarrollar **habilidades prácticas** en su implementación.
   - **Comparar eficiencia, precisión y convergencia**.
   - Aplicar estos métodos a **casos reales**: compresión de imágenes, segmentación de clientes y diagnóstico médico.

---

### 2. Lloyd’s K-Means

2.1. Es la versión más conocida y simple del algoritmo K-Means.

2.2. Pasos del algoritmo:
   1. Inicializar aleatoriamente K centroides.
   2. Asignar cada punto al clúster más cercano.
   3. Recalcular los centroides como la media de los puntos asignados.
   4. Repetir hasta que los centroides no cambien significativamente.

2.3. Implementación genérica en Python:
```python
def kmeans(X, K, max_iters=100, tolerance=1e-4):
    centroids = initialize_centroids(X, K)
    for i in range(max_iters):
        cluster_labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, cluster_labels, K)
        if np.all(np.abs(new_centroids - centroids) < tolerance):
            break
        centroids = new_centroids
    return centroids, cluster_labels
```

---

### 3. MacQueen’s K-Means

3.1. Actualiza los centroides **de forma continua**, con cada nueva asignación de punto.

3.2. Pasos del algoritmo:
   - Inicialización aleatoria de K centroides.
   - Para cada nuevo punto:
     - Se asigna al centroide más cercano.
     - Se actualiza el centroide inmediatamente.
   - Repetir hasta que no cambien las asignaciones.

3.3. Diferencias con Lloyd’s:
   - Lloyd’s actualiza centroides **después** de asignar todos los puntos.
   - MacQueen’s lo hace **en tiempo real**, punto por punto.
   - MacQueen’s puede ser más eficiente en **flujos de datos continuos**.

---

### 4. Hartigan-Wong K-Means

4.1. Mejora la inicialización y hace **reasignaciones locales** que pueden mejorar la función objetivo.

4.2. Pasos del algoritmo:
   - Inicialización aleatoria de centroides.
   - Asignación aleatoria de puntos a clústeres.
   - Se optimiza localmente reubicando puntos si mejora la función objetivo.
   - Iterar hasta convergencia.

4.3. Comparación con Lloyd:
   - Hartigan-Wong actualiza más frecuentemente los clústeres.
   - Puede requerir menos iteraciones.
   - Es más complejo de implementar.

---

### 5. Elkan’s K-Means

5.1. Diseñado para ser más eficiente mediante la **reducción de cálculos de distancia**.

5.2. Principios:
   - Usa **límites superiores e inferiores** de distancia para evitar cálculos innecesarios.
   - Se basa en la **desigualdad triangular**.

5.3. Pasos del algoritmo:
   1. Inicializar centroides.
   2. Mantener y actualizar límites de distancia.
   3. Asignar puntos sin calcular todas las distancias.
   4. Recalcular centroides y actualizar límites.
   5. Repetir hasta convergencia.

5.4. Ventajas:
   - Mayor eficiencia computacional.
   - Escalable para conjuntos de datos grandes.

5.5. Limitaciones:
   - Implementación más compleja.
   - Aún es sensible a la inicialización.

5.6. **Ejemplo práctico**: compresión de imágenes a 30 y 40 colores con K-Means estándar y Elkan.

---

### 6. Soft K-Means / Fuzzy K-Means

6.1. Introduce la idea de **pertenencia parcial** a múltiples clústeres.

6.2. En lugar de asignación rígida (hard), se calcula un **porcentaje de membresía**.

6.3. Pasos del algoritmo difuso:
   1. Inicializar K clústeres.
   2. Asignar valores de membresía entre 0 y 1.
   3. Calcular centroides considerando esos valores.
   4. Calcular distancias a los centroides.
   5. Actualizar los valores de membresía.
   6. Repetir hasta que se estabilicen los valores.
   7. **Desfuzzificación**: asignar cada punto al grupo con mayor pertenencia.

6.4. **Ejemplo detallado con el dataset Iris** usando la librería `fcmeans`.

6.5. Aplicaciones:
   - Segmentación de imágenes por color y textura.
   - Segmentación de clientes.
   - Diagnóstico médico.
   - Análisis de tráfico.
   - Evaluación de riesgos.

6.6. Ventajas:
   - Flexibilidad ante datos complejos.
   - Robusto ante valores atípicos.
   - Interpretabilidad de las relaciones.

6.7. Desventajas:
   - Computacionalmente más costoso.
   - Elegir K y la función de membresía puede ser difícil.

---

### 7. Cuaderno de ejercicios

7.1. **Ejercicio 1 – Implementación de Lloyd’s** con `make_blobs`:
   - Crear 300 muestras, 3 centros.
   - Agrupar y graficar con Scikit-learn.

7.2. **Ejercicio 2 – Comparación Lloyd vs. MacQueen**:
   - Lloyd más eficiente en datos estáticos.
   - MacQueen mejor para datos en flujo.

7.3. **Ejercicio 3 – Eficiencia Elkan vs. K-Means clásico**:
   - Comparación de tiempos de ejecución.
   - Elkan es más rápido al reducir distancias.

7.4. **Ejercicio 4 – Aplicación de Soft K-Means en segmentación de imágenes**:
   - Ventaja: suaviza transiciones.
   - Desventaja: mayor coste computacional.

---

### 8. Referencias y recursos adicionales

- Lazyprogrammer.me (2016): Unsupervised Machine Learning in Python.
- Slonim, Aharoni y Crammer (2013): comparación entre Hartigan y Lloyd.
- Wohlenberg (2021): comparativa de tres versiones de K-Means.
- Vídeos recomendados:
   - [Aplicaciones avanzadas de K-Means](https://unir.cloud.panopto.eu/Panopto/Pages/Embed.aspx?id=04202076-ca31-40b9-b608-b1be00da113d)
   - [Fuzzy C-Means explicado](https://www.youtube.com/watch?v=X7co6-U4BJY)

---

### 9. Preguntas tipo test (ejemplos de evaluación)

1. ¿Qué algoritmo actualiza centroides de forma continua?
   - **Respuesta correcta:** B. MacQueen

2. ¿Qué ventaja tiene Elkan sobre K-Means clásico?
   - **Respuesta correcta:** C. Usa límites superiores e inferiores.

3. ¿Qué desventaja tiene Fuzzy K-Means?
   - **Respuesta correcta:** C. Computacionalmente es más costoso.

4. ¿Qué técnica evita cálculos innecesarios en Elkan?
   - **Respuesta correcta:** C. Desigualdad triangular.

---

Si necesitas que desarrolle con más detalle alguno de los ejercicios, implementaciones en código o ejemplos visuales, estaré encantado de ayudarte.