# Tema 2 - Fundamentos y aplicaciones del agrupamiento de K-Means

---

## 1. Introducción y Objetivos

1.1. El agrupamiento es una técnica central del aprendizaje automático no supervisado, que permite identificar patrones o grupos en los datos sin utilizar etiquetas previas.

1.2. La utilidad del agrupamiento radica en descubrir estructuras ocultas cuando no se conoce la variable objetivo (por ejemplo, tipos de usuarios).

1.3. Los algoritmos de agrupamiento dividen los datos en clústeres o grupos de elementos similares entre sí y diferentes de los otros.

1.4. Este tema se centra en el algoritmo **K-Means**, uno de los más conocidos y utilizados por su simplicidad y efectividad.

1.5. Objetivos concretos:
   - Explicar detalladamente el funcionamiento interno de K-Means.
   - Analizar el proceso de asignación y actualización de centroides.
   - Implementar el algoritmo usando Python.

> **Nota:**
> Como podemos agrupar un conjunto de datos de cliente sin conocer previamente su comportamiento de compras?
> Si existe un comportamiento similar significa que existe unas preferencias similares.
> Cual es el objetivo de las técnicas de clustering (premisa)? -> Agrupamiento de datos no etiquetados y **descubrimiento** de patrones.
> Ejemplos de aplicación? -> Las figuras geométricas son un tipo de agrupación. Agrupaciones de productos (moviles con especificaciones parecidas).
> Como podemos saber si los resultados son buenos si no disponemos de etiquetas? -> La intuición es importante, de manera cuantitativa con fórmulas matemáticas.

MIN 11.

---

## 2. Algoritmo K-Means

2.1. K-Means agrupa los datos en **K clústeres**, donde cada clúster se representa mediante la media de sus puntos (de ahí su nombre: K-Medias).

2.2. Objetivo del algoritmo:
   - Minimizar la **varianza intra-clúster** (dentro de cada grupo).
   - Maximizar la **varianza inter-clúster** (entre grupos diferentes).

2.3. Concepto clave: **Diagrama de Voronoi**
   - Divide el espacio en regiones según la proximidad a puntos llamados sitios.
   - Cada región representa un clúster.
   - Útil para visualizar cómo se agrupan los datos alrededor de los centroides.

2.4. Implementación de Voronoi en Python:
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

points = np.random.rand(5, 2)
vor = Voronoi(points)
voronoi_plot_2d(vor)
plt.plot(points[:,0], points[:,1], 'ro')
plt.show()
```

2.5. Pasos del algoritmo K-Means:
   1. **Preprocesamiento**: limpieza y normalización de los datos.
   2. **Inicialización**: selección aleatoria de K centroides.
   3. **Asignación**: cada punto se asigna al centroide más cercano.
   4. **Actualización**: cada centroide se mueve al promedio de los puntos asignados.
   5. **Repetición**: se repiten los pasos hasta que los centroides no cambian significativamente (convergencia).

2.6. Resultado: K grupos, cada uno representado por su centroide.

#### Ejemplo de K-Means en Python

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# 1. Preprocesamiento: generación y normalización de datos
X, y = make_blobs(n_samples=300, centers=3, cluster_std=0.60, random_state=0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Inicialización: selección aleatoria de K centroides
def initialize_centroids(X, k):
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    return X[random_indices]

k = 3
centroids = initialize_centroids(X_scaled, k)

# Función para visualizar los clusters
def plot_clusters(X, centroids, labels=None):
    plt.figure(figsize=(8, 6))
    if labels is not None:
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    else:
        plt.scatter(X[:, 0], X[:, 1], s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means Clustering')
    plt.show()

plot_clusters(X_scaled, centroids)

# Algoritmo K-Means
max_iterations = 100
tolerance = 1e-4

for iteration in range(max_iterations):
    # 3. Asignación: cada punto al centroide más cercano
    distances = np.sqrt(((X_scaled - centroids[:, np.newaxis])**2).sum(axis=2))
    labels = np.argmin(distances, axis=0)
    
    # Visualización en cada iteración (opcional)
    if iteration % 5 == 0:
        plot_clusters(X_scaled, centroids, labels)
    
    # Guardar los centroides anteriores para verificar convergencia
    old_centroids = centroids.copy()
    
    # 4. Actualización: mover cada centroide al promedio de sus puntos asignados
    for i in range(k):
        points_in_cluster = X_scaled[labels == i]
        if len(points_in_cluster) > 0:
            centroids[i] = points_in_cluster.mean(axis=0)
    
    # 5. Repetición hasta convergencia (cambio mínimo en los centroides)
    centroid_shift = np.sqrt(((centroids - old_centroids)**2).sum(axis=1)).max()
    
    print(f"Iteración {iteration + 1}: Cambio máximo en centroides = {centroid_shift:.6f}")
    
    if centroid_shift < tolerance:
        print(f"Convergencia alcanzada después de {iteration + 1} iteraciones!")
        break

# Resultado final
plot_clusters(X_scaled, centroids, labels)
print("Centroides finales:")
print(scaler.inverse_transform(centroids))
```

## Explicación del código:

1. **Preprocesamiento**: Generamos datos sintéticos con `make_blobs` y los normalizamos usando `StandardScaler`.

2. **Inicialización**: Seleccionamos aleatoriamente puntos iniciales como centroides.

3. **Asignación**: Calculamos la distancia de cada punto a todos los centroides y lo asignamos al más cercano.

4. **Actualización**: Recalculamos la posición de cada centroide como el promedio de todos los puntos asignados a él.

5. **Repetición**: Iteramos hasta que el cambio en los centroides sea menor que una tolerancia (convergencia).

El código incluye visualizaciones para ver cómo evolucionan los clusters durante el proceso.

Para usar este código con tus propios datos, simplemente reemplaza `X` con tu matriz de características (filas=muestras, columnas=características).

---

## 3. ¿Cómo elegir el número de clústeres?

3.1. Elegir incorrectamente el valor de K puede producir agrupaciones inútiles.

3.2. Métodos para determinar el número óptimo de K:

### 3.2.1. Inercia

- Suma de las distancias cuadradas de los puntos a sus respectivos centroides.
- Fórmula:
  \[
  \text{Inercia} = \sum_{i=1}^{n} \| x_i - \mu_{c(i)} \|^2
  \]
  donde \( x_i \) es un punto de datos y \( \mu_{c(i)} \) el centroide correspondiente.

- La inercia disminuye a medida que K aumenta, pero demasiado K puede llevar a sobreajuste.

### 3.2.2. Método del Codo

- Se grafica la inercia contra K.
- El punto de inflexión (el “codo”) indica el valor óptimo de K.
- Este método puede ser subjetivo en datasets complejos.

### 3.2.3. Coeficiente de la Silueta

- Mide la separación y cohesión de los clústeres.
- Se define como:
  \[
  s = \frac{b - a}{\max(a, b)}
  \]
  donde:
  - \( a \) es la distancia media a los puntos del mismo clúster.
  - \( b \) es la distancia al clúster más cercano.

- El valor de \( s \) va de -1 a 1.
  - Cerca de 1: buena asignación.
  - Cerca de 0: en el límite.
  - Negativo: mala asignación.

### 3.2.4. Estadística de la Brecha

- Compara la inercia real con la de datos aleatorios.
- Pasos:
  1. Calcular la inercia con datos reales.
  2. Calcularla con datos aleatorios.
  3. La diferencia logarítmica es la **brecha**.
  4. El mejor K es el que maximiza esa brecha.

---

## 4. Implementación de K-Means en Python

### Dataset: Precios de viviendas en California
   - Columnas: `longitude`, `latitude`, `median_house_value`.

### Preprocesamiento:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
datos_escalados = scaler.fit_transform(datos_inmuebles)
```

### Gráfico de dispersión:
```python
import seaborn as sns
sns.scatterplot(data = datos_inmuebles, x='longitude', y='latitude', hue='median_house_value')
```

### Búsqueda de K óptimo con silueta:
```python
from sklearn.metrics import silhouette_score
coeficientes = []
for k in range(2,11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(datos_escalados)
    coeficiente = silhouette_score(datos_escalados, kmeans.labels_)
    coeficientes.append(coeficiente)
```

### Entrenamiento con K=2:
```python
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
kmeans.fit(datos_escalados)
```

### Visualización:
```python
plt.scatter(datos_escalados[:,0], datos_escalados[:, 1], c=kmeans.labels_, cmap='viridis')
```

### Método del codo:
```python
def plot_kmeans(dataset, max_k):
    inertias = []
    for i in range(2, max_k+1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dataset)
        inertias.append(kmeans.inertia_)
    plt.plot(range(2, max_k+1), inertias, marker='o')
    plt.title('Método del codo')
```

---

## 5. Ventajas y desventajas de K-Means

### Ventajas:
1. Rápido y eficiente.
2. Fácil de implementar.
3. Escalable a grandes volúmenes de datos.
4. Resultados fácilmente interpretables.

### Desventajas:
1. Sensible a la inicialización de centroides.
2. Requiere definir K a priori.
3. No funciona bien con clústeres de forma irregular.
4. Afectado por valores atípicos.

---

## 6. Cuaderno de ejercicios

### Dataset: Segmentación de clientes
   - Columnas: sexo, estado civil, edad, educación, ingresos, ocupación, tamaño del asentamiento.

### Ejercicio 1: Preprocesamiento
```python
df = pd.read_csv('segmentation data.csv', index_col=0)
scaler = StandardScaler()
df_std = scaler.fit_transform(df)
```

### Ejercicio 2: Método del codo
```python
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss, marker='o')
```

### Ejercicio 3: Estadística de brecha
```python
def calcular_brecha(datos, n_refs=20, max_k=10):
    ...
    return brechas
```

### Ejercicio 4: Silueta
```python
def plot_silhouette(dataset, max_k):
    silhouette_scores = []
    for i in range(2, max_k+1):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(dataset)
        score = silhouette_score(dataset, kmeans.labels_)
        silhouette_scores.append(score)
    plt.plot(range(2, max_k+1), silhouette_scores, marker='o')
```

### Conclusión: El coeficiente de la silueta resultó ser el método más confiable para este dataset.

---

## 7. Referencias bibliográficas

- Dutta, I. (2020). Implementación de 7 métodos para seleccionar K óptimo.
- Van Der Post, H. y Smith, M. (2023). *Unsupervised Machine Learning with Python*.

---

## 8. Recursos multimedia complementarios

1. Segmentación de clientes paso a paso – Raúl Valerio (YouTube)
2. K-Means Clustering – Six Sigma Pro (YouTube)
3. K-Means en R – Raúl Valerio (YouTube)

---