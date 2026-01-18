import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# 1. Загружаем данные
# Обратите внимание: разделитель столбцов - точка с запятой, десятичный разделитель - запятая
df = pd.read_csv('27_B_17834.csv', sep=';', decimal=',')

# Проверяем данные
print("Первые 5 строк данных:")
print(df.head())
print()

# 2. Преобразуем данные в numpy массив для работы с KMeans
X = df[['X', 'Y']].values

# 3. Создаем и обучаем модель KMeans с 3 кластерами
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# 4. Получаем метки кластеров для каждой точки
labels = kmeans.labels_
df['cluster'] = labels

print("Количество точек в каждом кластере:")
print(df['cluster'].value_counts().sort_index())
print()

# 5. Находим центроид каждого кластера (точку с минимальной суммой расстояний до других точек кластера)
centroids = []

for cluster_id in range(3):
    # Выбираем точки текущего кластера
    cluster_points = df[df['cluster'] == cluster_id][['X', 'Y']].values

    # Инициализируем переменные для поиска наилучшего центроида
    min_total_distance = float('inf')
    best_centroid_idx = -1
    best_centroid_coords = None

    # Для каждой точки в кластере вычисляем сумму расстояний до всех других точек
    for i in range(len(cluster_points)):
        point = cluster_points[i]

        # Вычисляем сумму расстояний от текущей точки до всех других точек в кластере
        distances = np.sqrt(np.sum((cluster_points - point) ** 2, axis=1))
        total_distance = np.sum(distances)

        # Если эта сумма меньше текущего минимума, обновляем лучший центроид
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_centroid_idx = i
            best_centroid_coords = point

    # Добавляем найденный центроид в список
    centroids.append(best_centroid_coords)

    print(f"Кластер {cluster_id}:")
    print(f"  Лучший центроид: ({best_centroid_coords[0]:.3f}, {best_centroid_coords[1]:.3f})")
    print(f"  Сумма расстояний: {min_total_distance:.3f}")
    print(f"  Количество точек в кластере: {len(cluster_points)}")
    print()

# 6. Преобразуем центроиды в DataFrame для удобного вывода
centroids_df = pd.DataFrame(centroids, columns=['X', 'Y'])
centroids_df.index.name = 'Cluster'

print("=" * 50)
print("Итоговые центроиды для каждого кластера:")
print(centroids_df)
print()

# 7. Для сравнения: центроиды, найденные KMeans (это средние точки, не обязательно из набора данных)
print("Центроиды, найденные алгоритмом KMeans (средние точки кластеров):")
print(pd.DataFrame(kmeans.cluster_centers_, columns=['X', 'Y']))

plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue']

# Рисуем точки кластеров
for cluster_id in range(3):
    cluster_data = df[df['cluster'] == cluster_id]
    plt.scatter(cluster_data['X'], cluster_data['Y'],
                c=colors[cluster_id], alpha=0.6,
                label=f'Кластер {cluster_id}')

# Рисуем центроиды
centroids_array = np.array(centroids)
plt.scatter(centroids_array[:, 0], centroids_array[:, 1],
            c='black', marker='X', s=200,
            label='Центроиды', edgecolors='white', linewidth=2)

plt.xlabel('Координата X')
plt.ylabel('Координата Y')
plt.title('Кластеризация звезд с центроидами')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
