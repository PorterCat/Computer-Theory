import heapq
import pandas as pd


def dijkstra_with_table(graph, start):
    """
    Алгоритм Дейкстры с генерацией таблицы.
    """
    # Инициализация расстояний и таблицы
    distances = {vertex: float("infinity") for vertex in graph}
    distances[start] = 0

    priority_queue = [(0, start)]
    visited = set()

    table = []  # Таблица для хранения шагов алгоритма

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        # Сохраняем текущую строку таблицы

        # Обрабатываем соседей
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

        table.append(
            {
                "S": sorted(list(visited), reverse=True),  # Обратный порядок
                "w": current_vertex,
                "D(w)": current_distance,
                "D()": dict(distances),  # Сохраняем текущие минимальные расстояния
            }
        )
    return table


def format_distances(distances, n):
    """
    Форматирует список расстояний для D(), заменяя infinity на '-'.
    Преобразуем в нужный формат для вывода как строки.
    """
    formatted = []
    for i in range(n):
        if distances.get(i, float("inf")) == float("inf"):
            formatted.append("-")
        else:
            formatted.append(str(distances[i]))
    return " ".join(formatted)


def main():
    # Ввод графа из методички
    graph = {
        0: {1: 25, 2: 15, 3: 7, 4: 2},
        1: {0: 25, 2: 6},
        2: {1: 6, 3: 4},
        3: {2: 4, 4: 3},
        4: {0: 2, 3: 3},
    }

    # Выполнение алгоритма Дейкстры
    start_vertex = 0
    table = dijkstra_with_table(graph, start_vertex)

    # Определяем размерность графа для корректного вывода
    n = len(graph)

    # Вывод таблицы
    print("\nТаблица:")
    print(f"{'S':<20} {'w':<3} {'D(w)':<6} {'D()':<30}")
    for row in table:
        s_str = ", ".join(map(str, row["S"]))  # Преобразование списка S в строку
        distances_str = format_distances(row["D()"], n)
        print(f"{s_str:<20} {row['w']:<3} {row['D(w)']:<6} {distances_str}")

    # Формирование таблицы в формате DataFrame
    df = pd.DataFrame(table)
    print("\nТаблица в формате DataFrame:")
    print(df)


if __name__ == "__main__":
    main()
