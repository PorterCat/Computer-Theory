import itertools
import numpy as np


c = np.array(
    [
        [float("inf"), 13, 7, 5, 2, 9],
        [8, float("inf"), 4, 7, 8, float("inf")],
        [8, 4, float("inf"), 3, 6, 2],
        [5, 8, 1, float("inf"), 0, 1],
        [float("inf"), 8, 1, 4, float("inf"), 9],
        [10, 0, 8, 3, 6, float("inf")],
    ]
)

n = len(c)


def brute_force_tsp(cost_matrix):
    nodes = list(range(len(cost_matrix)))
    min_cost = float("inf")
    best_route = None
    checked_routes = 0

    all_routes = list(itertools.permutations(nodes[1:]))
    for route in all_routes:
        checked_routes += 1
        full_route = (0,) + route + (0,)
        cost = sum(
            cost_matrix[full_route[i]][full_route[i + 1]]
            for i in range(len(full_route) - 1)
        )
        if cost == float("inf"):
            continue
        print(f"Проверен маршрут: {full_route} с ценой {int(cost)}")
        if cost < min_cost:
            min_cost = cost
            best_route = full_route
    return min_cost, best_route, checked_routes



def branch_and_bound_tsp(cost_matrix):
    n = len(cost_matrix)
    min_cost = float("inf")
    best_route = None
    checked_routes = 0

    def calculate_lower_bound(path):
        bound = 0
        visited = set(path)
        for i in range(len(path) - 1):
            bound += cost_matrix[path[i]][path[i + 1]]

        unvisited = [node for node in range(n) if node not in visited]
        for node in unvisited:
            min1, min2 = float("inf"), float("inf")
            for j in range(n):
                if j == node:
                    continue
                if cost_matrix[node][j] < min1:
                    min2 = min1
                    min1 = cost_matrix[node][j]
                elif cost_matrix[node][j] < min2:
                    min2 = cost_matrix[node][j]
            bound += min1 + min2

        return bound / 2

    def branch(path, cost):
        nonlocal min_cost, best_route, checked_routes
        if len(path) == n:
            checked_routes += 1
            total_cost = cost + cost_matrix[path[-1]][path[0]]
            if total_cost < min_cost:
                min_cost = total_cost
                best_route = path + [path[0]]
            print(f"Проверен полный маршрут: {path + [path[0]]} с ценой {int(total_cost)}")
            return

        if cost + calculate_lower_bound(path) >= min_cost:
            return

        unvisited = [node for node in range(n) if node not in path]
        next_nodes = sorted(unvisited, key=lambda x: cost_matrix[path[-1]][x])

        for next_node in next_nodes:
            new_cost = cost + cost_matrix[path[-1]][next_node]
            if new_cost < min_cost:
                print(f"Путь: {path} -> {next_node}, стоимость: {int(new_cost)}")
                branch(path + [next_node], new_cost)

    branch([0], 0)
    return min_cost, best_route, checked_routes

brute_force_result = brute_force_tsp(c)
bnb_result = branch_and_bound_tsp(c)

print("\nПрямой перебор:")
print(f"Минимальная стоимость: {brute_force_result[0]}")
print(f"Оптимальный маршрут: {brute_force_result[1]}")
print(f"Проверено маршрутов: {brute_force_result[2]}")

print("\nМетод ветвей и границ:")
print(f"Минимальная стоимость: {bnb_result[0]}")
print(f"Оптимальный маршрут: {bnb_result[1]}")
print(f"Проверено маршрутов: {bnb_result[2]}")
