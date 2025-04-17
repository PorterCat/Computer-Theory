class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def printArr(self, dist):
        print("Vertex Distance from Source")
        for i in range(self.V):
            print(f"{i} \t\t {dist[i]}")

    def BellmanFord(self, src):
        dist = [float("Inf")] * self.V
        dist[src] = 0

        print(f"Initial distances: {dist}")
        for i in range(self.V - 1):
            print(f"\nIteration {i + 1}:")
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    print(
                        f"Relaxed edge ({u}, {v}) with weight {w}. Updated distance of vertex {v} to {dist[v]}"
                    )
            print(f"Distances after iteration {i + 1}: {dist}")

        for u, v, w in self.graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Graph contains negative weight cycle")
                return

        self.printArr(dist)


def main():
    # Ввод из файла
    with open("graph_input.txt", "r") as file:
        lines = file.readlines()

    # Создаём граф
    V = int(lines[0].strip())  # Количество вершин
    g = Graph(V)

    print("Reading graph from file...\n")
    for line in lines[1:]:
        u, v, w = map(int, line.strip().split())
        g.addEdge(u, v, w)
        print(f"Read edge ({u}, {v}) with weight {w}")

    print("\nGraph successfully read. Starting algorithm...\n")
    start_vertex = 0  # Стартовая вершина для алгоритма Беллмана-Форда
    g.BellmanFord(start_vertex)


if __name__ == "__main__":
    main()
