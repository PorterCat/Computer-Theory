class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def add_edge(self, u, v, w):
        self.graph.append([w, u, v])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        root_x = self.find(parent, x)
        root_y = self.find(parent, y)

        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
            print(f"Union: root {root_x} -> root {root_y}")
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
            print(f"Union: root {root_y} -> root {root_x}")
        else:
            parent[root_y] = root_x
            rank[root_x] += 1
            print(f"Union: roots {root_x} and {root_y} tied, {root_x} becomes root")

    def kruskal_mst(self):
        result = []

        print("Sorting edges by weight...")
        self.graph = sorted(self.graph, key=lambda item: item[0])
        for w, u, v in self.graph:
            print(f"Edge {u} -- {v} with weight {w}")

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        e = 0
        i = 0

        print("\nProcessing edges for MST...")
        while e < self.V - 1:
            w, u, v = self.graph[i]
            i += 1
            root_u = self.find(parent, u)
            root_v = self.find(parent, v)

            print(f"Checking edge {u} -- {v} with weight {w}: roots {root_u}, {root_v}")
            if root_u != root_v:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, root_u, root_v)
                print(f"Added edge {u} -- {v} with weight {w} to MST")

        print("\nFinal MST:")
        for u, v, weight in result:
            print(f"Edge {u} -- {v} of weight {weight}")


def main():
    # Ввод из файла
    with open("graph_input.txt", "r") as file:
        lines = file.readlines()

    V = int(lines[0].strip())  # Количество вершин
    g = Graph(V)

    print("Reading edges from file...")
    for line in lines[1:]:
        u, v, w = map(int, line.strip().split())
        g.add_edge(u, v, w)
        print(f"Read edge {u} -- {v} with weight {w}")

    print("\nStarting Kruskal's algorithm...")
    g.kruskal_mst()


if __name__ == "__main__":
    main()
