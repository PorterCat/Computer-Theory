#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <climits>
#include <algorithm>

const int INF = INT_MAX;

struct Node 
{
    int id;
    Node(size_t id) : id(id){}
};

struct Edge
{
    const Node* a = nullptr;
    const Node* b = nullptr;
    int weight = 0;

    bool operator<(const Edge& other) const 
    {
        return weight < other.weight;
    }
};

struct Graph
{
    std::vector<Node*> nodes;
    std::vector<Edge> nodeEdges;
    Graph(size_t nodes_n) : nodes(nodes_n)
    {
        for(size_t i = 0; i < nodes_n; ++i) 
            nodes[i] = new Node(i);
    }
    size_t size() const { return nodes.size(); }
    size_t connectionSize() const { return nodeEdges.size(); }

    void connect(int a_index, int b_index, int weight = 0)
    {
        Edge nodeEdge { nodes[a_index], nodes[b_index], weight };
        nodeEdges.push_back(nodeEdge);
    }

    ~Graph() 
    {
        for (Node* node : nodes)
            delete node;
    }
};

struct ParsingOption_Bond
{
    int a_index;
    int b_index;
    int weight;
};

Graph parseFile(const std::string& filename) 
{

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t n;
    file >> n;
    
    Graph graph {n};

    std::string line;
    while(std::getline(file, line)) 
    {
        ParsingOption_Bond option;
        file >> option.a_index >> option.b_index >> option.weight;

        graph.connect(option.a_index, option.b_index, option.weight);
    }

    return graph;
}

class DSU 
{
    std::vector<int> parent;
    std::vector<int> rank;
    
public:
    DSU(int n) 
    {
        parent.resize(n);
        rank.resize(n, 0);
        for(int i = 0; i < n; ++i)
            parent[i] = i;
    }
    
    int find(int u) 
    {
        if (parent[u] != u) 
            parent[u] = find(parent[u]);
        return parent[u];
    }
    
    void unite(int u, int v) 
    {
        int rootU = find(u);
        int rootV = find(v);
        
        if (rootU == rootV) return;
        
        if (rank[rootU] < rank[rootV]) 
            parent[rootU] = rootV;
        else 
        {
            parent[rootV] = rootU;
            if (rank[rootU] == rank[rootV])
                ++rank[rootU];
        }
    }
};

void Kruskall(Graph& graph)
{
    std::sort(graph.nodeEdges.begin(), graph.nodeEdges.end());
    
    DSU dsu(static_cast<int>(graph.size()));
    int edges_added = 0;
    int total_weight = 0;
    
    std::cout << "MST edges:\n";
    for (const Edge& edge : graph.nodeEdges) 
    {
        int u = edge.a->id; // 6
        int v = edge.b->id; // 0
        
        int rootU = dsu.find(u); 
        int rootV = dsu.find(v); 
        
        if (rootU != rootV)
        {
            std::cout << u << " -- " << v 
                      << " [weight=" << edge.weight << "]\n";
            total_weight += edge.weight;
            dsu.unite(u, v);
            ++edges_added;
        }
        
        if (edges_added == graph.size() - 1) break;
    }
    
    std::cout << "Total MST weight: " << total_weight << "\n";

}

int main(int argc, char* argv[])
{
    if (argc < 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }

    Graph graph = parseFile(argv[1]);
    Kruskall(graph); 

    return 0;
}