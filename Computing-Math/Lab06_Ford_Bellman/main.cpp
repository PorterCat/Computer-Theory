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

bool BellmanFord(const Graph& graph, int src, bool verbose = true)
{
    int V = graph.size();
    std::vector<int> dist(V, INF);
    dist[src] = 0;

    if(verbose) 
    {
        std::cout << "Initial distances:\n";
        for(int i = 0; i < V; ++i) 
            std::cout << "Vertex " << i << ": " << (dist[i] == INF ? "INF" 
                : std::to_string(dist[i])) << "\n";
    }

    for(int i = 1; i < V; ++i) 
    {
        if(verbose) std::cout << "\nIteration " << i << ":\n";
        
        for(const auto& edge : graph.nodeEdges) 
        {
            int u = edge.a->id;
            int v = edge.b->id;
            int w = edge.weight;

            if(dist[u] != INF && dist[v] > dist[u] + w) 
            {
                dist[v] = dist[u] + w;
                if(verbose) {
                    std::cout << "Relaxed edge (" << u << " -> " << v 
                              << ") with weight " << w 
                              << ". Updated distance of " << v 
                              << " to " << dist[v] << "\n";
                }
            }
        }
    }

    bool hasNegativeCycle = false;
    for(const auto& edge : graph.nodeEdges) 
    {
        int u = edge.a->id;
        int v = edge.b->id;
        int w = edge.weight;

        if(dist[u] != INF && dist[v] > dist[u] + w) {
            hasNegativeCycle = true;
            break;
        }
    }

    if(verbose) {
        std::cout << "\nFinal distances:\n";
        for(int i = 0; i < V; ++i) {
            std::cout << "Vertex " << i << ": ";
            if(dist[i] == INF) std::cout << "INF";
            else std::cout << dist[i];
            std::cout << "\n";
        }
    }

    if(hasNegativeCycle)
    {
        std::cout << "\nGraph contains negative weight cycle!\n";
        return false;
    }
    return true;
}

int main(int argc, char* argv[])
{
    if (argc < 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
        return 1;
    }
    
    Graph graph = parseFile(argv[1]);
    bool success = BellmanFord(graph, 0); 
    
    if(success) 
        std::cout << "\nAlgorithm completed successfully!\n";
}

// 15 9 5 2