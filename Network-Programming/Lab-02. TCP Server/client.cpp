#include "client.hpp"

int main(int argc, char* argv[]) {
    try 
    {
        if (argc != 3) 
        {
            std::cerr << "Usage: ./client <server_ip> <port>" << std::endl;
            return 1;
        }

        const char* server_ip = argv[1];
        int server_port = std::stoi(argv[2]);

        TcpClient client(server_ip, server_port);
        client.run(5); 
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}