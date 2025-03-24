#include <iostream>
#include "server.hpp"

#define SERVER_PORT 0

int main(int argc, char* argv[]) {
    try 
    {
        if (argc > 2) 
        {
            std::cerr << "Usage: ./server <port>" << std::endl;
            return 1;
        }

        int port = 0;

        if(argc == 2)
            port = std::stoi(argv[1]);
            
        TcpServer server(port);
        server.run();
    } 
    catch (const std::exception& e) 
    {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}