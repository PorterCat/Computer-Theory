#include <iostream>
#include "server.hpp"

#define SERVER_PORT 0

int main(int argc, char* argv[])
{
    int serverPort = SERVER_PORT;
    if(argc > 1)
    {
        serverPort = std::stoi(argv[1]);
    }

    TcpServer server(serverPort);
    server.Run();
    return 0;
}