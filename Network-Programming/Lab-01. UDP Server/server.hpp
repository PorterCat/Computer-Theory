#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>
#include <csignal>
#include <vector>
#include <climits>
#include <array>

int shutdown_flag = 0;

void signal_handler(int signum) 
{
    shutdown_flag = 1;
}

class TcpServer
{
private:
    int _sockfd;
    struct sockaddr_in _serverAddress;
    int _serverPort;

    std::vector<sockaddr_in> _clients;

public:
    TcpServer(int serverPort) : _serverPort(serverPort)
    {   
        _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        _serverAddress.sin_family = AF_INET;
        _serverAddress.sin_addr.s_addr = INADDR_ANY;
        _serverAddress.sin_port = htons(_serverPort);

        // for(int i = _serverPort; i <= USHRT_MAX; ++i)
        // {
        //     if(i == USHRT_MAX) i = 1025;

        //     _serverPort = i;
        //     _serverAddress.sin_port = htons(_serverPort);

        //     if (bind(_sockfd, (struct sockaddr*)&_serverAddress, sizeof(_serverAddress)) >= 0)
        //         break;
        // }

        bind(_sockfd, (struct sockaddr*)&_serverAddress, sizeof(_serverAddress));
        
        socklen_t len = sizeof(_serverAddress);
        if(getsockname(_sockfd, (struct sockaddr*)&_serverAddress, &len) < 0)
        {
            std::string errnoStr(std::strerror(errno));
            throw std::runtime_error{ "Failed: " + errnoStr };
        }
        _serverPort = _serverAddress.sin_port;

        std::cout << "Server is working on port: " << ntohs(_serverPort) << std::endl; 
        
        struct sigaction sa;
        sa.sa_handler = signal_handler;
        sa.sa_flags = 0;
        sigemptyset(&sa.sa_mask);
        if (sigaction(SIGINT, &sa, nullptr) < 0) {
            std::cerr << "Cannot set signal handler" << std::endl;
            throw std::exception();
        }
    }
    
    ~TcpServer() { close(_sockfd); }
    int GetPort() const { return _serverPort; }

    void Run()
    {
        char buffer[1024];
        socklen_t len = sizeof(_serverAddress);

        while(!shutdown_flag)
        {
            struct sockaddr_in client;
            len = sizeof(client);
            
            ssize_t bytes_received = recvfrom(_sockfd, buffer, sizeof(buffer), 0, (sockaddr*)&client, &len);

            if(shutdown_flag) break;

            int num = atoi(buffer);
        
            std::array<char, 64> addressBuffer;
            addressBuffer.fill('\0');

            inet_ntop(AF_INET, &client.sin_addr.s_addr, addressBuffer.data(), sizeof(addressBuffer));
            std::cout << addressBuffer.data() << ":" << client.sin_port << " - " << num << std::endl;
            
            bool is_new_client = true;
            for(auto cl : _clients)
            {
                if(client.sin_port == cl.sin_port)
                {
                    is_new_client = false;
                    break;
                }
            }
            if(is_new_client)
            {
                _clients.push_back(client);
                std::cout << "New member | " << addressBuffer.data() << ":" << client.sin_port << std::endl;   
            }

            if(client.sin_port != 0)
                SendMessage(client, std::to_string(num + 1));
        }

        const char* shutdown_message = "-1";
        for(auto client : _clients)
        {
            SendMessage(client, shutdown_message);
        }

        std::cout << "\nServer was stopped" << std::endl;
    }
private:
    void SendMessage(sockaddr_in client, std::string message)
    {
        sendto(_sockfd, message.c_str(), sizeof(message) + 1, 0, 
                (sockaddr*)&client, sizeof(client));
    }
};