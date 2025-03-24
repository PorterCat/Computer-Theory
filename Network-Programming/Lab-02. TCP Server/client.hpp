#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

class TcpClient {
private:
    int _sockfd;
    struct sockaddr_in servaddr;

public:
    TcpClient(const char* server_ip, int server_port) 
    {
        _sockfd = socket(AF_INET, SOCK_STREAM, 0);

        servaddr.sin_family = AF_INET;
        servaddr.sin_port = htons(server_port);
        inet_pton(AF_INET, server_ip, &servaddr.sin_addr);

        connect(_sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr));
    }

    ~TcpClient() { close(_sockfd); }

    void run(int iterations) 
    {
        for (int i = 1; i <= iterations; ++i) 
        {
            std::string message = std::to_string(i);
            send(_sockfd, message.c_str(), message.size() + 1, 0);
            std::cout << "Send: " << i << std::endl;

            char buffer[1024];
            recv(_sockfd, buffer, sizeof(buffer), 0);
            std::cout << "Got: " << buffer << std::endl;

            sleep(i); 
        }
    }
};