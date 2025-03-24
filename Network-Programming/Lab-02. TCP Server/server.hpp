#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <arpa/inet.h>

class TcpServer 
{
private:
    int _sockfd;
    int port;

    void listenForClients() 
    {
        if (listen(_sockfd, 5) < 0) 
        {
            close(_sockfd);
            throw std::runtime_error("Listen error");
        }
    }

    void handleChildProcesses() 
    {
        signal(SIGCHLD, [](int sig) 
        {
            while (waitpid(-1, nullptr, WNOHANG) > 0);
        });
    }

public:
    explicit TcpServer(int port) : port(port) 
    {
        _sockfd = socket(AF_INET, SOCK_STREAM, 0);
        
        struct sockaddr_in servaddr;
        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = INADDR_ANY;
        servaddr.sin_port = htons(port);

        bind(_sockfd, (struct sockaddr*)&servaddr, sizeof(servaddr));

        listenForClients();
        handleChildProcesses();
        std::cout << "Server is working on port: " << port << std::endl;
    }

    ~TcpServer() 
    {
        close(_sockfd);
    }

    void run() 
    {
        struct sockaddr_in cliaddr;
        socklen_t len = sizeof(cliaddr);

        while (true) 
        {
            int client_sockfd = accept(_sockfd, (struct sockaddr*)&cliaddr, &len);

            pid_t pid = fork();

            if (pid == 0) 
            {
                close(_sockfd);
                handleClient(client_sockfd);
                exit(0);
            } 
            else 
            {
                close(client_sockfd);
            }
        }
    }

private:
    void handleClient(int client_sockfd) 
    {
        char buffer[1024];

        struct sockaddr_in clientAddr;
        socklen_t addrLen = sizeof(clientAddr);

        getpeername(client_sockfd, (struct sockaddr*)&clientAddr, &addrLen);

        const char* clientIP = inet_ntoa(clientAddr.sin_addr);
        int clientPort = ntohs(clientAddr.sin_port);

        while (true) 
        {
            ssize_t bytes_received = recv(client_sockfd, buffer, sizeof(buffer), 0);
            if (bytes_received <= 0) 
            {
                break;
            }

            buffer[bytes_received] = '\0';
            int number = atoi(buffer);
            std::cout << "Get: " << number << " from " << clientIP << ':' << clientPort << std::endl;

            int response = number + 1;
            send(client_sockfd, std::to_string(response).c_str(), std::to_string(response).size() + 1, 0);
        }

        close(client_sockfd);
    }
};