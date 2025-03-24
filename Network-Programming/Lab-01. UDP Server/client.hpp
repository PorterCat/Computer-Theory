#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

class Client
{
private:
    int _sockfd;
        
    struct sockaddr_in _serverAddress;

public:
    Client(std::string serverIP, int serverPort) 
    {
        _sockfd = socket(AF_INET, SOCK_DGRAM, 0);
        _serverAddress.sin_family = AF_INET;
        _serverAddress.sin_port = htons(serverPort);

        inet_pton(AF_INET, serverIP.c_str(), &_serverAddress.sin_addr);
    }

    ~Client() { close(_sockfd); }

    void SendRequest(int number) 
    {
        char message[10];
        snprintf(message, sizeof(message), "%d", number);
        sendto(_sockfd, message, strlen(message) + 1, 0, 
               (struct sockaddr*)&_serverAddress, sizeof(_serverAddress));
    }

    std::string ReceiveResponse() 
    {
        char reply[1024];
        recvfrom(_sockfd, reply, sizeof(reply), 0, nullptr, nullptr);
        return std::string(reply);
    }

    void Run() 
    {
        int i = 0;
        while (i != -1) 
        {
            SendRequest(i);
            std::string response = ReceiveResponse();
            sleep(i);
            i = stoi(response);
        }

        std::cout << "I'm dead :(" << std::endl;
    }

};