#include <iostream>
#include <thread>
#include <chrono>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

constexpr int MAX_ATTEMPTS = 5;

int main(int argc, char* argv[]) 
{
    if (argc != 3) 
    {
        std::cerr << "Usage: " << argv[0] << " <server_ip> <port>" << std::endl;
        return EXIT_FAILURE;
    }

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) 
    {
        std::cerr << "Socket creation failed" << std::endl;
        return EXIT_FAILURE;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(std::atoi(argv[2]));
    inet_pton(AF_INET, argv[1], &server_addr.sin_addr);

    int attempts = MAX_ATTEMPTS;
    while (connect(sock, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) == -1) 
    {
        if (--attempts == 0) 
        {
            std::cerr << "Connection failed" << std::endl;
            close(sock);
            return EXIT_FAILURE;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "Connected to server" << std::endl;

    for (int i = 1; i <= 10; ++i) 
    {
        std::string message = std::to_string(i) + "\n";
        send(sock, message.c_str(), message.size(), 0);
        std::cout << "Sent: " << i << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(i));
    }

    close(sock);
    return 0;
}