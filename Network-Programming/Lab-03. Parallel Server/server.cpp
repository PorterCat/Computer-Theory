#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

constexpr int MAX_CLIENTS = 10;
constexpr int BUFFER_SIZE = 1024;
constexpr const char* FILENAME = "data.txt";

std::mutex file_mutex;

void handle_client(int client_socket) 
{
    char buffer[BUFFER_SIZE];
    ssize_t bytes_read;

    while ((bytes_read = recv(client_socket, buffer, BUFFER_SIZE - 1, 0)) > 0) 
    {
        buffer[bytes_read] = '\0';

        std::lock_guard<std::mutex> lock(file_mutex);
        std::ofstream file(FILENAME, std::ios::app);
        if (file.is_open()) {
            file << buffer;
            file.close();
        }

        std::cout << '[' << std::this_thread::get_id() << ']' << " Received: " << buffer;
    }

    close(client_socket);
}

int main() 
{
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(0);

    bind(server_socket, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr));

    socklen_t addr_len = sizeof(server_addr);
    getsockname(server_socket, reinterpret_cast<sockaddr*>(&server_addr), &addr_len);
    
    std::cout << "Server started on port: " << ntohs(server_addr.sin_port) << std::endl;

    listen(server_socket, MAX_CLIENTS);

    std::cout << "Waiting for connections..." << std::endl;

    std::vector<std::thread> threads;
    while (true) 
    {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_socket = accept(server_socket, reinterpret_cast<sockaddr*>(&client_addr), &client_len);

        threads.emplace_back(std::thread(handle_client, client_socket));
        threads.back().detach();
    }

    close(server_socket);
    return 0;
}