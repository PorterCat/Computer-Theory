#include <iostream>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <cstring>
#include <sys/select.h>

using namespace std;

int main()
{
    int listen_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_socket == -1) {
        perror("socket");
        return 1;
    }

    int opt = 1;
    setsockopt(listen_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(0);

    if (bind(listen_socket, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("bind");
        close(listen_socket);
        return 1;
    }

    socklen_t addr_len = sizeof(server_addr);
    getsockname(listen_socket, (sockaddr*)&server_addr, &addr_len);
    cout << "Server is running on port " << ntohs(server_addr.sin_port) << endl;

    if (listen(listen_socket, 5) < 0) {
        perror("listen");
        close(listen_socket);
        return 1;
    }

    fd_set master_fds, read_fds;
    FD_ZERO(&master_fds);
    FD_SET(listen_socket, &master_fds);
    int max_fd = listen_socket;

    while (true) {
        read_fds = master_fds;
        if (select(max_fd + 1, &read_fds, nullptr, nullptr, nullptr) < 0) {
            perror("select");
            break;
        }

        for (int fd = 0; fd <= max_fd; fd++) {
            if (!FD_ISSET(fd, &read_fds)) continue;

            if (fd == listen_socket) {
                sockaddr_in client_addr;
                socklen_t client_len = sizeof(client_addr);
                int client_socket = accept(listen_socket, (sockaddr*)&client_addr, &client_len);
                if (client_socket < 0) {
                    perror("accept");
                    continue;
                }
                FD_SET(client_socket, &master_fds);
                max_fd = max(max_fd, client_socket);
                cout << "New client: " << inet_ntoa(client_addr.sin_addr) << ":" << ntohs(client_addr.sin_port) << endl;
            } else {
                char buffer[256];
                int bytes_read = recv(fd, buffer, sizeof(buffer) - 1, 0);
                if (bytes_read <= 0) {
                    close(fd);
                    FD_CLR(fd, &master_fds);
                    cout << "Client disconnected" << endl;
                } else {
                    buffer[bytes_read] = '\0';
                    cout << "Received: " << buffer;
                }
            }
        }
    }

    close(listen_socket);
    return 0;
}