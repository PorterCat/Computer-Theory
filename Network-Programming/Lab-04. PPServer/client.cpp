#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <cstring>
#include <string>

using namespace std;

int main(int argc, char* argv[]) 
{
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <IP> <PORT>" << endl;
        return 1;
    }

    const char* ip = argv[1];
    int port = stoi(argv[2]);

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket");
        return 1;
    }

    sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &server_addr.sin_addr) <= 0) {
        perror("inet_pton");
        close(sock);
        return 1;
    }

    if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect");
        close(sock);
        return 1;
    }

    int i;
    cout << "Enter a number (1-10): ";
    cin >> i;
    if (i < 1 || i > 10) {
        cerr << "Invalid number" << endl;
        close(sock);
        return 1;
    }

    while (true) {
        string msg = to_string(i) + "\n";
        if (send(sock, msg.c_str(), msg.size(), 0) < 0) {
            perror("send");
            break;
        }
        sleep(i);
    }

    close(sock);
    return 0;
}