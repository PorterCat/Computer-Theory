#include "client.hpp"

int main(int argc, char* argv[])
{
    Client client(std::string(argv[1]), std::stoi(argv[2]));
    client.Run();
}