#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <string>
#include <thread>

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

void handle_request(http::request<http::string_body> req, http::response<http::string_body>& res) {
    // Проверяем метод запроса и URI
    if (req.method() != http::verb::get || req.target() != "/") {
        res.result(http::status::not_found);
        res.set(http::field::content_type, "text/plain");
        res.body() = "404 Not Found";
        return;
    }

    res.result(http::status::ok);
    res.set(http::field::content_type, "text/html");
    res.body() = R"(
        <!DOCTYPE html>
        <html>
            <head>
                <title>My C++ HTTP Server</title>
            </head>
            <body>
                <h1>Hello from C++ HTTP Server!</h1>
                <p>This is a static HTML page served by a C++ program using Boost.Beast.</p>
            </body>
        </html>
    )";
}

int main(int argc, char* argv[]) {
    try {
        std::string address = "0.0.0.0";
        unsigned short port = 8888;
        int thread_count = std::thread::hardware_concurrency();

        net::io_context ioc{thread_count};

        tcp::acceptor acceptor{ioc, {net::ip::make_address(address), port}};
        std::cout << "HTTP server running on port " << port << std::endl;

        auto session = [&](tcp::socket socket) {
            try {
                beast::flat_buffer buffer;

                http::request<http::string_body> req;
                http::read(socket, buffer, req);

                http::response<http::string_body> res{http::status::ok, req.version()};
                res.set(http::field::server, "C++ Beast HTTP Server");

                handle_request(req, res);

                res.prepare_payload();
                http::write(socket, res);
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        };

        std::vector<std::thread> threads;
        for (int i = 0; i < thread_count; ++i) {`1
            threads.emplace_back([&]() {
                while (true) {
                    try {
                        tcp::socket socket{ioc};
                        acceptor.accept(socket);

                        session(std::move(socket));
                    } catch (const std::exception& e) {
                        std::cerr << "Accept error: " << e.what() << std::endl;
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}