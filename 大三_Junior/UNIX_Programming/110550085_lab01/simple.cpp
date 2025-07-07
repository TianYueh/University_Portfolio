#include <iostream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <netdb.h>

std::string extractIPAddress(const std::string& response) {
    size_t pos = response.find("\r\n\r\n");
    if (pos != std::string::npos) {
        std::string body = response.substr(pos + 4);
        pos = body.find_first_of("0123456789.");
        if (pos != std::string::npos) {
            std::string ip = body.substr(pos);
            pos = ip.find_first_not_of("0123456789.");
            if (pos != std::string::npos) {
                ip = ip.substr(0, pos);
                return ip;
            }
        }
    }
    return "";
}

int main() {
    // Create a socket
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1) {
        std::cerr << "Socket creation failed\n";
        return 1;
    }

    // Resolve the domain name
    struct hostent *host = gethostbyname("ipinfo.io");
    if (host == nullptr) {
        std::cerr << "Failed to resolve hostname\n";
        return 1;
    }

    // Address information of the server
    struct sockaddr_in server;
    server.sin_family = AF_INET;
    server.sin_port = htons(80); // HTTP port
    server.sin_addr.s_addr = *((unsigned long*)host->h_addr);

    // Connect to the server
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        std::cerr << "Connection failed\n";
        return 1;
    }

    // HTTP request
    std::string request = "GET /ip HTTP/1.1\r\n"
                          "Host: ipinfo.io\r\n"
                          "User-Agent: curl/7.88.1\r\n"
                          "Accept: */*\r\n\r\n";

    // Send the request
    if (send(sock, request.c_str(), request.length(), 0) < 0) {
        std::cerr << "Send failed\n";
        return 1;
    }

    // Receive the response
    char response[4096];
    if (recv(sock, response, sizeof(response), 0) < 0) {
        std::cerr << "Receive failed\n";
        return 1;
    }

    // Print the response
    //std::cout << "Response:\n" << response << std::endl;

    // Extract and print the IP address
    std::string ipAddress = extractIPAddress(response);
    if (!ipAddress.empty()) {
        std::cout << ipAddress << std::endl;
    } else {
        std::cerr << "Failed to extract IP address from the response\n";
    }

    // Close the socket
    close(sock);

    return 0;
}
