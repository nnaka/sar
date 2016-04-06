// client.cpp
// This program defines a simple interactive session for establishing a
// connection to a server running `collect` and issuing [START,STOP]_COLLECT
// commands to it.
//
// Usage: client <host> <port>
// @param host [String] IP address of host
// @param port [Integer] Port on which host is listening for connections

// Used to quickly enable and disable testing with an ad hoc versus
// infrastructure WiFi connection
#define AD_HOC 0

#include <stdlib.h>
#include <iostream>
#include <string.h>

#include <unistd.h>     // write
#include <sys/socket.h>
#include <netinet/in.h> // struct sockaddr_in
#include <arpa/inet.h>  // inet_addr

#if AD_HOC==0
#include <netdb.h>      // gethostbyaddr()
#endif

#include "debug.h"
#include "common.h"

using namespace std;

const int NUM_ARGS = 2;

inline void usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s <host> <port>\n", prog_name);
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    if (argc != NUM_ARGS + 1) {
        usage(argv[0]);
    }

    if (strspn(argv[2], "0123456789") != strlen(argv[2])) {
        fprintf(stderr,"Port number %s is not numeric\n", argv[2]);
        usage(argv[0]);
    }

    struct sockaddr_in server_addr;
    memset((char *)&server_addr, 0, sizeof(server_addr));

    // initialize addr struct for typical TCP connection
    server_addr.sin_family      = AF_INET;
    server_addr.sin_port        = htons(atoi(argv[2]));

#if AD_HOC
    server_addr.sin_addr.s_addr = inet_addr(argv[1]);
#else
    struct hostent *he = gethostbyname(argv[1]);
    if (he == nullptr) { herror("gethostbyname"); return 1; }

    struct in_addr **addr_list = (struct in_addr **)he->h_addr_list;
    server_addr.sin_addr = *addr_list[0];
#endif

    int fd = socket(AF_INET, SOCK_STREAM, 0);

    check_or_exit(fd < 0, "socket");
    check_or_exit(connect(fd, (struct sockaddr *)&server_addr,
                sizeof(server_addr)), "connect");

    LOG("connected on fd = %d", fd);

    cout << "Press ENTER to start collection";
    cin.ignore();

    write_message(fd, START_COLLECT);

    cout << "Press ENTER to stop collection";
    cin.ignore();
 
    write_message(fd, STOP_COLLECT);

    close(fd);

    return 0;
}
