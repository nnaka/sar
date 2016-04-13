// This program reads data from a peripheral GPS and radar over USB and
// synchronizes the results into a pulse history. The history is output to
// STDOUT in CSV format.
//
// Usage: collect <port> <gps_port> <radar_port>
//
// @param port [int] network port on which to listen for start / stop
// @param gps_port [string] tty device to which the GPS is connected
// @param radar_port [string] tty device to which the radar is connected

#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <unistd.h> // close

#include <sys/socket.h>
#include <netinet/in.h> // struct sockaddr_in

#include "debug.h"
#include "common.h"

#include "pulse_history.h"

using namespace std;

/*
 * TODO
 *
 * Add exception handling around `collect()`
 * Implement printing of PulseHistory
 *
 */

const int NUM_ARGS = 3;

inline void usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s <port> <gps_port> <radar_port>\n", prog_name);
    exit(EXIT_FAILURE);
}

// Listens on `port` for incoming TCP connection and accepts it
//
// @param port - port on which to listen
// @return - client socket from which to read
int setup_connection(int port) {
    int serverfd, clientfd;

    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    serverfd = socket(AF_INET, SOCK_STREAM, 0);
    check_or_exit(serverfd < 0, "socket");

    LOG("serverfd = %d", serverfd);

    server_addr.sin_family      = AF_INET;
    server_addr.sin_port        = htons(port);
    server_addr.sin_addr.s_addr = INADDR_ANY; // local machine IP address

    check_or_exit(bind(serverfd, (struct sockaddr *) &server_addr,
                sizeof(server_addr)), "bind");

    LOG("%s", "bind successful");

    check_or_exit(listen(serverfd, 5) != 0, "listen");

    clientfd = accept(serverfd, (struct sockaddr *) &client_addr, &client_len);
    check_or_exit(clientfd < 0, "accept");

    LOG("accepted connection on clientfd = %d", clientfd);

    return clientfd;
}

int main(int argc, char *argv[]) {
    if (argc != NUM_ARGS + 1) {
        usage(argv[0]);
    }

    if (strspn(argv[1], "0123456789") != strlen(argv[1])) {
        fprintf(stderr,"Port number %s is not numeric\n", argv[1]);     
        usage(argv[0]);
    }

    int sock = setup_connection(atoi(argv[1]));

    PulseHistory ph(argv[2], argv[3]);
    
    int row_num = 1;
    
    do {
        wait_for(sock, START_COLLECT);
        
        std::cout << "Row " << row_num << ",\n\n";

        do { ph.collect(); } while (!check_for(sock, STOP_COLLECT));

        cout << ph;

        ph.clearHistory();

        row_num++;
    } while (!check_for(sock, CLOSE_SOCKET));
    
    int end_file = -1;                          // signifies end of file for
    cout << (row_num-1) << "\n" << end_file;   // Matlab csv read file script
             
    close(sock);

    return 0;
}
