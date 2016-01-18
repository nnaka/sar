// This program reads data from a peripheral GPS and radar over USB and
// synchronizes the results into a pulse history. The history is output to
// STDOUT in CSV format.
//
// Usage: collect <port> <gps_port> <radar_port>
// @param port [Integer] Network port on which to listen for start / stop
// signals sent by `collectClient`
// @param gps_port [String] tty device to which the GPS is connected
// @param radar_port [String] tty device to which the radar is connected

#include <stdlib.h>
#include <iostream>
#include <string>

#include <sys/socket.h>
#include <netinet/in.h> // struct sockaddr_in

#include "debug.h"
#include "pulse_history.h"

/*
 TODO
    * Add exception handling around `collect()`
    * Implement `check_for()`
 */

const int NUM_ARGS = 3;
enum Message { START_COLLECT, STOP_COLLECT };

inline void usage(const char *prog_name) {
    fprintf(stderr, "Usage: %s <port> <gps_port> <radar_port>", prog_name);
    exit(EXIT_FAILURE);
}

inline void check_or_exit(bool cond, const char *msg) {
    if (cond) {
        perror(msg);
        exit(EXIT_FAILURE);
    }
}

bool check_for(Message type, int fd, bool blocking = false) {
    (void)type, (void)fd, (void)blocking;
    return false;

    // stub
}

// Listens on `port` for incoming TCP connection and accepts it
// @param port [int] port on which to listen
// @return [int] client socket from which to read
int setup_connection(int port) {
    int serverfd, clientfd;

    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    serverfd = socket(AF_INET, SOCK_STREAM, 0);
    check_or_exit(serverfd < 0, "socket");
    LOG("serverfd = %d", serverfd);

    // initialize addr struct for typical HTTP connection
    server_addr.sin_family      = AF_INET;
    server_addr.sin_port        = htons(port);
    server_addr.sin_addr.s_addr = INADDR_ANY; // local machine IP address

    check_or_exit(bind(serverfd, (struct sockaddr *) &server_addr,
                sizeof(server_addr)), "bind");
    LOG("%s", "bind successful");

    // NOTE: This cannot fail as serverfd is a valid socket
    // and 5 is the maximum queue length
    // source: http://www.linuxhowtos.org/C_C++/socket.htm
    listen(serverfd, 5);

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

    // block until START_COLLECT
    LOG("%s", "waiting for START_COLLECT...");
    check_for(START_COLLECT, sock, true);
    LOG("%s", "received START_COLLECT");

    PulseHistory ph(argv[2], argv[3]);

    do {
        LOG("%s", "collecting()");
        ph.collect();
    } while (check_for(STOP_COLLECT, sock));
    LOG("%s", "received STOP_COLLECT");

    return 0;
}
