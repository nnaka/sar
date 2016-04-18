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
#include <time.h>
#include <sys/time.h>


#include <sys/socket.h>
#include <netinet/in.h> // struct sockaddr_in

#include "debug.h"
#include "common.h"

#include "pulse_history.h"

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
    
    // initialize data for timing
    // data is collected at a rate of 10 Hz
    struct timeval currTime;
    struct timeval prevTime;
    gettimeofday(&currTime, NULL);
    prevTime = currTime;
    long diff = 0;
    
    do {
        wait_for(sock, START_COLLECT);
        
        std::cout << "Row " << row_num << ",\n\n";

        do {
            // update current time
            gettimeofday(&currTime, NULL);
            
            // update difference between currTime and prevTime
            // check if a second has elapsed
            // this is necessary because ns wrap when seconds elapse
            // 10^9 ns is one second
            if (currTime.tv_sec > prevTime.tv_sec) {
                diff = 1000000 + (currTime.tv_usec - prevTime.tv_usec);
            } else {
                diff = currTime.tv_usec - prevTime.tv_usec;
            }
            // if diff is greater than 100ms, take gps / radar data
            // (10 Hz data rate)
            if (diff > 100000) {
                ph.collect();
                prevTime = currTime;
            }
        
        } while (!check_for(sock, STOP_COLLECT));

        std::cout << ph;

        ph.clearHistory();

        row_num++;
    } while (!check_for(sock, CLOSE_SOCKET));
    
    int end_file = -1;                          // signifies end of file for
    std::cout << (row_num-1) << "\n" << end_file;   // Matlab csv read file script
             
    close(sock);

    return 0;
}


// So time.h was never implemented for Mac OS
// This code should compensate for that,
// while the code commented out above should
// work on the HummingBoard. However, this
// code may not work on the HummingBoard

//#include <mach/mach_time.h>
//#define ORWL_NANO (+1.0E-9)
//#define ORWL_GIGA UINT64_C(1000000000)
//
//static double orwl_timebase = 0.0;
//static uint64_t orwl_timestart = 0;
//
//struct timespec orwl_gettime(void) {
//    // be more careful in a multithreaded environement
//    if (!orwl_timestart) {
//        mach_timebase_info_data_t tb = { 0 };
//        mach_timebase_info(&tb);
//        orwl_timebase = tb.numer;
//        orwl_timebase /= tb.denom;
//        orwl_timestart = mach_absolute_time();
//    }
//    struct timespec t;
//    double diff = (mach_absolute_time() - orwl_timestart) * orwl_timebase;
//    t.tv_sec = diff * ORWL_NANO;
//    t.tv_nsec = diff - (t.tv_sec * ORWL_GIGA);
//    return t;
//}
