#pragma once

// Defines shared knowledge between server and client
//
// NOTE: All functions in this interface may `exit(1)` in the event of error.

enum Message { START_COLLECT, STOP_COLLECT };

// Writes `msg` to `fd`
//
// @param fd - socket to which to write
// @param msg - message to write
void write_message(int fd, Message msg);

// Blocks until `msg` is read on `fd`.
//
// @param fd - socket to from which to read
// @param msg - message on which to wait
void wait_for(int fd, Message msg);

// Non-blocking version of `wait_for()`.
//
// @param fd - socket to from which to read
// @param msg - message on which to wait
// @return - true iff `msg` is read on `fd`
bool check_for(int fd, Message msg);
