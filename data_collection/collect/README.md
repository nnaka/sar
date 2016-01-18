# Data Collection

This project defines a program `collect` to be run on a Hummingboard connected
to a radar and GPS over USB. It waits for a signal from a client to start
collection, and collects until a stop signal is received.

## Build

With logging to stderr: `make`
Without logging: `make no-debug`

## Usage

```shell
collect <port> <gps_port> <radar_port>
```

Here, `port` is the port to which the client connects, `gps_port` is the `tty`
path to the USB on which the GPS will communicate, and `radar_port` is similarly
the USB on which the radar will communicate.
