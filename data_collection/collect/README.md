# Data Collection

This project defines a program `collect` to be run on a Hummingboard connected
to a radar and GPS over USB. It waits for a signal from a client to start
collection, and collects until a stop signal is received.

## Build

### Dependencies

Ensure you have autoconf, aclocal, and libtool installed. On a Mac:

```shell
brew install autoconf
brew install automake
brew install libtool
```

Required: SwiftNav Binary Protocol (SBP) library
Instructions here: https://swift-nav.github.io/libsbp/c/build/docs/html/install.html

Required: `libserialport`
Insturctions here: https://sigrok.org/wiki/Libserialport

```shell
git clone git://sigrok.org/libserialport
cd libserialport
./autogen.sh
./configure
make
sudo make install
```

SBP is dependent on cmake. get cMake with homebrew: brew install cmake

With logging to stderr: `make`
Without logging: `make no-debug`

## Usage

```shell
collect <port> <gps_port> <radar_port>
```

Here, `port` is the port to which the client connects, `gps_port` is the `tty`
path to the USB on which the GPS will communicate, and `radar_port` is similarly
the USB on which the radar will communicate.
