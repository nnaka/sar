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
Instructions here: https://sigrok.org/wiki/Libserialport

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
client <host> <port>
```

Here, `port` is the port to which the client connects, `gps_port` is the `tty`
path to the USB on which the GPS will communicate, and `radar_port` is similarly
the USB on which the radar will communicate.

NOTE: When inside be sure to put the Piksi in simulation mode via the Piksi
Console GUI. Navigate to "Settings" and click on the sidemenu item "Simulation".
Make sure to set "Enable" to true.

## SSH to Hummingboard

1) Turning on the Hummingboard starts an ad hoc WiFi network 'SAR-PROJECT' -- wait a few
minutes and join this on your computer
2) Configure your network settings to use IP address 192.168.0.2 with a subnet
mask of 255.255.255.0.
3) SSH to the Hummingboard at IP address 192.168.0.1 via:

```shell
ssh root@192.168.0.1
```

The password is 'modelthegaussian' (no quotes).

## Misc

The top USB port on the Hummingboard is named `/dev/ttyUSB0`, the bottom
`/dev/ttyACM0`.
