# Clock Synchronization Code

This module defines a simple set of programs which ascertain the timing offsets
between the Piksi GPS, Hummingboard processor and PulsON radar.

## Usage:

```sh
$ offset
```

## Documentation

The code for the Piksi GPS SwiftNav Binary Protocol (SBP) is documented
[here](https://github.com/swift-nav). The implementation of
`get_piksi_timestamp` was heavily inspired by [this SBP
tutorial](https://github.com/swift-nav/sbp_tutorial) and in `../sbp_tutorial`.

The API for the PulsON 410 MRM is documented
[here](http://www.timedomain.com/datasheets/320-0298D%20MRM%20API%20Specification.pdf).
See page 11 for details regarding collecting a timestamp since boot. Sample code
which heavily inspired `get_pulson_timestamp` can be found
[here](http://www.timedomain.com/p400-mrm.php) and in `../sample_pulson_app`.
