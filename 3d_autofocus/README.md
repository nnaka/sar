# 3D Autofocus

## Overview

TBD

## Build

There are four build commands.

0) `make driver` - compiles and links a C++ version of the driver code
1) `make driver_cuda` - compiles and links a CUDA version of the driver code
2) `make cpp` - compiles and links a MEX / C++ version of `gradH`
3) `make cuda` - compiles and links a MEX / CUDA version of `gradH`

NOTE that to use MEX and CUDA on Halligan servers requires `use` commands, namely:

`use matlab2014a` and `use cuda`.

These set up the environment properly to compile with these tools. Therefore,
typical usage could be:

```shell
use matlab2014a
use cuda
make cpp
matlab -nodesktop -nojvm -nosplash -r "ProcessImage('<filename>')"
```

This will build a MATLAB function `ProcessImage` which will call into the C++
implementation of `gradH`. Then, it will run MATLAB headlessly (in the command
line) using `-nodesktop -nojvm -nosplash`. The first command executed (as
if at the MATLAB prompt) will be `ProcessImage` as specified by the `-r` flag.

To run the driver program, which is useful for profiling or quick tests (as it
generates its own data):

```shell
use cuda
make driver_cuda      # or `make driver`
./grad_h_driver_cuda  # or `./grad_h_driver`
```

This will build the driver program to use the CUDA implementation of `gradH`
and run it.
