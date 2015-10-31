# Makefile for MRM sample app for Linux

CFLAGS = -Wall -Wextra -g

.PHONY: clean all

all: getPulsonTimestamp

getPulsonTimestamp: getPulsonTimestamp.o mrm.c mrmIf.o

clean:
	-rm -f *.o getPulsonTimestamp
