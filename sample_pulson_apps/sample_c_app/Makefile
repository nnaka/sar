# Makefile for MRM sample app for Linux

CFLAGS = -Wall

.PHONY: clean all

all: mrmSampleApp

mrmSampleApp: mrmSampleApp.o mrm.o mrmIf.o

clean:
	-rm -f *.o mrmSampleApp
