# Makefile for RCM sample app for Linux

CFLAGS = -Wall

.PHONY: clean

PROGNAME = rcmSampleApp

$(PROGNAME): rcmSampleApp.o rcm.o rcmIf.o

clean:
	-rm -f *.o $(PROGNAME)
