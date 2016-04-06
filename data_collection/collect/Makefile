all:
	g++ -I/usr/local/include -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp piksi.cpp pulson.cpp mrmIf.cpp mrm.cpp common.cpp -L/usr/local/lib -lsbp -lserialport -o collect
	g++ -I/usr/local/include -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic client.cpp common.cpp -o client

no-debug: 
	g++ -I/usr/local/include -DDEBUG=0 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp piksi.cpp pulson.cpp mrmIf.cpp mrm.cpp common.cpp -L/usr/local/lib -lsbp -lserialport -o collect
	g++ -I/usr/local/include -DDEBUG=0 -std=c++11 -Wall -Wextra -pedantic client.cpp common.cpp -o client

client:
	g++ -I/usr/local/include -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic client.cpp common.cpp -o client

collect:
	g++ -I/usr/local/include -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp piksi.cpp pulson.cpp mrmIf.cpp mrm.cpp common.cpp -L/usr/local/lib -lsbp -lserialport -o collect

clean:
	rm -f collect client
