all:
	clang++ -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp common.cpp -o collect
	clang++ -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic client.cpp common.cpp -o client

no-debug: 
	clang++ -DDEBUG=0 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp common.cpp -o collect
	clang++ -DDEBUG=0 -std=c++11 -Wall -Wextra -pedantic client.cpp common.cpp -o client

client:
	clang++ -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic client.cpp common.cpp -o client

collect:
	clang++ -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp common.cpp -o collect

clean:
	rm -f collect
