all:
	clang++ -DDEBUG=1 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp -o collect

no-debug: 
	clang++ -DDEBUG=0 -std=c++11 -Wall -Wextra -pedantic collect.cpp pulse_history.cpp -o collect

clean:
	rm -f collect
