all: $(wildcard *.cpp)
	g++ -o model $(wildcard *.cpp)