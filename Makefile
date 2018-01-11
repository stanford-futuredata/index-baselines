source = fast.cpp range_search.cpp hashing.cpp gen_lognormal.cpp
objects = 
executables = range_search gen_lognormal fast hashing
CXX = g++
LANGFLAGS = -std=c++14
CXXFLAGS = -Ofast $(LANGFLAGS) -march=core-avx2 -msse -Istx-btree-0.9/include -pedantic -Wall -Wextra -Weffc++
LIBS =

all: gen_lognormal range_search fast hashing

hashing: hashing.o
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LIBS)

gen_lognormal: gen_lognormal.o
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LIBS)

range_search: range_search.o 
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LIBS)

fast: fast.o
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	-rm -f $(executables) *.o *~
