source = fast.cpp range_search.cpp interpolation_search.cpp hashing.cpp  generate_lognormal.cpp
objects = 
executables = range_search gen_lognormal fast hashing
CXX = g++
LANGFLAGS = -std=c++17
CXXFLAGS = -Ofast $(LANGFLAGS) -march=core-avx2 -msse -Istx-btree-0.9/include -pedantic -Wall -Wextra -Weffc++
LIBS =

all: gen_lognormal range_search fast hashing

hashing: hashing.o
	$(CXX) $(CXXFLAGS) -o $@ $+ $(LIBS)

gen_lognormal: generate_lognormal.o 
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
