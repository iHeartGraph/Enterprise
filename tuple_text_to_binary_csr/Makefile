exe = text_to_bin.bin

cc = "$(shell which g++)" 
flags = -O3

objs = $(patsubst %.cpp,%.o,$(wildcard ../../lib/*.cpp)) \
			$(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps = $(wildcard ../../lib/*.h) \
				$(wildcard *.h) \
				Makefile

%.o:%.cpp $(deps)
	$(cc) -c $< -o $@ $(flags)

$(exe):$(objs)
	$(cc) $(objs) -o $(exe) $(flags)

clean:
	rm -rf $(exe) $(objs)
