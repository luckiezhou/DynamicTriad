# this script aims at automatically calling cmake
# any compilation specific command are expected to be in CMakeLits.txt 

.PHONY: all clean install

all: 
	mkdir -p build && cd build && cmake ..
	make -C build

install:
	make -C build install

clean:
	rm -rf build


