all: devicePrograms.o
devicePrograms.o: devicePrograms.cu
	clang++ -xc++ -std=c++17 -I ../common/gdt -I optix_device.h -c devicePrograms.cu
c: clean
clean:
	rm devicePrograms.o
