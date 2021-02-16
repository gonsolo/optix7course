all: ex02_pipelineAndRayGen
	./ex02_pipelineAndRayGen
ex02_pipelineAndRayGen: main.cpp SampleRenderer.cpp ../common/gonzo.cpp
	clang++ -I../common -I../common/gdt -o $@ $^
c: clean
clean:
	rm -f ex02_pipelineAndRayGen devicePrograms.o

devicePrograms.o: devicePrograms.cu
	clang++ -xc++ -std=c++17 -I ../common/gdt -I optix_device.h -c devicePrograms.cu
