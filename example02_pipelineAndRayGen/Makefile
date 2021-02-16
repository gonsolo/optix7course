all: ex02_pipelineAndRayGen devicePrograms.bc
	./ex02_pipelineAndRayGen
ex02_pipelineAndRayGen: main.cpp SampleRenderer.cpp ../common/gonzo.cpp
	clang++ -g -I../common -I../common/gdt -o $@ $^
c: clean
clean:
	rm -f ex02_pipelineAndRayGen devicePrograms.bc
devicePrograms.bc: devicePrograms.cu
	clang++ -emit-llvm -xc++ -std=c++17 -I ../common/gdt -I optix_device.h -c devicePrograms.cu
gdb:
	gdb ./ex02_pipelineAndRayGen
