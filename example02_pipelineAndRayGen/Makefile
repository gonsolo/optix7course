all: ex02_pipelineAndRayGen
	./ex02_pipelineAndRayGen
ex02_pipelineAndRayGen: main.cpp SampleRenderer.cpp ../gonzo/gonzo.cpp
	clang++ -g -I../common -I../gonzo -I../common/gdt -o $@ $^ -ldl
c: clean
clean:
	rm -f ex02_pipelineAndRayGen devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so

devicePrograms.o: devicePrograms.bc
	clang -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -emit-llvm -xc++ -std=c++17 -I ../common/gdt -I ../gonzo/optix_device.h -c devicePrograms.cu
gdb:
	gdb ./ex02_pipelineAndRayGen
