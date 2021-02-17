all: ex03_inGLFWindow
	./ex03_inGLFWindow
ex03_inGLFWindow: main.cpp SampleRenderer.cpp ../gonzo/gonzo.cpp ../common/glfWindow/GLFWindow.cpp
	clang++ -g -rdynamic -I../common -I../gonzo -I../common/gdt -o $@ $^ -ldl -lglfw -lOpenGL
c: clean
clean:
	rm -f ex03_inGLFWindow devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so

devicePrograms.o: devicePrograms.bc
	clang -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -emit-llvm -xc++ -std=c++17 -I ../common/gdt -I ../gonzo/optix_device.h -c devicePrograms.cu
gdb:
	gdb ./ex03_inGLFWindow
