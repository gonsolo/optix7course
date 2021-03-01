include ../Makefile.config

TARGET = ex03_inGLFWindow
all: $(TARGET)
$(TARGET): main.cpp SampleRenderer.cpp ../common/glfWindow/GLFWindow.cpp
	clang++ -g -rdynamic -I../common -I../softoptix -I../common/gdt -o $@ $^ ../softoptix/libSoftOptix.a -ldl -lglfw -lOpenGL $(SWIFT_LIBS)
run: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../softoptix ./$(TARGET)
c: clean
clean:
	rm -f $(TARGET) devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so

devicePrograms.o: devicePrograms.bc
	clang -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -emit-llvm -xc++ -std=c++17 -I ../common/gdt -c devicePrograms.cu
gdb:
	gdb ./$(TARGET)
