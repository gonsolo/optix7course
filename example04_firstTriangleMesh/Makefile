include ../Makefile.config

TARGET = ex04_firstTriangleMesh
all: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux ./$(TARGET)
$(TARGET): main.cpp SampleRenderer.cpp ../common/glfWindow/GLFWindow.cpp
	clang++ -g -rdynamic -I../common -I../gonzo -I../common/gdt -o $@ $^ ../gonzo/libGonzo.a -ldl -lglfw -lOpenGL $(SWIFT_LIBS_PATH) $(SWIFT_LIBS)
c: clean
clean:
	rm -f $(TARGET) devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so

devicePrograms.o: devicePrograms.bc
	clang -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -emit-llvm -xc++ -std=c++17 -I ../common/gdt -I ../gonzo/optix_device.h -c devicePrograms.cu
gdb:
	gdb ./$(TARGET)

