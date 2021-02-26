include ../Makefile.config

TARGET = ex07_firstRealModel
all: $(TARGET)
$(TARGET): main.cpp SampleRenderer.cpp Model.cpp ../common/glfWindow/GLFWindow.cpp
	g++ -O2 -g -rdynamic -I../common -I../gonzo -I../common/gdt -o $@ $^ ../gonzo/libGonzo.a -ldl -lglfw -lOpenGL $(SWIFT_LIBS)
run: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../gonzo ./$(TARGET)
c: clean
clean:
	rm -f $(TARGET) devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so
devicePrograms.o: devicePrograms.bc
	clang -O2 -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -O2 -emit-llvm -xc++ -std=c++17 -I ../common/gdt -I ../gonzo/optix_device.h -c devicePrograms.cu
gdb:
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../gonzo gdb ./$(TARGET)
perf:
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../gonzo perf record ./$(TARGET)