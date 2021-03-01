include ../Makefile.config

TARGET = ex09_shadowRays
all: $(TARGET)
$(TARGET): main.cpp SampleRenderer.cpp Model.cpp ../common/glfWindow/GLFWindow.cpp
	g++ -O2 -g -rdynamic -I../common -I../softoptix -I../common/gdt -o $@ $^ ../softoptix/libSoftOptix.a -ldl -lglfw -lOpenGL $(SWIFT_LIBS)
run: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../softoptix ./$(TARGET)
c: clean
clean:
	rm -f $(TARGET) devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so
devicePrograms.o: devicePrograms.bc
	clang -O2 -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -O2 -emit-llvm -xc++ -std=c++17 -I../softoptix -I ../common/gdt -c devicePrograms.cu
gdb:
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../softoptix gdb ./$(TARGET)
perf:
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../softoptix perf record ./$(TARGET)
