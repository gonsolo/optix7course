include ../Makefile.config

TARGET = ex02_pipelineAndRayGen
all: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux ./$(TARGET)
	eog osc_example2.png
$(TARGET): main.cpp SampleRenderer.cpp ../gonzo/libGonzo.a
	clang++ -g -rdynamic -I../common -I../gonzo -I../common/gdt -o $@ $^ -ldl $(SWIFT_LIBS)
../gonzo/libGonzo.a:
	$(MAKE) -C ../gonzo
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
