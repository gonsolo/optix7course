include ../Makefile.config

TARGET = ex02_pipelineAndRayGen
all: $(TARGET)
$(TARGET): main.cpp SampleRenderer.cpp
	$(MAKE) -C ../softoptix
	clang++ -g -rdynamic -I../common -I../softoptix -I../common/gdt -o $@ $^ ../softoptix/libSoftOptix.a -ldl $(SWIFT_LIBS)
run: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../softoptix ./$(TARGET)
	eog osc_example2.png
c: clean
clean:
	$(MAKE) -C ../softoptix clean
	rm -f $(TARGET) devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so

devicePrograms.o: devicePrograms.bc
	clang -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -emit-llvm -xc++ -std=c++17 -I ../common/gdt -I ../softoptix/ -c devicePrograms.cu
gdb:
	gdb ./$(TARGET)
