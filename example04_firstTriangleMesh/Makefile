include ../Makefile.config

TARGET = ex04_firstTriangleMesh
all: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux:../gonzo ./$(TARGET)
//$(TARGET): main.cpp SampleRenderer.cpp ../common/glfWindow/GLFWindow.cpp ../gonzo/libGonzo.a

$(TARGET): main.cpp SampleRenderer.cpp ../common/glfWindow/GLFWindow.cpp
	clang++ -g -rdynamic -I../common -I../gonzo -I../common/gdt -o $@ $^ -ldl -lglfw -lOpenGL -L../gonzo -lGonzales $(SWIFT_LIBS)
#../gonzo/libGonzo.a:
#	$(MAKE) -C ../gonzo
c: clean
clean:
	rm -f $(TARGET) devicePrograms.bc devicePrograms.o
	rm -f dummy.cpp dummy.o dummy.so

devicePrograms.o: devicePrograms.bc
	clang -c $<
devicePrograms.bc: devicePrograms.cu
	clang++ -emit-llvm -xc++ -std=c++17 -I ../common/gdt -I ../gonzo/optix_device.h -c devicePrograms.cu
gdb:
	LD_LIBRARY_PATH=$(SWIFT_PATH)/linux gdb ./$(TARGET)

