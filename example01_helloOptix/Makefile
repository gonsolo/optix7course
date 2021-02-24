include ../Makefile.config

TARGET = ex01_helloOptix
all: $(TARGET)
	LD_LIBRARY_PATH=/home/gonsolo/downloads/swift-5.3.3-RELEASE-ubuntu20.04/usr/lib/swift/linux ./$(TARGET)
$(TARGET): main.cpp
		clang++ -I../gonzo -I../common/gdt -o $@ $^ ../gonzo/libGonzo.a -ldl $(SWIFT_LIBS_PATH) $(SWIFT_LIBS)
c: clean
clean:
	rm -f $(TARGET)
