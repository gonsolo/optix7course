SWIFT_PATH = -L/home/gonsolo/downloads/swift-5.3.3-RELEASE-ubuntu20.04/usr/lib/swift/linux 
SWIFT_LIBS = -lswiftSwiftOnoneSupport -lswiftCore -lFoundation -lswiftGlibc -lswiftDispatch \
             -ldispatch -lBlocksRuntime -licui18nswift -licuucswift -licudataswift
TARGET = ex01_helloOptix
all: $(TARGET)
	LD_LIBRARY_PATH=/home/gonsolo/downloads/swift-5.3.3-RELEASE-ubuntu20.04/usr/lib/swift/linux ./$(TARGET)
$(TARGET): main.cpp
		clang++ -I../gonzo -I../common/gdt -o $@ $^ ../gonzo/libGonzo.a -ldl $(SWIFT_PATH) $(SWIFT_LIBS)
c: clean
clean:
	rm -f $(TARGET)
