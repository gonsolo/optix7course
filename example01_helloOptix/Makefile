include ../Makefile.config

TARGET = ex01_helloOptix
all: $(TARGET)
$(TARGET): main.cpp 
		clang++ -I../gonzo -I../common/gdt -o $@ $^ ../gonzo/libGonzo.a -ldl $(SWIFT_LIBS)
run: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_LIBS_PATH):../gonzo ./$(TARGET)
c: clean
clean:
	rm -f $(TARGET)
