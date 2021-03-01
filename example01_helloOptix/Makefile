include ../Makefile.config

TARGET = ex01_helloOptix
all: $(TARGET)
$(TARGET): main.cpp 
	echo $(SWIFT_LIBS)
	clang++ -I../softoptix -I../common/gdt -o $@ $^ ../softoptix/libSoftOptix.a -ldl $(SWIFT_LIBS)
run: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_LIBS_PATH):../softoptix ./$(TARGET)
c: clean
clean:
	rm -f $(TARGET)
