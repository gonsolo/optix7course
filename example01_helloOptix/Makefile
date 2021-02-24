include ../Makefile.config

TARGET = ex01_helloOptix
all: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_LIBS_PATH):../gonzo ./$(TARGET)
$(TARGET): main.cpp 
		clang++ -I../gonzo -I../common/gdt -o $@ $^ -L../gonzo -lGonzales -ldl $(SWIFT_LIBS)
c: clean
clean:
	rm -f $(TARGET)
