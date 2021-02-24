include ../Makefile.config

TARGET = ex01_helloOptix
all: $(TARGET)
	LD_LIBRARY_PATH=$(SWIFT_LIBS_PATH) ./$(TARGET)
$(TARGET): main.cpp ../gonzo/libGonzo.a
		clang++ -I../gonzo -I../common/gdt -o $@ $^ -ldl $(SWIFT_LIBS)
../gonzo/libGonzo.a:
	$(MAKE) -C ../gonzo
c: clean
clean:
	rm -f $(TARGET)
