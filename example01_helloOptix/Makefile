all: ex01_helloOptix
	./ex01_helloOptix
ex01_helloOptix: main.cpp ../gonzo/gonzo.cpp
	clang++ -I../gonzo -I../common/gdt -o $@ $^ -ldl
c: clean
clean:
	rm -f ex01_helloOptix
