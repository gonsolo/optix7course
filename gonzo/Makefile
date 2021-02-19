all: gonzo.o libGonzales.so
gonzo.o: gonzo.cpp
	clang++ -I../common/gdt -c $<
c: clean
clean:
	rm -f gonzo.o libGonzales.so
libGonzales.so:
	swiftc -emit-library -module-name Gonzales Three.swift Vector3.swift Point3.swift Float.swift
