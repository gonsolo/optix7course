all: gonzo.so libGonzales.so
gonzo.o: gonzo.cpp
	clang++ -fpic -I../common/gdt -c $<
gonzo.so: gonzo.o
	clang++ -shared -o $@ $<
c: clean
clean:
	rm -f gonzo.o libGonzales.so
libGonzales.so:
	swiftc -emit-library -module-name Gonzales Triangle.swift Three.swift Vector3.swift Point3.swift Float.swift Normal3.swift Vector2.swift Bounds3.swift SurfaceInteraction.swift Ray.swift Geometry.swift Point2.swift Boundable.swift Intersectable.swift Utility.swift Interaction.swift
