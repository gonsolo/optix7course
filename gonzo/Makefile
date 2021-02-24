SWIFT_SOURCE = Assert.swift FixedArray64.swift Glue.swift Shape.swift BoundingHierarchy.swift BoundingHierarchyBuilder.swift Triangle.swift Three.swift Vector3.swift Point3.swift Float.swift Normal3.swift Vector2.swift Bounds3.swift SurfaceInteraction.swift Ray.swift Geometry.swift Point2.swift Boundable.swift Intersectable.swift Utility.swift Interaction.swift
SWIFT_OBJ = $(SWIFT_SOURCE:.swift=*.o)

#all: libGonzo.a
all: libGonzales.so
#libGonzo.so: gonzo.o
#	clang++ -shared -o $@ $<
gonzo.o: gonzo.cpp
	clang++ -fpic -I../common/gdt -c $<
#libGonzo.a: gonzo.o $(SWIFT_SOURCE)
#	swiftc -module-name Gonzales -emit-object $(SWIFT_SOURCE)
#	ar r $@ gonzo.o $(SWIFT_OBJ)
libGonzales.so: $(SWIFT_SOURCE) gonzo.o
	swiftc -emit-library -module-name Gonzales $^
c: clean
clean:
	rm -f gonzo.o libGonzo.a libGonzo.so libGonzales.so

