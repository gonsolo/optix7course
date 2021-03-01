SWIFT_SOURCE = Assert.swift FixedArray64.swift Glue.swift Shape.swift BoundingHierarchy.swift BoundingHierarchyBuilder.swift Triangle.swift Three.swift Vector3.swift Point3.swift Float.swift Normal3.swift Vector2.swift Bounds3.swift SurfaceInteraction.swift Ray.swift Geometry.swift Point2.swift Boundable.swift Intersectable.swift Utility.swift Interaction.swift
SWIFT_OBJ = $(SWIFT_SOURCE:.swift=*.o)

all: libSoftOptix.a
libSoftOptix.a: softoptix.o gonzales.o
	ar r $@ $^
softoptix.o: softoptix.cpp softoptix.h
	clang++ -O2 -fpic -I../common/gdt -c $<
gonzales.o: $(SWIFT_SOURCE)
	swiftc -Ounchecked -whole-module-optimization -emit-object -module-name Gonzales $^ -o $@
c: clean
clean:
	rm -f *.a *.o *.so 
