import Foundation

var hierarchy = BoundingHierarchy()

@_cdecl("accelBuild")
func accelBuild(
        numIndices: UInt,
        indexPointer: UnsafeMutablePointer<UInt32>,
        numVertices: UInt,
        vertexPointer: UnsafeMutablePointer<Float>) {
        //print("gonzo swift numIndices: \(numIndices), index: \(indexPointer[0])")
        var indices = [UInt32]()
        for i in 0..<Int(numIndices) {
                indices.append(indexPointer[i])
        }
        var points = [Point]()
        for i in 0..<Int(numVertices) {
                let index = 3 * i
                let point = Point(x: vertexPointer[index], y: vertexPointer[index+1], z: vertexPointer[index+2])
                points.append(point)
        }
        do {
                let triangles = try createTriangleMesh(
                        indices: indices,
                        points: points,
                        normals: [],
                        uvs: [],
                        faceIndices: [])
                let builder = BoundingHierarchyBuilder(primitives: triangles)
                hierarchy = builder.getBoundingHierarchy()
        } catch {
                print("Error accelBuild!")
        }
}

@_cdecl("trace")
func trace(
        ox: Float, oy: Float, oz: Float,
        dx: Float, dy: Float, dz: Float,
        tmax: Float,
        result: UnsafeMutablePointer<Float>) {
        let origin = Point(x: ox, y: oy, z: oz)
        let direction = Vector(x: dx, y: dy, z: dz)
        let ray = Ray(origin: origin, direction: direction)
        var tHit: Float = Float.infinity
        do {
                let interaction = try hierarchy.intersect(ray: ray, tHit: &tHit)
                if interaction != nil {
                        result[0] = 1
                        result[1] = 1
                        result[2] = 1
                } else {
                        result[0] = 0
                        result[1] = 0
                        result[2] = 0

                }
        } catch {
                print("Error trace!")
        }

}

