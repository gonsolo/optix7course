import Foundation

var triangles = [Triangle<UInt32>]()
var hierarchy = BoundingHierarchy()

@_cdecl("addTriangles")
func addTriangles(
        numIndices: UInt,
        indexPointer: UnsafeMutablePointer<UInt32>,
        numVertices: UInt,
        vertexPointer: UnsafeMutablePointer<Float>) {
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
                let mesh = try createTriangleMesh(
                        indices: indices,
                        points: points,
                        normals: [],
                        uvs: [],
                        faceIndices: [])
                triangles.append(contentsOf: mesh)
        } catch {
                print("Error accelBuild!")
        }
}

@_cdecl("accelBuild")
func accelBuild() {
        let builder = BoundingHierarchyBuilder(primitives: triangles)
        hierarchy = builder.getBoundingHierarchy()
}

@_cdecl("trace")
func trace(
        ox: Float, oy: Float, oz: Float,
        dx: Float, dy: Float, dz: Float,
        tmax: Float,
        result: UnsafeMutablePointer<Int>)
        {
        let origin = Point(x: ox, y: oy, z: oz)
        let direction = Vector(x: dx, y: dy, z: dz)
        let ray = Ray(origin: origin, direction: direction)
        var tHit: Float = Float.infinity
        do {
                guard let interaction = try hierarchy.intersect(ray: ray, tHit: &tHit) else {
                        result.pointee = -1
                        return
                }
                guard let triangle = interaction.primitive as? Triangle<UInt32> else {
                        result.pointee = -1
                        return
                }
                result.pointee = Int(triangle.idx) / 3
        } catch {
                print("Error trace!")
        }

}

