import Foundation

@_cdecl("bla")
func bla(
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
                let _ = try createTriangleMesh(
                        indices: indices,
                        points: points,
                        normals: [],
                        uvs: [],
                        faceIndices: [])
        } catch {
                print("Error!")
        }
}
