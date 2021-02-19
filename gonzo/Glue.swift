import Foundation

@_cdecl("bla")
func bla() {
        print("bla")
        let indices = [Int]()
        let points = [Point]()
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
