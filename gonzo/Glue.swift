
@_cdecl("bla")
func bla() {
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
