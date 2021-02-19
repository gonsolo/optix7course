import Foundation

@_cdecl("bla")
func bla(indices: UnsafeMutablePointer<UInt32>) {
        print("gonzo swift index: ", indices[0])
        /*
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
        */
}
