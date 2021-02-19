public struct Point3<T: SIMDScalar> {

        init() {
                self.xyz = SIMD3()
        }

        init(x: T, y: T, z: T) {
                self.xyz = [x, y, z]
        }

        init(_ point: Point3) {
                self.xyz = [point.x, point.y, point.z]
        }

        init(xyz: (T, T, T)) {
                self.xyz = [xyz.0, xyz.1, xyz.2]
        }

        subscript(index: Int) -> T {
                get {
                        return xyz[index]
                }

                set(newValue) {
                        xyz[index] = newValue
                }
        }

        var x: T {
                get { return xyz[0] }
                set { xyz[0] = newValue }
        }
        var y: T {
                get { return xyz[1] }
                set { xyz[1] = newValue }
        }
        var z: T {
                get { return xyz[2] }
                set { xyz[2] = newValue }
        }

        var xyz: SIMD3<T>
}

extension Point3: CustomStringConvertible {
        public var description: String {
                return "[ \(x) \(y) \(z) ]"
        }
}

func permute<T>(point: Point3<T>, x: Int, y: Int, z: Int) -> Point3<T> {
        return Point3(x: point[x], y: point[y], z: point[z])
}

public typealias Point = Point3<FloatX>

extension Point3: Three where T: FloatingPoint {
        init() {
                self.init(x: 0, y: 0, z: 0)
        }
}

extension Point3 where T: FloatingPoint {
//        init(_ normal: Normal3<T>) {
//                self.init(
//                        x: normal.x,
//                        y: normal.y,
//                        z: normal.z)
//        }
}

let origin = Point()

extension Point3 where T: FloatingPoint & SIMDScalar {

        public static func * (mul: T, point: Point3<T>) -> Point3 {
                return Point3(x: point.x * mul, y: point.y * mul, z: point.z * mul)
        }

        public static func / (point: Point3<T>, divisor: T) -> Point3 {
                return Point3(x: point.x / divisor, y: point.y / divisor, z: point.z / divisor)
        }

        public static func + (left: Point3<T>, right: Point3<T>) -> Point3 {
                return Point3(x: left.x + right.x, y: left.y + right.y, z: left.z + right.z)
        }

        public static func + (left: Point3<T>, right: Vector3<T>) -> Point3 {
                return Point3(x: left.x + right.x, y: left.y + right.y, z: left.z + right.z)
        }

        public static func - (left: Point3<T>, right: Point3<T>) -> Point3<T> {
                return Point3<T>(x: left.x - right.x, y: left.y - right.y, z: left.z - right.z)
        }

        public static func - (left: Point3<T>, right: Point3<T>) -> Vector3<T> {
                return Vector3<T>(x: left.x - right.x, y: left.y - right.y, z: left.z - right.z)
        }
}
