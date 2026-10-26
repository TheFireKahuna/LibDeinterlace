import Foundation
import Metal
import AetherLibavutil
@testable import LibDeinterlace

/// Pictures in Metal buffers, for the tests: a frame copied into one, an empty planar picture, and samples read back.
enum Pictures {
    static let device = MTLCreateSystemDefaultDevice()

    /// `frame`'s picture copied into a shared buffer, planes 64-byte aligned.
    static func copy(_ frame: UnsafePointer<AVFrame>) -> PlaneSet? {
        let format = AVPixelFormat(rawValue: frame.pointee.format)
        let (w, h) = (frame.pointee.width, frame.pointee.height)
        let size = av_image_get_buffer_size(format, w, h, 64)
        guard size > 0, let buffer = device?.makeBuffer(length: Int(size) + 64, options: .storageModeShared) else { return nil }
        var data = [UnsafeMutablePointer<UInt8>?](repeating: nil, count: 4)
        var linesize = [Int32](repeating: 0, count: 4)
        let base = buffer.contents().assumingMemoryBound(to: UInt8.self)
        guard av_image_fill_arrays(&data, &linesize, base, format, w, h, 64) >= 0 else { return nil }
        var source = withUnsafeBytes(of: frame.pointee.data) {
            Array($0.bindMemory(to: UnsafeMutablePointer<UInt8>?.self).prefix(4)).map { UnsafePointer($0) }
        }
        var sourceLinesize = withUnsafeBytes(of: frame.pointee.linesize) { Array($0.bindMemory(to: Int32.self).prefix(4)) }
        av_image_copy(&data, &linesize, &source, &sourceLinesize, format, w, h)
        let offsets = data.map { $0.map { UInt32(UnsafeRawPointer($0) - UnsafeRawPointer(base)) } ?? 0 }
        return PlaneSet(buffer: buffer, offsets: offsets, linesizes: linesize.map { UInt32(max($0, 0)) })
    }

    /// An empty planar picture of `layout`'s components at their depths (one byte a sample up to 8 bits, two above).
    static func planar(_ layout: PictureLayout, width: Int, height: Int) -> PlaneSet? {
        var offsets = [UInt32](repeating: 0, count: 4), linesizes = [UInt32](repeating: 0, count: 4)
        var length = 0
        for (c, component) in layout.components.enumerated() {
            let (w, h) = layout.size(component: c, width: width, height: height)
            let linesize = (w * (component.depth > 8 ? 2 : 1) + 63) & ~63
            offsets[c] = UInt32(length)
            linesizes[c] = UInt32(linesize)
            length += (linesize * h + 63) & ~63
        }
        guard let buffer = device?.makeBuffer(length: max(length, 64), options: .storageModeShared) else { return nil }
        return PlaneSet(buffer: buffer, offsets: offsets, linesizes: linesizes)
    }

    /// The samples of a planar picture, component by component, row by row.
    static func samples(_ p: PlaneSet, _ layout: PictureLayout, width: Int, height: Int) -> [[Int]] {
        let bytes = UnsafeRawPointer(p.buffer.contents())
        return layout.components.enumerated().map { c, component in
            let (w, h) = layout.size(component: c, width: width, height: height)
            let wide = component.depth > 8
            return (0..<h).flatMap { y in
                (0..<w).map { x -> Int in
                    let at = Int(p.offsets[c]) + y * Int(p.linesizes[c])
                    return wide ? Int(bytes.load(fromByteOffset: at + 2 * x, as: UInt16.self)) : Int(bytes.load(fromByteOffset: at + x, as: UInt8.self))
                }
            }
        }
    }
}
