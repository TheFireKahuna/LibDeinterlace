import Foundation
import Metal
import AetherLibavutil

/// A picture's planes inside one Metal buffer: byte offsets and linesizes of planes 0...3.
public struct PlaneSet {
    public var buffer: MTLBuffer
    public var offsets: [UInt32]
    public var linesizes: [UInt32]

    public init(buffer: MTLBuffer, offsets: [UInt32], linesizes: [UInt32]) {
        self.buffer = buffer
        self.offsets = offsets
        self.linesizes = linesizes
    }
}

/// Where one component's samples sit: FFmpeg's AVComponentDescriptor (plane, step and offset in bytes, shift and depth
/// in bits).
public struct ComponentLayout: Hashable, Sendable {
    public var plane: UInt32
    public var step: UInt32
    public var offset: UInt32
    public var shift: UInt32
    public var depth: UInt32

    public init(plane: UInt32, step: UInt32, offset: UInt32, shift: UInt32, depth: UInt32) {
        (self.plane, self.step, self.offset, self.shift, self.depth) = (plane, step, offset, shift, depth)
    }
}

/// How a picture's samples are laid out, the way the kernels read it: one component (grey) or three (Y'CbCr), their
/// layouts, byte order and chroma subsampling. Planar, semi-planar and packed layouts all read the same way, so every
/// stage works on components and a semi-planar or packed picture behaves as its planar equivalent.
public struct PictureLayout: Hashable, Sendable {
    public var components: [ComponentLayout]
    public var bigEndian: Bool
    public var log2ChromaW: UInt32
    public var log2ChromaH: UInt32

    public init(components: [ComponentLayout], bigEndian: Bool, log2ChromaW: UInt32, log2ChromaH: UInt32) {
        self.components = components
        self.bigEndian = bigEndian
        self.log2ChromaW = log2ChromaW
        self.log2ChromaH = log2ChromaH
    }

    /// The layout of `format`, or nil for one the kernels do not read: RGB, palette, bitstream, hardware, float,
    /// components wider than 16 bits.
    public init?(format: AVPixelFormat) {
        guard let d = av_pix_fmt_desc_get(format)?.pointee, d.nb_components > 0 else { return nil }
        let refused = UInt64(AV_PIX_FMT_FLAG_RGB) | UInt64(AV_PIX_FMT_FLAG_PAL) | UInt64(AV_PIX_FMT_FLAG_BITSTREAM)
            | UInt64(AV_PIX_FMT_FLAG_HWACCEL) | UInt64(AV_PIX_FMT_FLAG_FLOAT) | UInt64(AV_PIX_FMT_FLAG_BAYER)
            | UInt64(AV_PIX_FMT_FLAG_XYZ)
        guard d.flags & refused == 0 else { return nil }
        let all = withUnsafeBytes(of: d.comp) { Array($0.bindMemory(to: AVComponentDescriptor.self)) }
        let used = all.prefix(d.nb_components < 3 ? 1 : 3)
        guard used.allSatisfy({ $0.depth >= 8 && $0.depth <= 16 }) else { return nil }
        components = used.map { ComponentLayout(plane: UInt32($0.plane), step: UInt32($0.step), offset: UInt32($0.offset),
                                                shift: UInt32($0.shift), depth: UInt32($0.depth)) }
        bigEndian = d.flags & UInt64(AV_PIX_FMT_FLAG_BE) != 0
        log2ChromaW = UInt32(d.log2_chroma_w)
        log2ChromaH = UInt32(d.log2_chroma_h)
    }

    public var isGray: Bool { components.count == 1 }

    /// Component `c`'s plane size in a `width` x `height` picture.
    public func size(component c: Int, width: Int, height: Int) -> (width: Int, height: Int) {
        guard c > 0 else { return (width, height) }
        let sw = Int(log2ChromaW), sh = Int(log2ChromaH)
        return ((width + (1 << sw) - 1) >> sw, (height + (1 << sh) - 1) >> sh)
    }

    /// Bytes a component load reads, per plane: plane offsets and linesizes must be multiples of it.
    var planeAlignment: [Int] {
        var alignment = [1, 1, 1, 1]
        for c in components {
            let bits = Int(c.shift + c.depth)
            alignment[Int(c.plane)] = max(alignment[Int(c.plane)], bits <= 8 ? 1 : (bits <= 16 ? 2 : 4))
        }
        return alignment
    }

    func fits(_ p: PlaneSet) -> Bool {
        let alignment = planeAlignment
        return (0..<4).allSatisfy { i in
            alignment[i] <= 1 || (p.offsets[i] % UInt32(alignment[i]) == 0 && p.linesizes[i] % UInt32(alignment[i]) == 0)
        }
    }

    /// The function constants, in index order: five per component (plane, step, offset, shift, depth), then the byte order.
    var specialisation: [UInt32] {
        let comps = components + Array(repeating: components[0], count: 3 - components.count)
        return comps.flatMap { [$0.plane, $0.step, $0.offset, $0.shift, $0.depth] } + [bigEndian ? 1 : 0]
    }
}
