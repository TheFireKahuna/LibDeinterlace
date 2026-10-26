// Drives the transcriptions of FFmpeg's libavfilter (bwdifdsp.c, vf_fieldmatch.c, vf_decimate.c) in ShaderSource.swift;
// calc_combed_score's block maximum is computed here. Copyright notices: ShaderSource.swift. LGPL-2.1-or-later; see
// LICENSE.

import Foundation
import Metal
import os

/// The library's GPU work, submitted through the caller's ``DeinterlaceSubmission``. Each call runs kernels over
/// pictures of one ``PictureLayout`` and returns exact integer results (sums totalled in 64 bits), or encodes into a
/// caller's pass (``encodeBwdif(_:layout:width:height:into:pass:)``).
public final class DeinterlaceKernels: @unchecked Sendable {

    public let device: MTLDevice
    let submission: any DeinterlaceSubmission

    private static let libraries = OSAllocatedUnfairLock<[ObjectIdentifier: MTLLibrary]>(initialState: [:])
    private static let pipelines = OSAllocatedUnfairLock<[PipelineKey: MTLComputePipelineState]>(initialState: [:])

    private struct PipelineKey: Hashable {
        let device: ObjectIdentifier
        let function: String
        let constants: [UInt32]
    }

    public init(submission: any DeinterlaceSubmission) {
        self.submission = submission
        device = submission.device
    }

    private static func library(for device: MTLDevice) throws -> MTLLibrary {
        if let cached = libraries.withLock({ $0[ObjectIdentifier(device)] }) { return cached }
        let options = MTLCompileOptions()
        if #available(macOS 15, iOS 18, tvOS 18, visionOS 2, *) { options.mathMode = .safe } else { options.fastMathEnabled = false }
        let made = try device.makeLibrary(source: ShaderSource.source, options: options)
        libraries.withLock { $0[ObjectIdentifier(device)] = made }
        return made
    }

    /// The pipeline of `function` specialised for `layout`, compiled on first use and shared in the process.
    public func pipeline(for layout: PictureLayout, function name: String) throws -> MTLComputePipelineState {
        let key = PipelineKey(device: ObjectIdentifier(device), function: name, constants: layout.specialisation)
        if let ready = Self.pipelines.withLock({ $0[key] }) { return ready }
        let values = MTLFunctionConstantValues()
        for (index, value) in layout.specialisation.enumerated() {
            if index == 15 {
                var flag = value != 0
                values.setConstantValue(&flag, type: .bool, index: index)
            } else {
                var word = value
                values.setConstantValue(&word, type: .uint, index: index)
            }
        }
        let function = try Self.library(for: device).makeFunction(name: name, constantValues: values)
        let made = try device.makeComputePipelineState(function: function)
        Self.pipelines.withLock { $0[key] = made }
        return made
    }

    // MARK: - bwdif

    /// One output field: FFmpeg's previous, current and next pictures, and vf_bwdif's parities.
    public struct BwdifField {
        public var prev: PlaneSet
        public var cur: PlaneSet
        public var next: PlaneSet
        /// Rows with `(y ^ parity) & 1` are interpolated; the others are the current field's.
        public var parity: UInt32
        /// vf_bwdif's `parity ^ tff`: which neighbours stand in for the missing field.
        public var fieldParity: UInt32
        /// The stream's first or last field: interpolated from the current field alone.
        public var intra: Bool

        public init(prev: PlaneSet, cur: PlaneSet, next: PlaneSet, parity: UInt32, fieldParity: UInt32, intra: Bool) {
            (self.prev, self.cur, self.next) = (prev, cur, next)
            (self.parity, self.fieldParity, self.intra) = (parity, fieldParity, intra)
        }
    }

    /// Encodes `field`'s bwdif dispatches, one per component, writing the deinterlaced components at their depths into
    /// `out`: planes 0...2 (0 alone for grey), one byte a sample up to 8 bits and two above, little-endian. False when a
    /// picture breaks the layout's alignment or the pipeline cannot be made.
    public func encodeBwdif(_ field: BwdifField, layout: PictureLayout, width: Int, height: Int, into out: PlaneSet,
                            pass: any DeinterlacePass) -> Bool {
        guard [field.prev, field.cur, field.next].allSatisfy(layout.fits),
              let pipeline = try? pipeline(for: layout, function: "pf_bwdif") else { return false }
        pass.setPipeline(pipeline)
        pass.setBuffer(field.prev.buffer, offset: 0, index: 0)
        pass.setBuffer(field.cur.buffer, offset: 0, index: 5)
        pass.setBuffer(field.next.buffer, offset: 0, index: 6)
        pass.setBuffer(out.buffer, offset: 0, index: 7)
        let w = pipeline.threadExecutionWidth
        let group = MTLSize(width: w, height: max(1, pipeline.maxTotalThreadsPerThreadgroup / w), depth: 1)
        for (c, component) in layout.components.enumerated() {
            let (cw, ch) = layout.size(component: c, width: width, height: height)
            // PFBwdif: three PFPictures (offsets, linesizes), then the component's values.
            var words: [UInt32] = []
            for picture in [field.prev, field.cur, field.next] { words += picture.offsets + picture.linesizes }
            words += [UInt32(cw), UInt32(ch), UInt32(c), field.parity, field.fieldParity, field.intra ? 1 : 0,
                      (1 << component.depth) - 1, out.offsets[c], out.linesizes[c], component.depth > 8 ? 1 : 0]
            words.withUnsafeBytes { pass.setBytes($0, index: 1) }
            // One thread per four columns of a row pair.
            pass.dispatchThreads(MTLSize(width: (cw + 3) / 4, height: (ch + 1) / 2, depth: 1), threadsPerThreadgroup: group)
        }
        return true
    }

    /// ``encodeBwdif(_:layout:width:height:into:pass:)`` in a batch of its own, waited for.
    public func deinterlace(_ field: BwdifField, layout: PictureLayout, width: Int, height: Int, into out: PlaneSet) -> Bool {
        run { encodeBwdif(field, layout: layout, width: width, height: height, into: out, pass: $0) }
    }

    // MARK: - fieldmatch and decimate


    /// compare_fields' six accumulations.
    public struct FieldAccumulations: Equatable {
        var pc: UInt64 = 0, pm: UInt64 = 0, pml: UInt64 = 0
        var nc: UInt64 = 0, nm: UInt64 = 0, nml: UInt64 = 0
    }

    private func words(_ p: PlaneSet) -> [UInt32] { p.offsets + p.linesizes }

    /// `encode` in a batch of its own, committed and waited for: the calls that return results read them on the CPU.
    /// A batch whose encoding fails is still committed (with the dispatches it has), so it is never left open.
    private func run(_ encode: (any DeinterlacePass) -> Bool) -> Bool {
        guard let batch = submission.makeBatch() else { return false }
        guard let pass = batch.makePass() else {
            commitAndWait(batch)
            return false
        }
        let encoded = encode(pass)
        return commitAndWait(batch) && encoded
    }

    @discardableResult
    private func commitAndWait(_ batch: any DeinterlaceBatch) -> Bool {
        let done = DispatchSemaphore(value: 0)
        let completed = OSAllocatedUnfairLock(initialState: false)
        batch.commit { failure in
            completed.withLock { $0 = failure == nil }
            done.signal()
        }
        done.wait()
        return completed.withLock { $0 }
    }

    private func dispatch(_ pass: any DeinterlacePass, _ pipeline: MTLComputePipelineState, width: Int, height: Int,
                          group: MTLSize? = nil) -> MTLSize {
        let w = pipeline.threadExecutionWidth
        let g = group ?? MTLSize(width: w, height: max(1, pipeline.maxTotalThreadsPerThreadgroup / w), depth: 1)
        pass.dispatchThreads(MTLSize(width: max(width, 1), height: max(height, 1), depth: 1), threadsPerThreadgroup: g)
        return MTLSize(width: (max(width, 1) + g.width - 1) / g.width, height: (max(height, 1) + g.height - 1) / g.height, depth: 1)
    }

    private func buffer(_ length: Int) -> MTLBuffer? {
        device.makeBuffer(length: max(length, 16), options: .storageModeShared)
    }

    /// compare_fields(match1, match2, field) with mchroma on: per plane, the field difference map (build_diff_map),
    /// then the accumulations. `prev` is select_frame(match1), `next` select_frame(match2); `fbase1` / `fbase2` are
    /// get_field_base of each; `prevOdd` says which of build_diff_map's two calls the C makes (its rows fbase, or
    /// fbase + 2).
    public func fieldMatchCompare(src: PlaneSet, prev: PlaneSet, next: PlaneSet, layout: PictureLayout, width: Int, height: Int,
                           fbase1: Int, fbase2: Int, mapFromNextField: Bool, hsub: Int) -> FieldAccumulations? {
        guard let diffmap = try? pipeline(for: layout, function: "fm_diffmap"),
              let accumulate = try? pipeline(for: layout, function: "fm_accumulate") else { return nil }
        let sizes = (0..<layout.components.count).map { layout.size(component: $0, width: width, height: height) }
        var mapOffsets: [Int] = [], mapLength = 0
        for s in sizes { mapOffsets.append(mapLength); mapLength += s.width * s.height }
        guard let map = buffer(mapLength) else { return nil }
        memset(map.contents(), 0, map.length)
        // Six partials a threadgroup, per plane; sized from the accumulate grid.
        let group = MTLSize(width: accumulate.threadExecutionWidth,
                            height: max(1, accumulate.maxTotalThreadsPerThreadgroup / accumulate.threadExecutionWidth), depth: 1)
        var partialOffsets: [Int] = [], partialLength = 0
        for (c, s) in sizes.enumerated() {
            let startx = c == 0 ? 8 : 8 >> hsub
            let gx = max(s.width - 2 * startx, 0), gy = max((s.height - 3) / 2, 0)
            partialOffsets.append(partialLength)
            partialLength += 6 * ((max(gx, 1) + group.width - 1) / group.width) * ((max(gy, 1) + group.height - 1) / group.height)
        }
        guard let partials = buffer(4 * partialLength) else { return nil }
        memset(partials.contents(), 0, partials.length)
        let ok = run { pass in
            for (c, s) in sizes.enumerated() {
                let plane = [UInt32(s.width), UInt32(s.height), UInt32(c)]
                let mapStart = mapFromNextField ? fbase1 + 2 : fbase1
                let base1 = mapFromNextField ? fbase1 + 2 : fbase1, base2 = mapFromNextField ? fbase2 + 2 : fbase2
                // FMDiffMap
                var m = words(prev) + words(next) + plane
                m += [UInt32(base1), UInt32(base2), UInt32(mapStart), UInt32(mapOffsets[c])]
                pass.setPipeline(diffmap)
                pass.setBuffer(prev.buffer, offset: 0, index: 0)
                pass.setBuffer(next.buffer, offset: 0, index: 5)
                pass.setBuffer(map, offset: 0, index: 7)
                m.withUnsafeBytes { pass.setBytes($0, index: 1) }
                _ = dispatch(pass, diffmap, width: s.width - 2, height: (s.height - 3) / 2)
            }
            pass.barrier()
            for (c, s) in sizes.enumerated() {
                let startx = c == 0 ? 8 : 8 >> hsub
                // FMCompare
                var w = words(src) + words(prev) + words(next) + [UInt32(s.width), UInt32(s.height), UInt32(c)]
                w += [UInt32(fbase1), UInt32(fbase2), UInt32(startx), UInt32(mapOffsets[c]), UInt32(partialOffsets[c])]
                pass.setPipeline(accumulate)
                pass.setBuffer(src.buffer, offset: 0, index: 0)
                pass.setBuffer(prev.buffer, offset: 0, index: 5)
                pass.setBuffer(next.buffer, offset: 0, index: 6)
                pass.setBuffer(map, offset: 0, index: 7)
                pass.setBuffer(partials, offset: 0, index: 8)
                w.withUnsafeBytes { pass.setBytes($0, index: 1) }
                _ = dispatch(pass, accumulate, width: s.width - 2 * startx, height: (s.height - 3) / 2, group: group)
            }
            return true
        }
        guard ok else { return nil }
        let values = UnsafeBufferPointer(start: partials.contents().assumingMemoryBound(to: UInt32.self), count: partialLength)
        var sums = [UInt64](repeating: 0, count: 6)
        for (i, v) in values.enumerated() { sums[i % 6] += UInt64(v) }
        return FieldAccumulations(pc: sums[0], pm: sums[1], pml: sums[2], nc: sums[3], nm: sums[4], nml: sums[5])
    }

    /// calc_combed_score of the weave whose rows of `srcParity` come from `src` and the others from `other`, at the
    /// defaults (luma only, blocks of 16 x 16): the most pixels combed on three consecutive rows inside any block.
    public func combScore(src: PlaneSet, other: PlaneSet, srcParity: Int, layout: PictureLayout, width: Int, height: Int, cthresh: Int) -> Int? {
        guard let comb = try? pipeline(for: layout, function: "fm_comb") else { return nil }
        let cellsX = (width + 7) / 8, cellsY = (height + 7) / 8
        guard let cells = buffer(4 * cellsX * cellsY) else { return nil }
        memset(cells.contents(), 0, cells.length)
        // FMComb
        var w = words(src) + words(other) + [UInt32(width), UInt32(height), 0]
        w += [UInt32(srcParity), UInt32(bitPattern: Int32(cthresh)), UInt32(cellsX)]
        let ok = run { pass in
            pass.setPipeline(comb)
            pass.setBuffer(src.buffer, offset: 0, index: 0)
            pass.setBuffer(other.buffer, offset: 0, index: 5)
            pass.setBuffer(cells, offset: 0, index: 8)
            w.withUnsafeBytes { pass.setBytes($0, index: 1) }
            _ = dispatch(pass, comb, width: width, height: height - 2)
            return true
        }
        guard ok else { return nil }
        let counts = UnsafeBufferPointer(start: cells.contents().assumingMemoryBound(to: UInt32.self), count: cellsX * cellsY)
        func cell(_ y: Int, _ x: Int) -> Int {
            y < 0 || x < 0 || y >= cellsY || x >= cellsX ? 0 : Int(counts[y * cellsX + x])
        }
        // The C's four interleaved block grids (offsets 0 and half a block on each axis) are every 2 x 2 window of cells.
        var best = 0
        for y in -1..<cellsY {
            for x in -1..<cellsX {
                best = max(best, cell(y, x) + cell(y, x + 1) + cell(y + 1, x) + cell(y + 1, x + 1))
            }
        }
        return best
    }

    /// luma_abs_diff.
    public func lumaAbsDiff(_ a: PlaneSet, _ b: PlaneSet, layout: PictureLayout, width: Int, height: Int) -> Int64? {
        guard let diff = try? pipeline(for: layout, function: "fm_absdiff") else { return nil }
        let group = MTLSize(width: diff.threadExecutionWidth, height: max(1, diff.maxTotalThreadsPerThreadgroup / diff.threadExecutionWidth), depth: 1)
        let groups = ((width + group.width - 1) / group.width) * ((height + group.height - 1) / group.height)
        guard let out = buffer(4 * groups) else { return nil }
        memset(out.contents(), 0, out.length)
        let w = words(a) + words(b) + [UInt32(width), UInt32(height), 0, 0, 0, 0]
        let ok = run { pass in
            pass.setPipeline(diff)
            pass.setBuffer(a.buffer, offset: 0, index: 0)
            pass.setBuffer(b.buffer, offset: 0, index: 5)
            pass.setBuffer(out, offset: 0, index: 8)
            w.withUnsafeBytes { pass.setBytes($0, index: 1) }
            _ = dispatch(pass, diff, width: width, height: height, group: group)
            return true
        }
        guard ok else { return nil }
        return UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: UInt32.self), count: groups).reduce(Int64(0)) { $0 + Int64($1) }
    }

    /// calc_diffs' block sums: |a - b| over every plane (chroma on), per block of half `blockx` x half `blocky` luma
    /// samples (the chroma blocks subsampled with the planes), on the luma grid of `blocksX` x `blocksY`.
    public func blockDiffs(_ a: PlaneSet, _ b: PlaneSet, layout: PictureLayout, width: Int, height: Int, blockx: Int, blocky: Int,
                    blocksX: Int, blocksY: Int) -> [Int64]? {
        guard let diff = try? pipeline(for: layout, function: "fm_absdiff") else { return nil }
        guard let out = buffer(4 * blocksX * blocksY) else { return nil }
        memset(out.contents(), 0, out.length)
        let sizes = (0..<layout.components.count).map { layout.size(component: $0, width: width, height: height) }
        let hsub = Int(layout.log2ChromaW), vsub = Int(layout.log2ChromaH)
        let ok = run { pass in
            pass.setPipeline(diff)
            pass.setBuffer(a.buffer, offset: 0, index: 0)
            pass.setBuffer(b.buffer, offset: 0, index: 5)
            pass.setBuffer(out, offset: 0, index: 8)
            for (c, s) in sizes.enumerated() {
                let bw = c == 0 ? blockx / 2 : (blockx / 2) >> hsub, bh = c == 0 ? blocky / 2 : (blocky / 2) >> vsub
                let w = words(a) + words(b) + [UInt32(s.width), UInt32(s.height), UInt32(c), UInt32(bw), UInt32(bh), UInt32(blocksX)]
                w.withUnsafeBytes { pass.setBytes($0, index: 1) }
                _ = dispatch(pass, diff, width: s.width, height: s.height)
            }
            return true
        }
        guard ok else { return nil }
        return UnsafeBufferPointer(start: out.contents().assumingMemoryBound(to: UInt32.self), count: blocksX * blocksY).map { Int64($0) }
    }

    /// create_weave_frame's pixels: rows of `srcParity` from `src`, the others from `other`, written into `out` with the
    /// planar 8-bit layout `outPlaneSet` describes (its buffer is `out`).
    public func weave(src: PlaneSet, other: PlaneSet, srcParity: Int, layout: PictureLayout, width: Int, height: Int, into out: PlaneSet) -> Bool {
        guard let pipeline = try? pipeline(for: layout, function: "fm_weave_out") else { return false }
        let sizes = (0..<layout.components.count).map { layout.size(component: $0, width: width, height: height) }
        return run { pass in
            pass.setPipeline(pipeline)
            pass.setBuffer(src.buffer, offset: 0, index: 0)
            pass.setBuffer(other.buffer, offset: 0, index: 5)
            pass.setBuffer(out.buffer, offset: 0, index: 7)
            for (c, s) in sizes.enumerated() {
                let w = words(src) + words(other) + [UInt32(s.width), UInt32(s.height), UInt32(c), UInt32(srcParity),
                                                     out.offsets[c], out.linesizes[c]]
                w.withUnsafeBytes { pass.setBytes($0, index: 1) }
                _ = dispatch(pass, pipeline, width: s.width, height: s.height)
            }
            return true
        }
    }
}
