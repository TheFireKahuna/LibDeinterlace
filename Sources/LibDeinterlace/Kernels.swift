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

    /// Encodes `field`'s bwdif, every component in one dispatch, writing the deinterlaced components at their depths into
    /// `out`: planes 0...2 (0 alone for grey), one byte a sample up to 8 bits and two above, little-endian. False when a
    /// picture breaks the layout's alignment or the pipeline cannot be made.
    public func encodeBwdif(_ field: BwdifField, layout: PictureLayout, width: Int, height: Int, into out: PlaneSet,
                            pass: any DeinterlacePass) -> Bool {
        encodeBwdif(field, layout: layout, width: width, height: height, into: out, decided: nil, pass: pass)
    }

    /// `decided`: inverse telecine's record ring and the picture's record, which say whether to deinterlace and whether
    /// the field is intra (iv_window_decide), in place of `field.intra`.
    func encodeBwdif(_ field: BwdifField, layout: PictureLayout, width: Int, height: Int, into out: PlaneSet,
                     decided: (records: MTLBuffer, record: UInt32)?, pass: any DeinterlacePass) -> Bool {
        guard [field.prev, field.cur, field.next].allSatisfy(layout.fits),
              let pipeline = try? pipeline(for: layout, function: "pf_bwdif") else { return false }
        pass.setPipeline(pipeline)
        pass.setBuffer(field.prev.buffer, offset: 0, index: 0)
        pass.setBuffer(decided?.records ?? out.buffer, offset: 0, index: 2)
        pass.setBuffer(field.cur.buffer, offset: 0, index: 5)
        pass.setBuffer(field.next.buffer, offset: 0, index: 6)
        pass.setBuffer(out.buffer, offset: 0, index: 7)
        // PFBwdifs: per component, three PFPictures (offsets, linesizes), then the component's values.
        var arguments = Arguments()
        var grid = (width: 1, height: 1)
        for c in 0..<3 {
            let component = c < layout.components.count ? layout.components[c] : layout.components[0]
            let (cw, ch) = c < layout.components.count ? layout.size(component: c, width: width, height: height) : (0, 0)
            grid = (max(grid.width, (cw + 3) / 4), max(grid.height, (ch + 1) / 2))
            for picture in [field.prev, field.cur, field.next] { arguments.words(picture.offsets + picture.linesizes) }
            arguments.words([UInt32(cw), UInt32(ch), UInt32(min(c, layout.components.count - 1)), field.parity,
                             field.fieldParity, field.intra ? 1 : 0, (1 << component.depth) - 1, out.offsets[c],
                             out.linesizes[c], component.depth > 8 ? 1 : 0, decided == nil ? 0 : 1, decided?.record ?? 0])
        }
        arguments.bind(pass, index: 1)
        // One thread per four columns of a row pair.
        let w = pipeline.threadExecutionWidth
        pass.dispatchThreads(MTLSize(width: grid.width, height: grid.height, depth: layout.components.count),
                             threadsPerThreadgroup: MTLSize(width: w, height: max(1, pipeline.maxTotalThreadsPerThreadgroup / w),
                                                            depth: 1))
        return true
    }

    // MARK: - fieldmatch and decimate

    private func words(_ p: PlaneSet) -> [UInt32] { p.offsets + p.linesizes }

    /// A kernel's constant argument, laid out as its Metal struct: 32-bit words, and 64-bit ones at 8-byte boundaries.
    struct Arguments {
        private(set) var bytes: [UInt8] = []

        mutating func words(_ values: [UInt32]) {
            for v in values { withUnsafeBytes(of: v) { bytes += $0 } }
        }

        mutating func longs(_ values: [Int64]) {
            while bytes.count % 8 != 0 { bytes.append(0) }
            for v in values { withUnsafeBytes(of: v) { bytes += $0 } }
        }

        func bind(_ pass: any DeinterlacePass, index: Int) {
            var padded = bytes
            while padded.count % 8 != 0 { padded.append(0) }
            padded.withUnsafeBytes { pass.setBytes($0, index: index) }
        }
    }

    /// `encode` in a batch of its own, committed and waited for: the probe's question, read on the CPU. A batch whose
    /// encoding fails is still committed (with the dispatches it has), so it is never left open.
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

    private func group(_ pipeline: MTLComputePipelineState) -> MTLSize {
        let w = pipeline.threadExecutionWidth
        return MTLSize(width: w, height: max(1, pipeline.maxTotalThreadsPerThreadgroup / w), depth: 1)
    }

    private func buffer(_ length: Int) -> MTLBuffer? {
        device.makeBuffer(length: max(length, 16), options: .storageModeShared)
    }

    /// calc_combed_score of the weave whose rows of `srcParity` come from `src` and the others from `other`, at the
    /// defaults (luma only, blocks of 16 x 16): the most pixels combed on three consecutive rows inside any block. Waits
    /// for the GPU: the load probe's question, asked a few times before playback.
    public func combScore(src: PlaneSet, other: PlaneSet, srcParity: Int, layout: PictureLayout, width: Int, height: Int, cthresh: Int) -> Int? {
        guard let comb = try? pipeline(for: layout, function: "fm_comb") else { return nil }
        let cellsX = (width + 7) / 8, cellsY = (height + 7) / 8
        guard let cells = buffer(4 * cellsX * cellsY) else { return nil }
        memset(cells.contents(), 0, cells.length)
        // FMComb
        var arguments = Arguments()
        arguments.words(words(src) + words(other) + [UInt32(width), UInt32(height), 0])
        arguments.words([UInt32(srcParity), UInt32(bitPattern: Int32(cthresh)), UInt32(cellsX)])
        let ok = run { pass in
            pass.setPipeline(comb)
            pass.setBuffer(src.buffer, offset: 0, index: 0)
            pass.setBuffer(other.buffer, offset: 0, index: 5)
            pass.setBuffer(cells, offset: 0, index: 8)
            arguments.bind(pass, index: 1)
            pass.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: group(comb))
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

    // MARK: Inverse telecine's dispatches (InverseTelecine)

    /// Zeroes `words` 32-bit words of `buffer`.
    func encodeClear(_ buffer: MTLBuffer, words count: Int, layout: PictureLayout, pass: any DeinterlacePass) -> Bool {
        guard let clear = try? pipeline(for: layout, function: "iv_clear") else { return false }
        pass.setPipeline(clear)
        pass.setBuffer(buffer, offset: 0, index: 8)
        var arguments = Arguments()
        arguments.words([UInt32(count)])
        arguments.bind(pass, index: 1)
        pass.dispatchThreads(MTLSize(width: max(count, 1), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        return true
    }

    /// The sizes of compare_fields' working buffers for pictures of this layout: the diff map (one byte a sample, every
    /// plane) and the accumulations (six words a threadgroup of the shared grid, every plane).
    func fieldMatchSizes(layout: PictureLayout, width: Int, height: Int, hsub: Int) -> (map: Int, partials: Int)? {
        guard let accumulate = try? pipeline(for: layout, function: "fm_accumulate") else { return nil }
        let planes = layout.components.count
        let sizes = (0..<planes).map { layout.size(component: $0, width: width, height: height) }
        let map = sizes.reduce(0) { $0 + $1.width * $1.height }
        let g = group(accumulate)
        let gx = sizes.enumerated().map { c, s in max(s.width - 2 * (c == 0 ? 8 : 8 >> hsub), 1) }.max()!
        let gy = sizes.map { max(($0.height - 3) / 2, 1) }.max()!
        let groups = ((gx + g.width - 1) / g.width) * ((gy + g.height - 1) / g.height)
        return (map, planes * 6 * groups)
    }

    /// compare_fields(match1, match2, field) with mchroma on: every plane's field difference map (build_diff_map), then,
    /// behind a barrier, the accumulations into `partials`. `map` must be zeroed and visible to this pass. `prev` is
    /// select_frame(match1), `next` select_frame(match2); `fbase1` / `fbase2` are get_field_base of each;
    /// `mapFromNextField` says which of build_diff_map's two calls the C makes (its rows fbase, or fbase + 2).
    func encodeFieldMatchCompare(src: PlaneSet, prev: PlaneSet, next: PlaneSet, layout: PictureLayout, width: Int, height: Int,
                                 fbase1: Int, fbase2: Int, mapFromNextField: Bool, hsub: Int, map: MTLBuffer,
                                 partials: MTLBuffer, pass: any DeinterlacePass) -> Bool {
        guard let diffmap = try? pipeline(for: layout, function: "fm_diffmap"),
              let accumulate = try? pipeline(for: layout, function: "fm_accumulate") else { return false }
        let planes = layout.components.count
        let sizes = (0..<planes).map { layout.size(component: $0, width: width, height: height) }
        var mapOffsets: [Int] = [], mapLength = 0
        for s in sizes { mapOffsets.append(mapLength); mapLength += s.width * s.height }
        let g = group(accumulate)
        let gx = sizes.enumerated().map { c, s in max(s.width - 2 * (c == 0 ? 8 : 8 >> hsub), 1) }.max()!
        let gy = sizes.map { max(($0.height - 3) / 2, 1) }.max()!
        let groups = ((gx + g.width - 1) / g.width) * ((gy + g.height - 1) / g.height)
        let base1 = mapFromNextField ? fbase1 + 2 : fbase1, base2 = mapFromNextField ? fbase2 + 2 : fbase2
        // FMDiffMaps
        var maps = Arguments()
        for c in 0..<3 {
            let s = c < planes ? sizes[c] : (width: 0, height: 0)
            maps.words(words(prev) + words(next) + [UInt32(s.width), UInt32(s.height), UInt32(min(c, planes - 1))])
            maps.words([UInt32(base1), UInt32(base2), UInt32(base1), UInt32(c < planes ? mapOffsets[c] : 0)])
        }
        pass.setPipeline(diffmap)
        pass.setBuffer(prev.buffer, offset: 0, index: 0)
        pass.setBuffer(next.buffer, offset: 0, index: 5)
        pass.setBuffer(map, offset: 0, index: 7)
        maps.bind(pass, index: 1)
        pass.dispatchThreads(MTLSize(width: max(sizes.map { $0.width - 2 }.max()!, 1), height: max(sizes.map { ($0.height - 3) / 2 }.max()!, 1),
                                     depth: planes), threadsPerThreadgroup: group(diffmap))
        pass.barrier()
        // FMCompares
        var compares = Arguments()
        for c in 0..<3 {
            let s = c < planes ? sizes[c] : (width: 0, height: 0)
            let startx = c == 0 ? 8 : 8 >> hsub
            compares.words(words(src) + words(prev) + words(next) + [UInt32(s.width), UInt32(s.height), UInt32(min(c, planes - 1))])
            compares.words([UInt32(fbase1), UInt32(fbase2), UInt32(startx), UInt32(c < planes ? mapOffsets[c] : 0),
                            UInt32(min(c, planes - 1) * 6 * groups)])
        }
        pass.setPipeline(accumulate)
        pass.setBuffer(src.buffer, offset: 0, index: 0)
        pass.setBuffer(prev.buffer, offset: 0, index: 5)
        pass.setBuffer(next.buffer, offset: 0, index: 6)
        pass.setBuffer(map, offset: 0, index: 7)
        pass.setBuffer(partials, offset: 0, index: 8)
        compares.bind(pass, index: 1)
        pass.dispatchThreads(MTLSize(width: gx, height: gy, depth: planes), threadsPerThreadgroup: g)
        return true
    }

    /// One picture's luma statistics (iv_luma): |prv - src| and |src - nxt| per 8 x 8 cell, then the comb counts of the
    /// weaves p, c and n, into `stats` (five grids of `cells` words).
    func encodeLumaStatistics(prv: PlaneSet, src: PlaneSet, nxt: PlaneSet, pnParity: Int, layout: PictureLayout, width: Int,
                              height: Int, cthresh: Int, stats: MTLBuffer, pass: any DeinterlacePass) -> Bool {
        guard let luma = try? pipeline(for: layout, function: "iv_luma") else { return false }
        let cellsX = (width + 7) / 8, cellsY = (height + 7) / 8
        var arguments = Arguments()
        arguments.words(words(prv) + words(src) + words(nxt))
        arguments.words([UInt32(width), UInt32(height), UInt32(pnParity), UInt32(bitPattern: Int32(cthresh)),
                         UInt32(cellsX), UInt32(cellsY)])
        pass.setPipeline(luma)
        pass.setBuffer(prv.buffer, offset: 0, index: 0)
        pass.setBuffer(src.buffer, offset: 0, index: 5)
        pass.setBuffer(nxt.buffer, offset: 0, index: 6)
        pass.setBuffer(stats, offset: 0, index: 7)
        pass.setBuffer(stats, offset: 4 * 2 * cellsX * cellsY, index: 8)
        arguments.bind(pass, index: 1)
        pass.dispatchThreadgroups(MTLSize(width: cellsX, height: cellsY, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        return true
    }

    /// One threadgroup of 256, for the decisions.
    private func encodeDecision(_ name: String, layout: PictureLayout, pass: any DeinterlacePass,
                                bind: (any DeinterlacePass) -> Void) -> Bool {
        guard let pipeline = try? pipeline(for: layout, function: name) else { return false }
        pass.setPipeline(pipeline)
        bind(pass)
        pass.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        return true
    }

    /// The match decision (iv_match_decide), with bwdif's window decision for the picture before it when `window` is set.
    func encodeMatchDecide(partials: MTLBuffer, partialCount: Int, stats: MTLBuffer, cells: (x: Int, y: Int), state: MTLBuffer,
                           records: MTLBuffer, record: UInt32, reset: Bool,
                           window: (prev: UInt32, cur: UInt32, prevRepeat: Bool, nextRepeat: Bool, start: Bool)?,
                           emitted: Int64, scthresh: Int64, layout: PictureLayout, pass: any DeinterlacePass) -> Bool {
        var arguments = Arguments()
        arguments.words([UInt32(partialCount), UInt32(cells.x), UInt32(cells.y), record, reset ? 1 : 0, window == nil ? 0 : 1,
                         window?.prev ?? 0, window?.cur ?? 0, window?.prevRepeat == true ? 1 : 0, window?.nextRepeat == true ? 1 : 0,
                         window?.start == true ? 1 : 0, 0])
        arguments.longs([emitted, scthresh])
        return encodeDecision("iv_match_decide", layout: layout, pass: pass) { pass in
            pass.setBuffer(partials, offset: 0, index: 0)
            pass.setBuffer(stats, offset: 0, index: 5)
            pass.setBuffer(state, offset: 0, index: 7)
            pass.setBuffer(records, offset: 0, index: 8)
            arguments.bind(pass, index: 1)
        }
    }

    /// bwdif's window decision for the stream's last picture, standing in for its own successor (iv_window_end).
    func encodeWindowEnd(prev: UInt32, cur: UInt32, prevRepeat: Bool, curRepeat: Bool, start: Bool, state: MTLBuffer,
                         records: MTLBuffer, layout: PictureLayout, pass: any DeinterlacePass) -> Bool {
        guard let end = try? pipeline(for: layout, function: "iv_window_end") else { return false }
        var arguments = Arguments()
        arguments.words([prev, cur, prevRepeat ? 1 : 0, curRepeat ? 1 : 0, start ? 1 : 0])
        pass.setPipeline(end)
        pass.setBuffer(state, offset: 0, index: 7)
        pass.setBuffer(records, offset: 0, index: 8)
        arguments.bind(pass, index: 1)
        pass.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        return true
    }

    /// The weave of the match `record` holds (iv_weave), written planar at 8 bits into `out`.
    func encodeWeave(prv: PlaneSet, src: PlaneSet, nxt: PlaneSet, srcParity: Int, records: MTLBuffer, record: UInt32,
                     layout: PictureLayout, width: Int, height: Int, into out: PlaneSet, pass: any DeinterlacePass) -> Bool {
        guard let weave = try? pipeline(for: layout, function: "iv_weave") else { return false }
        let planes = layout.components.count
        let sizes = (0..<planes).map { layout.size(component: $0, width: width, height: height) }
        var arguments = Arguments()
        arguments.words(words(prv) + words(src) + words(nxt))
        for c in 0..<3 {
            let s = c < planes ? sizes[c] : (width: 0, height: 0)
            arguments.words([UInt32(s.width), UInt32(s.height), UInt32(min(c, planes - 1)), out.offsets[min(c, planes - 1)],
                             out.linesizes[min(c, planes - 1)]])
        }
        arguments.words([UInt32(srcParity), record])
        pass.setPipeline(weave)
        pass.setBuffer(prv.buffer, offset: 0, index: 0)
        pass.setBuffer(records, offset: 0, index: 2)
        pass.setBuffer(src.buffer, offset: 0, index: 5)
        pass.setBuffer(nxt.buffer, offset: 0, index: 6)
        pass.setBuffer(out.buffer, offset: 0, index: 7)
        arguments.bind(pass, index: 1)
        pass.dispatchThreads(MTLSize(width: width, height: height, depth: planes), threadsPerThreadgroup: group(weave))
        return true
    }

    /// calc_diffs' block sums of `a` against `b`, every plane (iv_blocks): one threadgroup per block of half `blockx` x
    /// half `blocky` luma samples, on the luma grid of `blocksX` x `blocksY`.
    func encodeBlockDiffs(_ a: PlaneSet, _ b: PlaneSet, layout: PictureLayout, width: Int, height: Int, blockx: Int, blocky: Int,
                          blocksX: Int, blocksY: Int, into blocks: MTLBuffer, pass: any DeinterlacePass) -> Bool {
        guard let sums = try? pipeline(for: layout, function: "iv_blocks") else { return false }
        let planes = layout.components.count
        let hsub = Int(layout.log2ChromaW), vsub = Int(layout.log2ChromaH)
        var arguments = Arguments()
        arguments.words(words(a) + words(b))
        for c in 0..<3 {
            let s = c < planes ? layout.size(component: c, width: width, height: height) : (width: 0, height: 0)
            let bw = c == 0 ? blockx / 2 : (blockx / 2) >> hsub, bh = c == 0 ? blocky / 2 : (blocky / 2) >> vsub
            arguments.words([UInt32(s.width), UInt32(s.height), UInt32(min(c, planes - 1)), UInt32(bw), UInt32(bh)])
        }
        arguments.words([UInt32(planes), UInt32(blocksX)])
        pass.setPipeline(sums)
        pass.setBuffer(a.buffer, offset: 0, index: 0)
        pass.setBuffer(b.buffer, offset: 0, index: 5)
        pass.setBuffer(blocks, offset: 0, index: 8)
        arguments.bind(pass, index: 1)
        pass.dispatchThreadgroups(MTLSize(width: blocksX, height: blocksY, depth: 1),
                                  threadsPerThreadgroup: MTLSize(width: blockx / 2, height: blocky / 2, depth: 1))
        return true
    }

    /// vf_decimate.c's differences for one picture into its cycle slot, and at the end of a cycle the drop and the
    /// timestamps (iv_decimate). `cycle` is the cycle's pictures' rescaled timestamps (nil: none) when it ends here.
    func encodeDecimate(blocks: MTLBuffer, blocksX: Int, blocksY: Int, slot: Int, diffs: Bool, none: Bool,
                        cycle: (rescaled: [Int64?], full: Bool, result: Int)?, dupthresh: Int64, scthresh: Int64,
                        state: MTLBuffer, results: MTLBuffer, layout: PictureLayout, pass: any DeinterlacePass) -> Bool {
        var arguments = Arguments()
        let rescaled = (cycle?.rescaled ?? []) + Array(repeating: nil, count: 5 - (cycle?.rescaled.count ?? 0))
        arguments.words([UInt32(blocksX), UInt32(blocksY), UInt32(slot), diffs ? 1 : 0, none ? 1 : 0,
                         UInt32(cycle?.rescaled.count ?? 0), cycle?.full == true ? 1 : 0, UInt32(cycle?.result ?? 0)])
        arguments.longs([dupthresh, scthresh] + rescaled.map { $0 ?? 0 })
        arguments.words(rescaled.map { $0 == nil ? 0 : 1 })
        return encodeDecision("iv_decimate", layout: layout, pass: pass) { pass in
            pass.setBuffer(blocks, offset: 0, index: 5)
            pass.setBuffer(state, offset: 0, index: 7)
            pass.setBuffer(results, offset: 0, index: 8)
            arguments.bind(pass, index: 1)
        }
    }
}
