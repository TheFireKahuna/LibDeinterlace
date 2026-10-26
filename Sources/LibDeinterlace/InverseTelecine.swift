// FieldMatcher and Decimator are transcriptions of FFmpeg's libavfilter/vf_fieldmatch.c and libavfilter/vf_decimate.c:
//   Copyright (c) 2012 Fredrik Mellbin
//   Copyright (c) 2013 Clément Bœsch
// This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option)
// any later version. See LICENSE.

import Foundation
import AetherLibavutil

/// Hard telecine and progressive-segmented film back to their film frames (docs/software-route-gpu.md 6.2, 6.3), as
/// FFmpeg's documented recipe does it: `fieldmatch,bwdif=mode=send_frame:deint=interlaced,decimate`, every option at its
/// default. Every decision runs on the GPU in the caller's pass, with the per-pixel work it decides on, so the chain
/// never waits for the GPU: fieldmatch's (vf_fieldmatch.c `filter_frame`, `compare_fields`, `checkmm`), bwdif's window
/// at frame rate (yadif_common.c), and decimate's (vf_decimate.c `filter_frame`, `calc_diffs`), each picture's verdict in
/// a record and the chain's running state in a buffer of its own. What the CPU keeps is what does not depend on the
/// pictures: the three sliding windows, the counts, the timestamps the C carries forward. A picture's fate (kept, and
/// at what timestamp, or dropped) is read once its pass has completed (``resolve(_:)``).
///
/// Scope is fieldmatch's: 8-bit planar 4:4:4, 4:2:2, 4:2:0, 4:1:1 and 4:1:0 (``takes(format:width:height:)``).
/// Not thread-safe: one caller pushes; ``resolve(_:)`` may be called from a completion.
public final class InverseTelecine {

    /// The formats vf_fieldmatch.c takes without a clean source.
    public static let formats: Set<Int32> = Set([AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV420P,
                                          AV_PIX_FMT_YUV411P, AV_PIX_FMT_YUV410P].map(\.rawValue))

    /// Whether a stream of these pictures can be matched: a format fieldmatch takes, even plane heights (every read the
    /// C makes then stays inside the plane), planes big enough for its margins.
    public static func takes(format: AVPixelFormat, width: Int, height: Int) -> Bool {
        guard formats.contains(format.rawValue), let d = av_pix_fmt_desc_get(format)?.pointee else { return false }
        let ch = (height + (1 << Int(d.log2_chroma_h)) - 1) >> Int(d.log2_chroma_h)
        return height % 2 == 0 && ch % 2 == 0 && ch >= 8 && width >= 32
    }

    /// Makes an empty picture of the stream's format and size (a slab of the caller's), with what returns it.
    public typealias PictureMaker = (AVPixelFormat, Int, Int) -> (planes: PlaneSet, release: () -> Void)?

    /// A film picture the chain puts out: written by the work of the pass it came from. Whether it goes out, and at what
    /// timestamp, is ``resolve(_:)``'s once that work has completed.
    public struct Output {
        public let picture: BwdifWindow.Picture
        let fixedPTS: Int64?
        let result: Int?
        /// Where the GPU writes whether decimate keeps the picture: a 64-bit word, 1 kept and 0 dropped, written by the
        /// pass that puts the picture out, for work later in its batch to be dispatched on (the picture's finish). Nil
        /// when nothing drops it.
        public let kept: (buffer: MTLBuffer, offset: Int)?
    }

    /// What one ``push(_:into:)`` encoded: the pictures it puts out, and every picture its work reads or writes, which
    /// the caller holds until the pass has completed.
    public struct Pushed {
        public var outputs: [Output] = []
        public var touched: [BwdifWindow.Picture] = []
    }

    /// How many passes may be in flight at once: each takes the next of this many sets of working buffers.
    public static let passesInFlight = 8

    let kernels: DeinterlaceKernels
    let makeBuffer: PictureMaker
    let layout: PictureLayout
    let format: AVPixelFormat
    let width: Int
    let height: Int
    private let decimates: Bool

    /// The time base of every output timestamp: decimate's (the input frame period times 5/4), or bwdif's (half the
    /// input's) without decimation.
    public let outputTimeBase: AVRational
    /// bwdif's output base and the factor that takes an input timestamp onto it.
    private let windowTimeBase: AVRational
    private let windowMultiplier: Int64

    // The GPU's side: the chain's running state, a record per matched picture (a ring), decimate's results (a ring of
    // cycles), and the working buffers of each pass in flight.
    private let state: MTLBuffer
    private let records: MTLBuffer
    private let results: MTLBuffer
    /// Updated once a push's statistics are written (the chain's pass waits for them), and once its chain work is (the
    /// next push's chain waits, and so does any pass reading the pictures it puts out: ``fence``).
    private let statistics: MTLFence
    /// Updated at the end of every push's chain work: a caller's pass reading a push's outputs (its finish) waits on it.
    public let fence: MTLFence
    private static let recordRing = 64
    private static let resultRing = 64 * Decimator.cycle
    private var nextRecord = 0
    private var nextResult = 0
    private struct Scratch {
        let map: MTLBuffer
        let partials: MTLBuffer
        let stats: MTLBuffer
        let blocks: MTLBuffer
        /// Updated once the chain work of the push that last took this set has read it: the next push to take it
        /// writes only after that, however far its statistics run ahead.
        let released: MTLFence
    }
    private let scratch: [Scratch]
    private var nextScratch = 0
    private let partialCount: Int
    private let cells: (x: Int, y: Int)
    private var resetPending = true

    // fieldmatch's sliding window (vf_fieldmatch.c at its defaults: order=auto, mode=pc_n, field=auto, mchroma=1, no
    // exclusion band, scthresh=12, combmatch=sc, cthresh=9, chroma=0, blockx=blocky=16, combpel=80).
    public static let combpel = 80
    public static let cthresh = 9
    private let scthresh: Int64
    private var prv: BwdifWindow.Picture?
    private var src: BwdifWindow.Picture?
    private var nxt: BwdifWindow.Picture?
    /// The output link's frame_count_in: pictures put out so far.
    private var emitted: Int64 = 0

    /// A matched picture and its record.
    private struct Matched {
        let picture: BwdifWindow.Picture
        let record: UInt32
        /// The source's field order: the weave's, when it is still combed.
        let field: Int
    }
    // bwdif's window at frame rate (yadif_common.c, deint=interlaced): the first picture stands in for its own
    // predecessor, the last for its own successor.
    private var windowPrev: Matched?
    private var windowCur: Matched?
    private var windowNext: Matched?
    private var windowStartPending = false
    private var windowDone = false

    // decimate (vf_decimate.c at its defaults: cycle=5, dupthresh=1.1, scthresh=15, blockx=blocky=32, chroma=1, mixed=0).
    private enum Decimator {
        static let cycle = 5
        static let blockx = 32
        static let blocky = 32
    }
    private let dupthresh: Int64
    private let decimateSCThresh: Int64
    private let nxblocks: Int
    private let nyblocks: Int
    private var queue: [(picture: BwdifWindow.Picture, pts: Int64)] = []
    private var last: BwdifWindow.Picture?

    /// `timeBase` is the decoded pictures'; `frameRate` the carriage rate (decimate needs a constant one). Without
    /// `decimates` (2:2, progressive-segmented film) every matched picture goes out.
    public init?(kernels: DeinterlaceKernels, makePicture: @escaping PictureMaker, format: AVPixelFormat, width: Int,
                 height: Int, timeBase: AVRational, frameRate: AVRational, decimates: Bool = true) {
        guard Self.takes(format: format, width: width, height: height), frameRate.num > 0, frameRate.den > 0,
              let layout = PictureLayout(format: format),
              let sizes = kernels.fieldMatchSizes(layout: layout, width: width, height: height, hsub: Int(layout.log2ChromaW))
        else { return nil }
        self.kernels = kernels
        self.makeBuffer = makePicture
        self.layout = layout
        self.format = format
        self.width = width
        self.height = height
        self.decimates = decimates
        (windowTimeBase, windowMultiplier) = BwdifWindow.outputTimeBase(for: timeBase)
        // nondec_tb = 1 / fps; dec_tb = nondec_tb * cycle / (cycle - 1)
        outputTimeBase = decimates
            ? av_mul_q(av_inv_q(frameRate), AVRational(num: Int32(Decimator.cycle), den: Int32(Decimator.cycle - 1)))
            : windowTimeBase
        // (int64_t)((w * h * 255.0 * fm->scthresh_flt) / 100.0)
        scthresh = Int64((Double(width * height) * 255.0 * 12.0) / 100.0)
        dupthresh = Int64((Double(255 * Int64(Decimator.blockx) * Int64(Decimator.blocky)) * 1.1) / 100)
        decimateSCThresh = Int64((Double(255 * Int64(width) * Int64(height)) * 15.0) / 100)
        nxblocks = (width + Decimator.blockx / 2 - 1) / (Decimator.blockx / 2)
        nyblocks = (height + Decimator.blocky / 2 - 1) / (Decimator.blocky / 2)
        cells = ((width + 7) / 8, (height + 7) / 8)
        partialCount = sizes.partials
        let device = kernels.device
        guard let state = device.makeBuffer(length: 8 * State.words, options: .storageModeShared),
              let records = device.makeBuffer(length: 16 * Self.recordRing, options: .storageModeShared),
              let results = device.makeBuffer(length: 16 * Self.resultRing, options: .storageModeShared) else { return nil }
        guard let statistics = device.makeFence(), let fence = device.makeFence() else { return nil }
        memset(state.contents(), 0, state.length)
        (self.state, self.records, self.results) = (state, records, results)
        (self.statistics, self.fence) = (statistics, fence)
        var sets: [Scratch] = []
        for _ in 0..<Self.passesInFlight {
            // Whole words, as a fill takes.
            guard let map = device.makeBuffer(length: max((sizes.map + 3) & ~3, 16), options: .storageModePrivate),
                  let partials = device.makeBuffer(length: 4 * max(sizes.partials, 4), options: .storageModePrivate),
                  let stats = device.makeBuffer(length: 4 * DeinterlaceKernels.statisticsLength(cells: cells.x * cells.y),
                                                options: .storageModePrivate),
                  let blocks = device.makeBuffer(length: 4 * nxblocks * nyblocks, options: .storageModePrivate),
                  let released = device.makeFence()
            else { return nil }
            sets.append(Scratch(map: map, partials: partials, stats: stats, blocks: blocks, released: released))
        }
        scratch = sets
    }

    /// The state buffer's 64-bit words (the shader's IV_*).
    private enum State {
        static let words = 19
        static let cycles = 7
        static let cyclesWithDuplicate = 8
    }

    /// decimate's full cycles, and those with a picture under the duplicate threshold (the load probe's evidence of 3:2):
    /// counted on the GPU, read once the passes pushed so far have completed.
    public var cycles: Int { Int(state.contents().load(fromByteOffset: 8 * State.cycles, as: Int64.self)) }
    public var cyclesWithDuplicate: Int {
        Int(state.contents().load(fromByteOffset: 8 * State.cyclesWithDuplicate, as: Int64.self))
    }

    /// Whether `output` goes out, and at what timestamp on ``outputTimeBase`` (AV_NOPTS_VALUE when it has none): nil
    /// when decimate dropped it. Valid once the pass that put it out has completed.
    public func resolve(_ output: Output) -> Int64? {
        if let fixed = output.fixedPTS { return fixed }
        guard let index = output.result else { return nil }
        let slot = results.contents().advanced(by: 16 * index)
        guard slot.load(fromByteOffset: 8, as: Int64.self) != 0 else { return nil }
        return slot.load(as: Int64.self)
    }

    /// Feeds a decoded picture (nil: end of stream), encoding its work into two passes of `batch`, and returns the film
    /// pictures it releases. The first reads only decoded pictures, so it may run beside earlier batches' work (once
    /// the push that last used its working buffers has read them); the second waits for it and for the previous push's
    /// chain work (``fence``), and updates ``fence`` when done. Pushes are committed in order to one queue.
    public func push(_ picture: BwdifWindow.Picture?, into batch: any DeinterlaceBatch) -> Pushed {
        var pushed = Pushed()
        let work = scratch[nextScratch]
        nextScratch = (nextScratch + 1) % scratch.count
        let compared = window3(picture)
        if let compared {
            pushed.touched += [compared.prv, compared.src, compared.nxt]
            guard let first = batch.makePass() else { return pushed }
            first.waitForFence(work.released)
            guard statistics(compared, work: work, pass: first) else { return pushed }
            first.updateFence(statistics)
        }
        guard let pass = batch.makePass() else { return pushed }
        pass.waitForFence(statistics)
        pass.waitForFence(fence)
        defer {
            pass.updateFence(work.released)
            pass.updateFence(fence)
        }
        if let compared, let matched = match(compared, work: work, pass: pass, pushed: &pushed),
           let out = window(matched, finishing: false, pass: pass, pushed: &pushed) {
            decimate(out, work: work, pass: pass, pushed: &pushed)
        }
        // End of stream (ff_yadif_request_frame at EOF): the last picture, its own successor standing in for the next.
        if picture == nil, !windowDone, let windowCur, let windowNext {
            windowDone = true
            let successor = windowNext.picture.clone(pts: Self.extrapolated(windowCur.picture.pts, windowNext.picture.pts))
            if let out = window(Matched(picture: successor, record: windowNext.record, field: windowNext.field),
                                finishing: true, pass: pass, pushed: &pushed) {
                decimate(out, work: work, pass: pass, pushed: &pushed)
            }
        }
        if picture == nil, decimates, !queue.isEmpty {
            // decimate's end of stream: the partial cycle, nothing dropped.
            pass.barrier()
            endCycle(full: false, work: work, slot: 0, diffs: false, none: true, pass: pass, pushed: &pushed)
        }
        return pushed
    }

    /// Drops everything held (a seek): the chain starts over, its GPU state with the next picture's pass.
    public func reset() {
        prv = nil
        src = nil
        nxt = nil
        emitted = 0
        windowPrev = nil
        windowCur = nil
        windowNext = nil
        windowStartPending = false
        windowDone = false
        queue.removeAll()
        last = nil
        resetPending = true
    }

    private static let noPTS = Int64.min

    private static func extrapolated(_ cur: Int64, _ next: Int64) -> Int64 {
        next == noPTS || cur == noPTS ? noPTS : next * 2 - cur
    }

    /// An empty picture in a slab with `props`'s properties, flagged progressive: the weaves and bwdif's outputs.
    private func makePicture(props: UnsafePointer<AVFrame>, pts: Int64) -> (BwdifWindow.Picture, PlaneSet)? {
        guard let made = makeBuffer(format, width, height), let frame = av_frame_alloc() else { return nil }
        _ = av_frame_copy_props(frame, props)
        frame.pointee.format = format.rawValue
        frame.pointee.width = Int32(width)
        frame.pointee.height = Int32(height)
        frame.pointee.pts = pts
        frame.pointee.flags &= ~(AV_FRAME_FLAG_INTERLACED | AV_FRAME_FLAG_TOP_FIELD_FIRST)
        let storage = BwdifWindow.Storage(frame: frame, planes: made.planes, release: made.release)
        return (BwdifWindow.Picture(storage: storage, pts: pts), made.planes)
    }

    // MARK: fieldmatch

    /// fieldmatch's three pictures once `picture` is in (SLIDING_FRAME_WINDOW; nil flushes the last one), and the
    /// source's field order: nil until the window has a picture to match.
    private func window3(_ picture: BwdifWindow.Picture?) -> Compared? {
        prv = src
        src = nxt
        if let picture { nxt = picture }
        if prv == nil { prv = src }
        guard let prv, let src, let nxt else { return nil }
        return Compared(prv: prv, src: src, nxt: nxt, field: src.interlaced ? (src.topFieldFirst ? 1 : 0) : 1)
    }

    private struct Compared {
        let prv: BwdifWindow.Picture
        let src: BwdifWindow.Picture
        let nxt: BwdifWindow.Picture
        let field: Int
    }

    /// The statistics a match reads, from the decoded pictures alone: the luma differences, then the field difference
    /// map and its accumulations (compare_fields(c, p)).
    private func statistics(_ c: Compared, work: Scratch, pass: any DeinterlacePass) -> Bool {
        // field ^ order is 0 at the defaults: fxo0m, the matches in their own order. compare_fields(c, p): fbase of c and
        // p is 2 - field; the map comes from the next field's rows when field is 1.
        let fbase = 2 - c.field
        // The diff map's rows the C never writes are read as 0.
        pass.fill(work.map, range: 0..<work.map.length, value: 0)
        guard kernels.encodeLumaDifferences(prv: c.prv.storage.planes, src: c.src.storage.planes, nxt: c.nxt.storage.planes,
                                            layout: layout, width: width, height: height, stats: work.stats, pass: pass)
        else { return false }
        pass.barrier()
        return kernels.encodeFieldMatchCompare(src: c.src.storage.planes, prev: c.src.storage.planes, next: c.prv.storage.planes,
                                               layout: layout, width: width, height: height, fbase1: fbase, fbase2: fbase,
                                               mapFromNextField: c.field == 1, hsub: Int(layout.log2ChromaW), map: work.map,
                                               partials: work.partials, pass: pass)
    }

    /// filter_frame on the statistics: the scene-change test, the comb counts (a scene change's alone), the decision and
    /// the weave, each behind a barrier on the one before.
    private func match(_ c: Compared, work: Scratch, pass: any DeinterlacePass, pushed: inout Pushed) -> Matched? {
        let (prv, src, nxt, field) = (c.prv, c.src, c.nxt, c.field)
        let record = UInt32(nextRecord)
        nextRecord = (nextRecord + 1) % Self.recordRing
        guard let (out, planes) = makePicture(props: src.frame, pts: src.pts) else { return nil }
        pushed.touched.append(out)
        let windowArguments = windowPending(next: out)
        guard kernels.encodeScene(stats: work.stats, cells: cells, state: state, reset: resetPending, emitted: emitted,
                                  scthresh: scthresh, layout: layout, pass: pass) else { return nil }
        pass.barrier()
        guard kernels.encodeCombCounts(prv: prv.storage.planes, src: src.storage.planes, nxt: nxt.storage.planes,
                                       pnParity: 1 - field, layout: layout, width: width, height: height,
                                       cthresh: Self.cthresh, stats: work.stats, pass: pass) else { return nil }
        resetPending = false
        emitted += 1
        pass.barrier()
        guard kernels.encodeMatchDecide(partials: work.partials, partialCount: partialCount, stats: work.stats, cells: cells,
                                        state: state, records: records, record: record, window: windowArguments,
                                        layout: layout, pass: pass) else { return nil }
        pass.barrier()
        guard kernels.encodeWeave(prv: prv.storage.planes, src: src.storage.planes, nxt: nxt.storage.planes,
                                  srcParity: 1 - field, records: records, record: record, layout: layout, width: width,
                                  height: height, into: planes, pass: pass) else { return nil }
        pass.barrier()
        return Matched(picture: out, record: record, field: field)
    }

    // MARK: bwdif's window

    /// The window's decision for the picture before `next`, which the match decision takes on (iv_match_decide), when
    /// the window will have one to make once `next` is in.
    private func windowPending(next: BwdifWindow.Picture) -> (prev: UInt32, cur: UInt32, prevRepeat: Bool, nextRepeat: Bool, start: Bool)? {
        // The window after this push: prev = cur, cur = next (or, at the start, the first picture standing in for
        // itself), next = this one.
        let cur = windowNext ?? nil
        guard let cur else { return nil }
        let prev = windowCur ?? cur
        return (prev.record, cur.record, prev.picture.repeatPict, next.repeatPict, windowStartPending || windowCur == nil)
    }

    /// ff_yadif_filter_frame at frame rate: the window slides and, once full, the current picture goes out, decided on
    /// the GPU (passed through, or its first field deinterlaced). The decision was made with the match (or, at the end,
    /// by iv_window_end); this encodes bwdif after it.
    private func window(_ m: Matched, finishing: Bool, pass: any DeinterlacePass, pushed: inout Pushed) -> (BwdifWindow.Picture, Int64)? {
        windowPrev = windowCur
        windowCur = windowNext
        windowNext = m
        if windowCur == nil {
            windowCur = m
            windowStartPending = true
            return nil
        }
        guard let prev = windowPrev ?? windowCur, let cur = windowCur, let next = windowNext else { return nil }
        if finishing {
            guard kernels.encodeWindowEnd(prev: prev.record, cur: cur.record, prevRepeat: prev.picture.repeatPict,
                                          curRepeat: cur.picture.repeatPict, start: windowStartPending, state: state,
                                          records: records, layout: layout, pass: pass) else { return nil }
            pass.barrier()
        }
        windowStartPending = false
        let pts = cur.picture.pts == Self.noPTS ? Self.noPTS : cur.picture.pts * windowMultiplier
        guard let (out, planes) = makePicture(props: cur.picture.frame, pts: pts) else { return nil }
        pushed.touched += [prev.picture, cur.picture, next.picture, out]
        // A combed weave's field order is its source's; the field is the first (send_frame).
        let tff = UInt32(cur.field == 1 ? 1 : 0)
        let parity = tff ^ 1
        let field = DeinterlaceKernels.BwdifField(prev: prev.picture.storage.planes, cur: cur.picture.storage.planes,
                                                   next: next.picture.storage.planes, parity: parity, fieldParity: parity ^ tff,
                                                   intra: false)
        guard kernels.encodeBwdif(field, layout: layout, width: width, height: height, into: planes,
                                  decided: (records, cur.record), pass: pass) else { return nil }
        pass.barrier()
        guard decimates else {
            pushed.outputs.append(Output(picture: out, fixedPTS: pts, result: nil, kept: nil))
            return nil
        }
        return (out, pts)
    }

    // MARK: decimate

    /// filter_frame with a picture: its differences from the one before into the cycle's slot, and at the end of a cycle
    /// the drop and the timestamps.
    private func decimate(_ item: (BwdifWindow.Picture, Int64), work: Scratch, pass: any DeinterlacePass, pushed: inout Pushed) {
        let (picture, pts) = item
        let slot = queue.count
        let prv = queue.last?.picture ?? last
        queue.append((picture, pts))
        if let prv {
            pushed.touched.append(prv)
            guard kernels.encodeBlockDiffs(prv.storage.planes, picture.storage.planes, layout: layout, width: width,
                                           height: height, blockx: Decimator.blockx, blocky: Decimator.blocky,
                                           blocksX: nxblocks, blocksY: nyblocks, into: work.blocks, pass: pass) else { return }
            pass.barrier()
        }
        if queue.count == Decimator.cycle {
            last = picture
            endCycle(full: true, work: work, slot: slot, diffs: true, none: prv == nil, pass: pass, pushed: &pushed)
        } else {
            _ = kernels.encodeDecimate(blocks: work.blocks, blocksX: nxblocks, blocksY: nyblocks, slot: slot, diffs: true,
                                       none: prv == nil, cycle: nil, dupthresh: dupthresh, scthresh: decimateSCThresh,
                                       state: state, results: results, layout: layout, pass: pass)
            pass.barrier()
        }
    }

    /// The end of a cycle (or of the stream): the GPU decides, and every picture of the cycle goes out with a result slot.
    private func endCycle(full: Bool, work: Scratch, slot: Int, diffs: Bool, none: Bool, pass: any DeinterlacePass,
                          pushed: inout Pushed) {
        let base = nextResult
        nextResult = (nextResult + Decimator.cycle) % Self.resultRing
        let rescaled: [Int64?] = queue.map { $0.pts == Self.noPTS ? nil : av_rescale_q($0.pts, windowTimeBase, outputTimeBase) }
        _ = kernels.encodeDecimate(blocks: work.blocks, blocksX: nxblocks, blocksY: nyblocks, slot: slot, diffs: diffs,
                                   none: none, cycle: (rescaled, full, base), dupthresh: dupthresh, scthresh: decimateSCThresh,
                                   state: state, results: results, layout: layout, pass: pass)
        pass.barrier()
        for (i, item) in queue.enumerated() {
            pushed.outputs.append(Output(picture: item.picture, fixedPTS: nil, result: base + i,
                                         kept: (results, 16 * (base + i) + 8)))
        }
        queue.removeAll()
    }
}
