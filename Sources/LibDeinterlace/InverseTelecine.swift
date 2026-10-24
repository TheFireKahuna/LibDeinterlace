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
/// default. ``FieldMatcher`` and ``Decimator`` transcribe the C's decisions (vf_fieldmatch.c `filter_frame`,
/// `compare_fields`, `checkmm`; vf_decimate.c `filter_frame`, `calc_diffs`), the per-pixel sums run on the GPU
/// (``DeinterlaceKernels``), and a picture the matcher cannot make progressive goes through ``BwdifWindow`` at frame
/// rate. Pictures are held, never copied: a decoded picture by its frame reference, a weave or a deinterlaced picture in
/// a slab of its own.
///
/// Scope is fieldmatch's: 8-bit planar 4:4:4, 4:2:2, 4:2:0, 4:1:1 and 4:1:0 (``takes(format:width:height:)``).
/// Not thread-safe; the owning decoder serialises access with its lock.
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

    let kernels: DeinterlaceKernels
    let makeBuffer: PictureMaker
    let layout: PictureLayout
    let format: AVPixelFormat
    let width: Int
    let height: Int

    private let matcher: FieldMatcher
    private let window: BwdifWindow
    public private(set) var decimator: Decimator?

    /// The time base of every output timestamp: decimate's (the input frame period times 5/4), or bwdif's (half the
    /// input's) without decimation.
    public var outputTimeBase: AVRational { decimator?.outputTimeBase ?? window.outputTimeBase }

    /// `timeBase` is the decoded pictures'; `frameRate` the carriage rate (decimate needs a constant one). Without
    /// `decimates` (2:2, progressive-segmented film) every matched picture goes out.
    public init?(kernels: DeinterlaceKernels, makePicture: @escaping PictureMaker, format: AVPixelFormat, width: Int,
                 height: Int, timeBase: AVRational, frameRate: AVRational, decimates: Bool = true) {
        guard Self.takes(format: format, width: width, height: height), frameRate.num > 0, frameRate.den > 0,
              let layout = PictureLayout(format: format) else { return nil }
        self.kernels = kernels
        self.makeBuffer = makePicture
        self.layout = layout
        self.format = format
        self.width = width
        self.height = height
        window = BwdifWindow(timeBase: timeBase, fieldRate: false)
        matcher = FieldMatcher(kernels: kernels, layout: layout, width: width, height: height)
        decimator = decimates ? Decimator(kernels: kernels, layout: layout, width: width, height: height,
                                          inputTimeBase: window.outputTimeBase, frameRate: frameRate) : nil
    }

    /// Feeds a decoded picture (nil: end of stream) and returns the film pictures it releases, with their timestamps on
    /// ``outputTimeBase``.
    public func push(_ picture: BwdifWindow.Picture?) -> [(picture: BwdifWindow.Picture, pts: Int64)] {
        var matched: [BwdifWindow.Picture] = []
        if let picture {
            if let m = matcher.push(picture, makeWeave: makePicture) { matched.append(m) }
        } else {
            if let m = matcher.push(nil, makeWeave: makePicture) { matched.append(m) }
        }
        var deinterlaced: [(BwdifWindow.Picture, Int64)] = []
        for m in matched { deinterlaced += materialise(window.push(m)) }
        if picture == nil { deinterlaced += materialise(window.finish()) }
        guard let decimator else { return deinterlaced.map { (picture: $0.0, pts: $0.1) } }
        var out: [(picture: BwdifWindow.Picture, pts: Int64)] = []
        for (p, pts) in deinterlaced { out += decimator.push(p, pts: pts) }
        if picture == nil { out += decimator.finish() }
        return out
    }

    /// Drops everything held (a seek).
    public func reset() {
        matcher.reset()
        window.reset()
        decimator?.reset()
    }

    /// An empty picture in a slab with `props`'s properties: the matcher's weaves and bwdif's outputs.
    private func makePicture(props: UnsafePointer<AVFrame>, interlaced: Bool, topFieldFirst: Bool,
                             pts: Int64) -> (BwdifWindow.Picture, PlaneSet)? {
        guard let made = makeBuffer(format, width, height), let frame = av_frame_alloc() else { return nil }
        _ = av_frame_copy_props(frame, props)
        frame.pointee.format = format.rawValue
        frame.pointee.width = Int32(width)
        frame.pointee.height = Int32(height)
        frame.pointee.pts = pts
        Self.setFlags(frame, interlaced: interlaced, topFieldFirst: topFieldFirst)
        let storage = BwdifWindow.Storage(frame: frame, planes: made.planes, release: made.release)
        return (BwdifWindow.Picture(storage: storage, pts: pts), made.planes)
    }

    static func setFlags(_ frame: UnsafeMutablePointer<AVFrame>, interlaced: Bool, topFieldFirst: Bool) {
        frame.pointee.flags &= ~(AV_FRAME_FLAG_INTERLACED | AV_FRAME_FLAG_TOP_FIELD_FIRST)
        if interlaced { frame.pointee.flags |= AV_FRAME_FLAG_INTERLACED }
        if topFieldFirst { frame.pointee.flags |= AV_FRAME_FLAG_TOP_FIELD_FIRST }
    }

    /// bwdif's outputs as held pictures: a passed-through picture as it is, a deinterlaced one written into a slab.
    private func materialise(_ outputs: [BwdifWindow.Output]) -> [(BwdifWindow.Picture, Int64)] {
        outputs.compactMap { output in
            switch output {
            case .picture(let p, let pts):
                return (p, pts)
            case .field(let prev, let cur, let next, let parity, let fieldParity, let intra, let pts):
                guard let (picture, planes) = makePicture(props: cur.frame, interlaced: false, topFieldFirst: false, pts: pts)
                else { return nil }
                let field = DeinterlaceKernels.BwdifField(prev: prev.storage.planes, cur: cur.storage.planes,
                                                           next: next.storage.planes, parity: parity, fieldParity: fieldParity,
                                                           intra: intra)
                guard kernels.deinterlace(field, layout: layout, width: width, height: height, into: planes) else { return nil }
                return (picture, pts)
            }
        }
    }
}

/// vf_fieldmatch.c at its defaults (`order=auto`, `mode=pc_n`, `field=auto`, `mchroma=1`, no exclusion band,
/// `scthresh=12`, `combmatch=sc`, `cthresh=9`, `chroma=0`, `blockx=blocky=16`, `combpel=80`).
public final class FieldMatcher {
    enum Match: Int { case p = 0, c, n, b, u }

    public static let combpel = 80
    public static let cthresh = 9

    let kernels: DeinterlaceKernels
    let layout: PictureLayout
    let width: Int
    let height: Int
    private let scthresh: Int64

    private var prv: BwdifWindow.Picture?
    private var src: BwdifWindow.Picture?
    private var nxt: BwdifWindow.Picture?
    private var lastscdiff: Int64 = 0
    private var lastn: Int64 = 0
    /// The output link's frame_count_in: pictures put out so far.
    private var emitted: Int64 = 0

    init(kernels: DeinterlaceKernels, layout: PictureLayout, width: Int, height: Int) {
        self.kernels = kernels
        self.layout = layout
        self.width = width
        self.height = height
        // (int64_t)((w * h * 255.0 * fm->scthresh_flt) / 100.0)
        scthresh = Int64((Double(width * height) * 255.0 * 12.0) / 100.0)
    }

    func reset() {
        prv = nil
        src = nil
        nxt = nil
        lastscdiff = 0
        lastn = 0
        emitted = 0
    }

    private static func fieldBase(_ match: Match, _ field: Int) -> Int { match.rawValue < 3 ? 2 - field : 1 + field }

    private func select(_ match: Match) -> BwdifWindow.Picture {
        switch match {
        case .p, .b: return prv!
        case .n, .u: return nxt!
        case .c: return src!
        }
    }

    /// The weave `match` makes (create_weave_frame): which picture gives the rows of which parity.
    private func weaveParts(_ match: Match, field: Int) -> (other: BwdifWindow.Picture, srcParity: Int) {
        switch match {
        case .p: return (prv!, 1 - field)
        case .n: return (nxt!, 1 - field)
        case .b: return (prv!, field)
        case .u: return (nxt!, field)
        case .c: return (src!, 0)
        }
    }

    /// filter_frame: `picture` nil flushes the last one.
    func push(_ picture: BwdifWindow.Picture?,
              makeWeave: (UnsafePointer<AVFrame>, Bool, Bool, Int64) -> (BwdifWindow.Picture, PlaneSet)?)
        -> BwdifWindow.Picture? {
        // SLIDING_FRAME_WINDOW
        prv = src
        src = nxt
        if let picture { nxt = picture }
        if prv == nil { prv = src }
        guard prv != nil, let src, nxt != nil else { return nil }

        let order = src.interlaced ? (src.topFieldFirst ? 1 : 0) : 1
        let field = order
        // field ^ order is 0 at the defaults: fxo0m, the matches in their own order.
        var combs = [Int](repeating: -1, count: 5)

        var match = compareFields(.c, .p, field: field)

        var sc = false
        if lastn == emitted - 1 {
            if lastscdiff > scthresh { sc = true }
        } else if let d = kernels.lumaAbsDiff(prv!.storage.planes, src.storage.planes, layout: layout, width: width, height: height),
                  d > scthresh {
            sc = true
        }
        if !sc {
            lastn = emitted
            lastscdiff = kernels.lumaAbsDiff(src.storage.planes, nxt!.storage.planes, layout: layout, width: width, height: height) ?? 0
            sc = lastscdiff > scthresh
        }
        if sc { match = checkmm(&combs, match, .n, field: field) }

        let interlacedFrame = combs[match.rawValue] >= Self.combpel
        emitted += 1

        let out: BwdifWindow.Picture
        if match == .c {
            // av_frame_clone(src), flags set below: the decoded picture itself, with props of its own.
            guard let frame = av_frame_alloc() else { return nil }
            _ = av_frame_copy_props(frame, src.frame)
            frame.pointee.format = src.frame.pointee.format
            frame.pointee.width = src.frame.pointee.width
            frame.pointee.height = src.frame.pointee.height
            let held = src.storage
            InverseTelecine.setFlags(frame, interlaced: interlacedFrame, topFieldFirst: interlacedFrame && field == 1)
            out = BwdifWindow.Picture(storage: .init(frame: frame, planes: held.planes, release: { _ = held }), pts: src.pts)
        } else {
            let parts = weaveParts(match, field: field)
            guard let (picture, planes) = makeWeave(src.frame, interlacedFrame, interlacedFrame && field == 1, src.pts),
                  kernels.weave(src: src.storage.planes, other: parts.other.storage.planes, srcParity: parts.srcParity,
                                 layout: layout, width: width, height: height, into: planes) else { return nil }
            out = picture
        }
        return out
    }

    /// checkmm.
    private func checkmm(_ combs: inout [Int], _ m1: Match, _ m2: Match, field: Int) -> Match {
        for m in [m1, m2] where combs[m.rawValue] < 0 {
            let parts = weaveParts(m, field: field)
            combs[m.rawValue] = kernels.combScore(src: src!.storage.planes, other: parts.other.storage.planes,
                                                   srcParity: parts.srcParity, layout: layout, width: width, height: height,
                                                   cthresh: Self.cthresh) ?? 0
        }
        let c1 = combs[m1.rawValue], c2 = combs[m2.rawValue]
        if (c2 * 3 < c1 || (c2 * 2 < c1 && c1 > Self.combpel)) && abs(c2 - c1) >= 30 && c2 < Self.combpel {
            return m2
        }
        return m1
    }

    /// compare_fields: the GPU's accumulations, then the C's ratios in single precision.
    private func compareFields(_ match1: Match, _ match2: Match, field: Int) -> Match {
        let fromNext = !((match1.rawValue >= 3 && field == 1) || (match1.rawValue < 3 && field != 1))
        guard let a = kernels.fieldMatchCompare(src: src!.storage.planes, prev: select(match1).storage.planes,
                                                 next: select(match2).storage.planes, layout: layout, width: width,
                                                 height: height, fbase1: Self.fieldBase(match1, field),
                                                 fbase2: Self.fieldBase(match2, field), mapFromNextField: fromNext,
                                                 hsub: Int(layout.log2ChromaW)) else { return match1 }
        return Self.decide(a, match1, match2)
    }

    /// compare_fields' decision, from its six accumulations.
    static func decide(_ a: DeinterlaceKernels.FieldAccumulations, _ match1: Match, _ match2: Match) -> Match {
        var accumPm = a.pm, accumNm = a.nm
        if accumPm < 500 && accumNm < 500 && (a.pml >= 500 || a.nml >= 500) && max(a.pml, a.nml) > 3 * min(a.pml, a.nml) {
            accumPm = a.pml
            accumNm = a.nml
        }
        // (int)((accum / 6.0f) + 0.5f)
        func norm(_ v: UInt64) -> Int { Int((Float(v) / Float(6.0)) + Float(0.5)) }
        let norm1 = norm(a.pc), norm2 = norm(a.nc), mtn1 = norm(accumPm), mtn2 = norm(accumNm)
        let c1 = Float(max(norm1, norm2)) / Float(max(min(norm1, norm2), 1))
        let c2 = Float(max(mtn1, mtn2)) / Float(max(min(mtn1, mtn2), 1))
        let mr = Float(max(mtn1, mtn2)) / Float(max(max(norm1, norm2), 1))
        if ((mtn1 >= 500 || mtn2 >= 500) && (mtn1 * 2 < mtn2 * 1 || mtn2 * 2 < mtn1 * 1))
            || ((mtn1 >= 1000 || mtn2 >= 1000) && (mtn1 * 3 < mtn2 * 2 || mtn2 * 3 < mtn1 * 2))
            || ((mtn1 >= 2000 || mtn2 >= 2000) && (mtn1 * 5 < mtn2 * 4 || mtn2 * 5 < mtn1 * 4))
            || ((mtn1 >= 4000 || mtn2 >= 4000) && c2 > c1) {
            return mtn1 > mtn2 ? match2 : match1
        } else if Double(mr) > 0.005 && max(mtn1, mtn2) > 150 && (mtn1 * 2 < mtn2 * 1 || mtn2 * 2 < mtn1 * 1) {
            return mtn1 > mtn2 ? match2 : match1
        }
        return norm1 > norm2 ? match2 : match1
    }
}

/// vf_decimate.c at its defaults (`cycle=5`, `dupthresh=1.1`, `scthresh=15`, `blockx=blocky=32`, `chroma=1`,
/// `mixed=0`), 8-bit.
public final class Decimator {
    static let cycle = 5
    static let blockx = 32
    static let blocky = 32

    let kernels: DeinterlaceKernels
    let layout: PictureLayout
    let width: Int
    let height: Int
    let inputTimeBase: AVRational
    /// dec_tb: the input frame period times cycle / (cycle - 1).
    let outputTimeBase: AVRational

    private struct Item {
        var picture: BwdifWindow.Picture
        var pts: Int64
        var maxbdiff: Int64
        var totdiff: Int64
    }

    /// Full cycles seen, and those with a picture under the duplicate threshold (the load probe's evidence of 3:2).
    public private(set) var cycles = 0
    public private(set) var cyclesWithDuplicate = 0

    private let dupthresh: Int64
    private let scthresh: Int64
    private let nxblocks: Int
    private let nyblocks: Int
    private var queue: [Item?]
    private var fid = 0
    private var last: BwdifWindow.Picture?
    private var startPTS: Int64?
    private var lastPTS: Int64 = 0
    private var lastDuration: Int64 = 0

    init(kernels: DeinterlaceKernels, layout: PictureLayout, width: Int, height: Int,
         inputTimeBase: AVRational, frameRate: AVRational) {
        self.kernels = kernels
        self.layout = layout
        self.width = width
        self.height = height
        self.inputTimeBase = inputTimeBase
        // nondec_tb = 1 / fps; dec_tb = nondec_tb * cycle / (cycle - 1)
        outputTimeBase = av_mul_q(av_inv_q(frameRate), AVRational(num: Int32(Self.cycle), den: Int32(Self.cycle - 1)))
        let maxValue: Int64 = 255
        dupthresh = Int64((Double(maxValue * Int64(Self.blockx) * Int64(Self.blocky)) * 1.1) / 100)
        scthresh = Int64((Double(maxValue * Int64(width) * Int64(height)) * 15.0) / 100)
        nxblocks = (width + Self.blockx / 2 - 1) / (Self.blockx / 2)
        nyblocks = (height + Self.blocky / 2 - 1) / (Self.blocky / 2)
        queue = Array(repeating: nil, count: Self.cycle)
    }

    func reset() {
        queue = Array(repeating: nil, count: Self.cycle)
        fid = 0
        last = nil
        startPTS = nil
        lastPTS = 0
        lastDuration = 0
    }

    /// filter_frame with a picture.
    func push(_ picture: BwdifWindow.Picture, pts: Int64) -> [(picture: BwdifWindow.Picture, pts: Int64)] {
        let prv = fid > 0 ? queue[fid - 1]?.picture : last
        var item = Item(picture: picture, pts: pts, maxbdiff: Int64.max, totdiff: Int64.max)
        if let prv { (item.maxbdiff, item.totdiff) = calcDiffs(prv, picture) }
        queue[fid] = item
        fid += 1
        guard fid == Self.cycle else { return [] }
        last = picture
        fid = 0
        var scpos = -1, duppos = -1, lowest = 0
        for i in 0..<Self.cycle {
            if queue[i]!.totdiff > scthresh { scpos = i }
            if queue[i]!.maxbdiff < queue[lowest]!.maxbdiff { lowest = i }
        }
        if queue[lowest]!.maxbdiff < dupthresh { duppos = lowest }
        cycles += 1
        if duppos >= 0 { cyclesWithDuplicate += 1 }
        let drop = scpos >= 0 && duppos < 0 ? scpos : lowest
        return emit(drop: drop)
    }

    /// filter_frame at end of stream: the partial cycle, nothing dropped.
    func finish() -> [(picture: BwdifWindow.Picture, pts: Int64)] {
        let out = emit(drop: Int.min)
        fid = 0
        return out
    }

    private func emit(drop: Int) -> [(picture: BwdifWindow.Picture, pts: Int64)] {
        var out: [(picture: BwdifWindow.Picture, pts: Int64)] = []
        for i in 0..<Self.cycle {
            guard let item = queue[i] else { break }
            queue[i] = nil
            if i == drop { continue }
            if item.pts != Int64.min, startPTS == nil {
                startPTS = av_rescale_q(item.pts, inputTimeBase, outputTimeBase)
            }
            let pts = lastDuration != 0 ? lastPTS + lastDuration : (startPTS ?? 0)
            lastDuration = 1
            lastPTS = pts
            out.append((item.picture, pts))
        }
        return out
    }

    /// calc_diffs: the largest sum over 2 x 2 neighbouring blocks, and the total.
    private func calcDiffs(_ a: BwdifWindow.Picture, _ b: BwdifWindow.Picture) -> (Int64, Int64) {
        guard let bdiffs = kernels.blockDiffs(a.storage.planes, b.storage.planes, layout: layout, width: width, height: height,
                                               blockx: Self.blockx, blocky: Self.blocky, blocksX: nxblocks, blocksY: nyblocks)
        else { return (Int64.max, Int64.max) }
        var maxdiff: Int64 = -1
        for i in 0..<max(nyblocks - 1, 0) {
            for j in 0..<max(nxblocks - 1, 0) {
                let tmp = bdiffs[i * nxblocks + j] + bdiffs[i * nxblocks + j + 1]
                    + bdiffs[(i + 1) * nxblocks + j] + bdiffs[(i + 1) * nxblocks + j + 1]
                if tmp > maxdiff { maxdiff = tmp }
            }
        }
        return (maxdiff, bdiffs.reduce(0, +))
    }
}
