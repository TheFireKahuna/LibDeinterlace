import Foundation
import Testing
import AetherLibavutil
@testable import LibDeinterlace

/// Inverse telecine against its reference: the same interlaced frames through FFmpeg's
/// `fieldmatch,bwdif=mode=send_frame:deint=interlaced,decimate` (every option at its default) and through
/// ``InverseTelecine``, compared output by output: how many, their timestamps, every sample of every plane.
///
/// FFmpegBuild does not build fieldmatch or decimate, so the reference is the `ffmpeg` command-line tool on the PATH
/// (Homebrew's); the test is skipped without one. FFmpeg 9.0.2's vf_fieldmatch.c, vf_decimate.c, vf_bwdif.c,
/// bwdifdsp.c and yadif_common.c are byte for byte n8.1.3's, the sources the library transcribes.
///
/// The film is progressive pictures with gradients, texture, motion and a little noise; the test applies 3:2 pulldown
/// itself (fields A A B B B C C D D D: frames AA BB BC CD DD, in either field order). Cases cover a plain run, bottom
/// field first, 4:2:2, a scene cut, an edit that breaks the cycle, a stream that ends mid-cycle, and 2:2 (each film
/// picture's own two fields, matched without decimation).
struct InverseTelecineReferenceTests {

    struct Case: CustomStringConvertible, Sendable {
        var name: String
        var width = 96
        var height = 64
        var format = AV_PIX_FMT_YUV420P
        var topFieldFirst = true
        var films = 24
        /// Film pictures from here on show other content (a scene cut).
        var cutAt: Int? = nil
        /// A film picture left out before the pulldown (an edit that breaks the cycle).
        var editAt: Int? = nil
        /// 2:2: each film picture woven from its own two fields; decimation off.
        var psf = false
        var description: String { name }
    }

    static let cases: [Case] = [
        Case(name: "3:2 tff"),
        Case(name: "3:2 bff", topFieldFirst: false),
        Case(name: "3:2 4:2:2", format: AV_PIX_FMT_YUV422P),
        Case(name: "3:2 with a scene cut", films: 32, cutAt: 13),
        Case(name: "3:2 with an edit", films: 32, editAt: 11),
        Case(name: "3:2 ending mid-cycle", films: 9),
        Case(name: "2:2", films: 20, psf: true),
    ]

    static let ffmpeg: String? = ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"].first {
        FileManager.default.isExecutableFile(atPath: $0)
    }

    struct Output: Equatable {
        /// The timestamp in seconds, as an exact fraction.
        var time: (num: Int64, den: Int64)
        var samples: [[Int]]
        static func == (a: Output, b: Output) -> Bool {
            a.time.num * b.time.den == b.time.num * a.time.den && a.samples == b.samples
        }
    }

    static let carriage = AVRational(num: 30000, den: 1001)
    static let timeBase = AVRational(num: 1001, den: 30000)

    @Test("fieldmatch, bwdif and decimate on the GPU equal FFmpeg's", .enabled(if: InverseTelecineReferenceTests.ffmpeg != nil,
          "needs the ffmpeg command-line tool"), arguments: InverseTelecineReferenceTests.cases)
    func matchesFFmpeg(_ c: Case) throws {
        let layout = try #require(PictureLayout(format: c.format))
        let fields = Self.pulldown(c)
        let expected = try Self.ffmpegChain(fields, c, layout)
        let got = try Self.gpu(fields, c, layout)
        #expect(!expected.isEmpty, "FFmpeg put out nothing")
        #expect(got.count == expected.count, "outputs: \(got.count) vs FFmpeg's \(expected.count)")
        for (i, (g, e)) in zip(got, expected).enumerated() {
            #expect(g.time.num * e.time.den == e.time.num * g.time.den,
                    "output \(i): time \(g.time.num)/\(g.time.den) vs \(e.time.num)/\(e.time.den)")
            if let mismatch = Self.firstMismatch(g.samples, e.samples, c, layout) {
                Issue.record("output \(i): \(mismatch)")
                return
            }
        }
    }

    // MARK: - Input

    private static func hash(_ a: Int, _ b: Int, _ t: Int, _ comp: Int) -> Int {
        var h = UInt32(truncatingIfNeeded: a &* 73856093 ^ b &* 19349663 ^ t &* 83492791 ^ comp &* 2654435761)
        h ^= h >> 13
        h = h &* 0x5bd1e995
        h ^= h >> 15
        return Int(h)
    }

    /// Film picture `t`'s sample: gradients moving at different speeds, a texture, a little noise.
    private static func filmValue(_ comp: Int, _ x: Int, _ y: Int, _ t: Int, scene: Int) -> UInt8 {
        let base = scene == 0 ? 0 : 97
        let v = (x * 3 + y * 2 + t * 5 * (comp + 1) + base) % 200 + 28
        let texture = ((x / 6 + y / 4 + t + scene) % 2) * 24
        return UInt8(min(max(v + texture + hash(x, y, t, comp) % 5 - 12, 16), 235))
    }

    /// The interlaced frames' samples, component by component, row by row: rows of each parity from the film
    /// picture its field shows.
    private static func pulldown(_ c: Case) -> [[[UInt8]]] {
        let films = (0..<c.films).filter { $0 != c.editAt }
        // Field n shows film picture fieldFilm[n]; fields alternate parity, the first is `topFieldFirst`'s.
        var fieldFilm: [Int] = []
        for (i, t) in films.enumerated() { fieldFilm += Array(repeating: t, count: c.psf ? 2 : (i % 2 == 0 ? 2 : 3)) }
        let frames = fieldFilm.count / 2
        let layout = PictureLayout(format: c.format)!
        return (0..<frames).map { n in
            // This frame's first field is field 2n, its second 2n + 1; the first is top when top field first.
            let topFilm = c.topFieldFirst ? fieldFilm[2 * n] : fieldFilm[2 * n + 1]
            let bottomFilm = c.topFieldFirst ? fieldFilm[2 * n + 1] : fieldFilm[2 * n]
            return (0..<layout.components.count).map { comp in
                let (w, h) = layout.size(component: comp, width: c.width, height: c.height)
                var plane = [UInt8](repeating: 0, count: w * h)
                for y in 0..<h {
                    let t = y % 2 == 0 ? topFilm : bottomFilm
                    let scene = c.cutAt.map { t >= $0 ? 1 : 0 } ?? 0
                    for x in 0..<w { plane[y * w + x] = filmValue(comp, x, y, t, scene: scene) }
                }
                return plane
            }
        }
    }

    private static func firstMismatch(_ got: [[Int]], _ expected: [[Int]], _ c: Case, _ layout: PictureLayout) -> String? {
        guard got.count == expected.count else { return "components \(got.count) vs \(expected.count)" }
        for comp in got.indices {
            let (w, _) = layout.size(component: comp, width: c.width, height: c.height)
            guard got[comp].count == expected[comp].count else { return "component \(comp) size" }
            if let i = got[comp].indices.first(where: { got[comp][$0] != expected[comp][$0] }) {
                let differing = zip(got[comp], expected[comp]).filter { $0 != $1 }.count
                return "component \(comp) (\(i % w), \(i / w)): \(got[comp][i]) vs \(expected[comp][i]), \(differing) samples differ"
            }
        }
        return nil
    }

    // MARK: - FFmpeg

    private static func run(_ arguments: [String]) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: ffmpeg!)
        process.arguments = ["-hide_banner", "-loglevel", "error", "-y"] + arguments
        let out = Pipe(), err = Pipe()
        process.standardOutput = out
        process.standardError = err
        try process.run()
        let data = out.fileHandleForReading.readDataToEndOfFile()
        let errors = err.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        try #require(process.terminationStatus == 0, "ffmpeg: \(String(decoding: errors, as: UTF8.self))")
        return String(decoding: data, as: UTF8.self)
    }

    private static func ffmpegChain(_ frames: [[[UInt8]]], _ c: Case, _ layout: PictureLayout) throws -> [Output] {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("libdeinterlace-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let input = dir.appendingPathComponent("in.yuv"), output = dir.appendingPathComponent("out.yuv")
        try Data(frames.flatMap { $0.flatMap { $0 } }).write(to: input)
        let name = String(cString: av_get_pix_fmt_name(c.format))
        let chain = "setfield=\(c.topFieldFirst ? "tff" : "bff"),fieldmatch,bwdif=mode=send_frame:deint=interlaced"
            + (c.psf ? "" : ",decimate")
        let common = ["-f", "rawvideo", "-pix_fmt", name, "-s", "\(c.width)x\(c.height)", "-framerate", "30000/1001",
                      "-i", input.path, "-vf", chain, "-fps_mode", "passthrough", "-enc_time_base", "filter", "-c:v", "rawvideo"]
        _ = try run(common + ["-f", "rawvideo", output.path])
        let crc = try run(common + ["-f", "framecrc", "-"])
        // "#tb 0: num/den", then "stream, dts, pts, duration, size, crc" per frame.
        var tb = (num: Int64(1), den: Int64(1))
        var times: [Int64] = []
        for line in crc.split(separator: "\n") {
            if line.hasPrefix("#tb 0:") {
                let parts = line.dropFirst(6).trimmingCharacters(in: .whitespaces).split(separator: "/")
                tb = (Int64(parts[0])!, Int64(parts[1])!)
            } else if !line.hasPrefix("#") {
                let fields = line.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) }
                times.append(Int64(fields[2])!)
            }
        }
        let bytes = [UInt8](try Data(contentsOf: output))
        let sizes = (0..<layout.components.count).map { layout.size(component: $0, width: c.width, height: c.height) }
        let frameSize = sizes.reduce(0) { $0 + $1.width * $1.height }
        try #require(bytes.count == frameSize * times.count, "\(bytes.count) bytes for \(times.count) frames")
        return times.enumerated().map { n, pts in
            var at = n * frameSize
            let samples = sizes.map { s -> [Int] in
                defer { at += s.width * s.height }
                return bytes[at..<(at + s.width * s.height)].map(Int.init)
            }
            return Output(time: (pts * tb.num, tb.den), samples: samples)
        }
    }

    // MARK: - The GPU

    private static func gpu(_ frames: [[[UInt8]]], _ c: Case, _ layout: PictureLayout) throws -> [Output] {
        let device = try #require(Pictures.device, "needs a Metal device")
        let gpu = try #require(Metal3Submission(device: device))
        let kernels = DeinterlaceKernels(submission: gpu)
        let chain = try #require(InverseTelecine(
            kernels: kernels,
            makePicture: { _, w, h in Pictures.planar(layout, width: w, height: h).map { ($0, {}) } },
            format: c.format, width: c.width, height: c.height, timeBase: timeBase, frameRate: carriage,
            decimates: !c.psf))
        // A batch a picture, none waited for until the end, as a player submits them; the pictures each pass touched held
        // until then.
        var outputs: [InverseTelecine.Output] = []
        var held: [BwdifWindow.Picture] = []
        func push(_ picture: BwdifWindow.Picture?) {
            gpu.runBatch(wait: false) { batch in
                let pushed = chain.push(picture, into: batch)
                outputs += pushed.outputs
                held += pushed.touched
                return true
            }
        }
        for (n, planes) in frames.enumerated() {
            let set = try #require(Pictures.planar(layout, width: c.width, height: c.height))
            let base = set.buffer.contents().assumingMemoryBound(to: UInt8.self)
            for (comp, plane) in planes.enumerated() {
                let (w, h) = layout.size(component: comp, width: c.width, height: c.height)
                for y in 0..<h {
                    for x in 0..<w { base[Int(set.offsets[comp]) + y * Int(set.linesizes[comp]) + x] = plane[y * w + x] }
                }
            }
            let frame = try #require(av_frame_alloc())
            frame.pointee.format = c.format.rawValue
            frame.pointee.width = Int32(c.width)
            frame.pointee.height = Int32(c.height)
            frame.pointee.pts = Int64(n)
            frame.pointee.flags |= AV_FRAME_FLAG_INTERLACED
            if c.topFieldFirst { frame.pointee.flags |= AV_FRAME_FLAG_TOP_FIELD_FIRST }
            push(BwdifWindow.Picture(storage: .init(frame: frame, planes: set, release: nil), pts: Int64(n)))
        }
        push(nil)
        gpu.run { _ in true }
        withExtendedLifetime(held) {}
        let tb = chain.outputTimeBase
        return outputs.compactMap { output in
            chain.resolve(output).map { pts in
                Output(time: (pts * Int64(tb.num), Int64(tb.den)),
                       samples: Pictures.samples(chain.shown(output).storage.planes, layout, width: c.width, height: c.height))
            }
        }
    }
}
