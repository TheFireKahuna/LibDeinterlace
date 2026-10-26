import Foundation
import Metal
import Testing
import AetherLibavutil
@_spi(Archive) @testable import LibDeinterlace

struct PipelineTests {
    /// Every pipeline prewarm makes and the archive carries, made from the bundle's precompiled library: a kernel name
    /// the library has not got, or a layout its kernels cannot be specialised for, fails here rather than as a pipeline
    /// a player never gets.
    @Test("every pipeline for the common formats is made from the precompiled library", arguments: DeinterlaceKernels.commonFormats)
    func pipelinesMade(_ format: AVPixelFormat) throws {
        let device = try #require(Pictures.device)
        let gpu = try #require(Metal3Submission(device: device))
        let kernels = DeinterlaceKernels(submission: gpu)
        let layout = try #require(PictureLayout(format: format))
        for name in DeinterlaceKernels.kernels(for: format) {
            #expect(throws: Never.self, "\(name)") { _ = try kernels.pipeline(for: layout, function: name) }
        }
    }

    @Test("the archive's requests are prewarm's pipelines")
    func archiveRequests() throws {
        let device = try #require(Pictures.device)
        let library = try DeinterlaceKernels.library(for: device)
        let requests = DeinterlaceKernels.archiveRequests(library: library)
        let expected = DeinterlaceKernels.commonFormats.reduce(0) { $0 + DeinterlaceKernels.kernels(for: $1).count }
        #expect(requests.count == expected)
        #expect(Set(requests.map(\.function)) == Set(DeinterlaceKernels.kernels(for: AV_PIX_FMT_YUV420P)))
    }
}
