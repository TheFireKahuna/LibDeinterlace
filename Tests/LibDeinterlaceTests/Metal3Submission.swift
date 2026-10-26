import Foundation
import Metal
import LibDeinterlace

/// The tests' submission: a Metal 3 queue, a command buffer a batch and a compute encoder a pass. A player brings its
/// own (its Metal 4 or Metal 3 queue); this is the plainest one that runs the kernels.
final class Metal3Submission: DeinterlaceSubmission, @unchecked Sendable {
    let device: MTLDevice
    private let queue: MTLCommandQueue

    init?(device: MTLDevice) {
        guard let queue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.queue = queue
    }

    func makeBatch() -> (any DeinterlaceBatch)? {
        queue.makeCommandBuffer().map(Batch.init)
    }

    private final class Batch: DeinterlaceBatch {
        let commandBuffer: MTLCommandBuffer
        var open: Pass?

        init(_ commandBuffer: MTLCommandBuffer) { self.commandBuffer = commandBuffer }

        func makePass() -> (any DeinterlacePass)? {
            open?.encoder.endEncoding()
            open = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent).map(Pass.init)
            return open
        }

        func commit(_ completion: @escaping @Sendable (String?) -> Void) {
            open?.encoder.endEncoding()
            open = nil
            commandBuffer.addCompletedHandler { completion($0.error.map { "\($0)" }) }
            commandBuffer.commit()
        }
    }

    private final class Pass: DeinterlacePass {
        let encoder: MTLComputeCommandEncoder

        init(_ encoder: MTLComputeCommandEncoder) { self.encoder = encoder }

        func setPipeline(_ pipeline: MTLComputePipelineState) { encoder.setComputePipelineState(pipeline) }
        func setBuffer(_ buffer: MTLBuffer, offset: Int, index: Int) { encoder.setBuffer(buffer, offset: offset, index: index) }
        func setBytes(_ bytes: UnsafeRawBufferPointer, index: Int) {
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: index)
        }
        func dispatchThreads(_ grid: MTLSize, threadsPerThreadgroup: MTLSize) {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        func barrier() { encoder.memoryBarrier(scope: [.buffers, .textures]) }
    }
}
