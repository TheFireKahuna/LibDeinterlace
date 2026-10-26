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

    /// Compiled by the device, the archive unread: Metal 3 takes no Metal 4 archive.
    func makeComputePipeline(_ request: DeinterlacePipelineRequest) throws -> MTLComputePipelineState {
        try device.makeComputePipelineState(function: request.library.makeFunction(name: request.function,
                                                                                   constantValues: request.constants))
    }

    /// `encode` in a pass of a batch of its own, committed; waits for it and every batch before it when `wait`.
    @discardableResult
    func run(wait: Bool = true, _ encode: (any DeinterlacePass) -> Bool) -> Bool {
        runBatch(wait: wait) { batch in batch.makePass().map(encode) ?? false }
    }

    /// `encode` into a batch of its own, committed; waits for it and every batch before it when `wait`.
    @discardableResult
    func runBatch(wait: Bool = true, _ encode: (any DeinterlaceBatch) -> Bool) -> Bool {
        guard let batch = makeBatch() else { return false }
        let encoded = encode(batch)
        let done = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var failed = false
        batch.commit { failure in
            failed = failure != nil
            done.signal()
        }
        if wait { done.wait() }
        return encoded && !failed
    }

    private final class Batch: DeinterlaceBatch {
        let commandBuffer: MTLCommandBuffer
        var open: Pass?

        init(_ commandBuffer: MTLCommandBuffer) { self.commandBuffer = commandBuffer }

        func makePass() -> (any DeinterlacePass)? {
            open?.encoder.endEncoding()
            open = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent).map { Pass($0, commandBuffer: commandBuffer) }
            return open
        }

        func commit(_ completion: @escaping @Sendable (String?) -> Void) {
            open?.encoder.endEncoding()
            open = nil
            commandBuffer.addCompletedHandler { completion($0.error.map { "\($0)" }) }
            commandBuffer.commit()
        }
    }

    /// A compute encoder, ended for a blit encoder at each fill and started again after it, its bindings carried over.
    private final class Pass: DeinterlacePass {
        let commandBuffer: MTLCommandBuffer
        var encoder: MTLComputeCommandEncoder
        private var pipeline: MTLComputePipelineState?
        private var buffers: [Int: (MTLBuffer, Int)] = [:]
        private var waits: [MTLFence] = []

        init(_ encoder: MTLComputeCommandEncoder, commandBuffer: MTLCommandBuffer) {
            self.encoder = encoder
            self.commandBuffer = commandBuffer
        }

        func fill(_ buffer: MTLBuffer, range: Range<Int>, value: UInt8) {
            encoder.endEncoding()
            let blit = commandBuffer.makeBlitCommandEncoder()!
            blit.fill(buffer: buffer, range: range, value: value)
            blit.endEncoding()
            encoder = commandBuffer.makeComputeCommandEncoder(dispatchType: .concurrent)!
            for fence in waits { encoder.waitForFence(fence) }
            if let pipeline { encoder.setComputePipelineState(pipeline) }
            for (index, (buffer, offset)) in buffers { encoder.setBuffer(buffer, offset: offset, index: index) }
        }

        func setPipeline(_ pipeline: MTLComputePipelineState) {
            self.pipeline = pipeline
            encoder.setComputePipelineState(pipeline)
        }
        func setBuffer(_ buffer: MTLBuffer, offset: Int, index: Int) {
            buffers[index] = (buffer, offset)
            encoder.setBuffer(buffer, offset: offset, index: index)
        }
        func setBytes(_ bytes: UnsafeRawBufferPointer, index: Int) {
            encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: index)
        }
        func dispatchThreads(_ grid: MTLSize, threadsPerThreadgroup: MTLSize) {
            encoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        func dispatchThreadgroups(_ groups: MTLSize, threadsPerThreadgroup: MTLSize) {
            encoder.dispatchThreadgroups(groups, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        func dispatchThreadgroups(indirectBuffer: MTLBuffer, offset: Int, threadsPerThreadgroup: MTLSize) {
            encoder.dispatchThreadgroups(indirectBuffer: indirectBuffer, indirectBufferOffset: offset,
                                         threadsPerThreadgroup: threadsPerThreadgroup)
        }
        func barrier() { encoder.memoryBarrier(scope: [.buffers, .textures]) }
        func waitForFence(_ fence: MTLFence) {
            waits.append(fence)
            encoder.waitForFence(fence)
        }
        func updateFence(_ fence: MTLFence) { encoder.updateFence(fence) }
    }
}
