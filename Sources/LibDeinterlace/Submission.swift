// The GPU submission the library encodes into, which the caller provides. LGPL-2.1-or-later; see LICENSE.

import Foundation
import Metal

/// The caller's GPU submission. The library owns no command queue: every dispatch it makes goes into a batch from
/// here, so its work runs on the caller's queue, with the caller's Metal (a Metal 4 argument table or a Metal 3
/// encoder behind the same calls), and a deinterlace pass can share a batch with the caller's own passes
/// (``DeinterlaceKernels/encodeBwdif(_:layout:width:height:into:pass:)``).
public protocol DeinterlaceSubmission: AnyObject, Sendable {
    var device: MTLDevice { get }
    /// A new batch, which starts on the GPU only after the work committed before it from this submission: inverse
    /// telecine's passes carry their running state from one to the next.
    func makeBatch() -> (any DeinterlaceBatch)?
}

/// Work for one commit, encoded by one caller and committed once.
public protocol DeinterlaceBatch: AnyObject {
    /// A compute pass in this batch; the previous one, if still open, is ended first.
    func makePass() -> (any DeinterlacePass)?
    /// Ends the open pass and commits the batch. `completion` runs once, off the caller's thread, with nil when the
    /// batch completed or why it failed.
    func commit(_ completion: @escaping @Sendable (_ failure: String?) -> Void)
}

/// A compute pass. Dispatches in a pass may run together unless a ``barrier()`` separates them; bindings are by index,
/// as the kernels declare them.
public protocol DeinterlacePass: AnyObject {
    func setPipeline(_ pipeline: MTLComputePipelineState)
    func setBuffer(_ buffer: MTLBuffer, offset: Int, index: Int)
    /// `bytes` copied into the batch and bound at `index`, as a small constant argument.
    func setBytes(_ bytes: UnsafeRawBufferPointer, index: Int)
    func dispatchThreads(_ grid: MTLSize, threadsPerThreadgroup: MTLSize)
    func dispatchThreadgroups(_ groups: MTLSize, threadsPerThreadgroup: MTLSize)
    /// Threadgroups counted on the GPU: three 32-bit words at `offset` (a dimension of 0 dispatches nothing).
    func dispatchThreadgroups(indirectBuffer: MTLBuffer, offset: Int, threadsPerThreadgroup: MTLSize)
    /// Every dispatch after this sees every write before it.
    func barrier()
}
