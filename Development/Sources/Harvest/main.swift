// harvest LIBRARY.metallib OUT.mtl4-json: every pipeline the library prewarms (DeinterlaceKernels.archiveRequests),
// specialised from LIBRARY (a tvOS build's default.metallib, which Metal on the Mac loads), recorded by a Metal 4
// compiler as a pipeline script that metal-tt translates for the TV's GPU (Scripts/build-xcframework.sh).

import Foundation
import Metal
@_spi(Archive) import LibDeinterlace

@available(macOS 26, *)
func record(library path: String, to out: String, device: MTLDevice) throws {
    let library = try device.makeLibrary(URL: URL(fileURLWithPath: path))
    let serializerDescriptor = MTL4PipelineDataSetSerializerDescriptor()
    serializerDescriptor.configuration = .captureDescriptors
    let serializer = device.makePipelineDataSetSerializer(descriptor: serializerDescriptor)
    let compilerDescriptor = MTL4CompilerDescriptor()
    compilerDescriptor.pipelineDataSetSerializer = serializer
    let compiler = try device.makeCompiler(descriptor: compilerDescriptor)
    let requests = DeinterlaceKernels.archiveRequests(library: library)
    for request in requests {
        let function = MTL4LibraryFunctionDescriptor()
        function.name = request.function
        function.library = request.library
        let specialised = MTL4SpecializedFunctionDescriptor()
        specialised.functionDescriptor = function
        specialised.constantValues = request.constants
        let pipeline = MTL4ComputePipelineDescriptor()
        pipeline.computeFunctionDescriptor = specialised
        _ = try compiler.makeComputePipelineState(descriptor: pipeline)
    }
    try serializer.serializeAsPipelinesScript().write(to: URL(fileURLWithPath: out))
    print("recorded \(requests.count) pipelines")
}

let arguments = CommandLine.arguments
if #available(macOS 26, *), arguments.count == 3, let device = MTLCreateSystemDefaultDevice() {
    try record(library: arguments[1], to: arguments[2], device: device)
} else {
    FileHandle.standardError.write(Data("usage: harvest LIBRARY.metallib OUT.mtl4-json (macOS 26, a Metal device)\n".utf8))
    exit(2)
}
