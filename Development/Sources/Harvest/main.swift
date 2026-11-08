// harvest LIBRARY.metallib OUT.mtl4-json: every pipeline the library prewarms (DeinterlaceKernels.archiveRequests),
// specialised from LIBRARY (a tvOS build's default.metallib, which Metal on the Mac loads), recorded by a Metal 4
// compiler as a pipeline script, OUT.mtl4-json, and as the same pipelines in a Metal 3 script, OUT.mtlp-json (a
// Metal 4 script translates for no GPU without Metal 4), which metal-tt translates for the TVs' GPUs
// (Scripts/build-xcframework.sh).

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
    let script = try serializer.serializeAsPipelinesScript()
    let url = URL(fileURLWithPath: out)
    try script.write(to: url)
    try metal3Script(script).write(to: url.deletingPathExtension().appendingPathExtension("mtlp-json"))
    print("recorded \(requests.count) pipelines")
}

struct ScriptError: Error, CustomStringConvertible {
    let description: String
}

/// The pipelines of a Metal 4 pipeline script as a Metal 3 one: each library by its path; each specialised function a
/// specialised library of one function (its label, the function it specialises, the same constant values), which a
/// pipeline names as `alias:<label>#<function>`; a plain function `alias:<library>#<function>`. The recorded pipelines
/// set nothing but their function, as the caller makes them; a script with anything else is refused, since the
/// translation would drop it and the archive's entries would not match what is asked for.
func metal3Script(_ metal4: Data) throws -> Data {
    func object(_ value: Any?, _ what: String) throws -> [String: Any] {
        guard let value = value as? [String: Any] else { throw ScriptError(description: "no \(what) in the script") }
        return value
    }
    func objects(_ value: Any?, _ what: String) throws -> [[String: Any]] {
        guard let value = value as? [[String: Any]] else { throw ScriptError(description: "no \(what) in the script") }
        return value
    }
    let root = try object(try JSONSerialization.jsonObject(with: metal4), "root")
    let libraries = try objects(root["libraries"], "libraries")
    let functions = try object(root["function_descriptors"], "function_descriptors")
    var named: [String: (library: String, name: String)] = [:]
    for function in try objects(functions["library_function_descriptors"], "library functions") {
        guard let label = function["label"] as? String, let library = function["library"] as? String,
              let name = function["name"] as? String else { throw ScriptError(description: "a library function without a name") }
        named[label] = (library, name)
    }
    var specialised: [[String: Any]] = []
    var specialisedNames: [String: String] = [:]
    for function in (functions["specialized_function_descriptors"] as? [[String: Any]]) ?? [] {
        guard let label = function["label"] as? String, let base = function["function_descriptor"] as? String,
              base.hasPrefix("fnd:"), let plain = named[String(base.dropFirst(4))],
              let constants = function["constant_values"] else {
            throw ScriptError(description: "a specialised function without its function")
        }
        specialised.append(["label": label, "function": "alias:\(plain.library)#\(plain.name)",
                            "specialized_name": plain.name, "constant_values": constants])
        specialisedNames[label] = plain.name
    }
    var pipelines: [[String: Any]] = []
    let descriptors = try object(root["pipeline_descriptors"], "pipeline_descriptors")
    for pipeline in try objects(descriptors["compute_pipeline_descriptors"], "compute pipelines") {
        guard let reference = pipeline["compute_function_descriptor"] as? String, reference.hasPrefix("fnd:") else {
            throw ScriptError(description: "a compute pipeline without its function")
        }
        let others = pipeline.filter { $0.key != "compute_function_descriptor" }
        guard others.keys.sorted() == ["required_threads_per_threadgroup"],
              (others["required_threads_per_threadgroup"] as? [Int]) == [0, 0, 0] else {
            throw ScriptError(description: "a compute pipeline sets more than its function: \(others)")
        }
        let label = String(reference.dropFirst(4))
        if let name = specialisedNames[label] {
            pipelines.append(["compute_function": "alias:\(label)#\(name)"])
        } else if let plain = named[label] {
            pipelines.append(["compute_function": "alias:\(plain.library)#\(plain.name)"])
        } else {
            throw ScriptError(description: "a compute pipeline names an unknown function \(label)")
        }
    }
    let script: [String: Any] = [
        "libraries": ["paths": libraries, "specialized_functions": specialised],
        "pipelines": ["compute_pipelines": pipelines],
    ]
    return try JSONSerialization.data(withJSONObject: script, options: [.prettyPrinted, .sortedKeys])
}

let arguments = CommandLine.arguments
if #available(macOS 26, *), arguments.count == 3, let device = MTLCreateSystemDefaultDevice() {
    try record(library: arguments[1], to: arguments[2], device: device)
} else {
    FileHandle.standardError.write(Data("usage: harvest LIBRARY.metallib OUT.mtl4-json (macOS 26, a Metal device)\n".utf8))
    exit(2)
}
