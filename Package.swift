// swift-tools-version: 6.0

import PackageDescription

// LibDeinterlace as players take it: the XCFramework Development/Scripts/build-xcframework.sh builds from the source
// package in Development/ (where the tests run). Library evolution gives it a stable ABI, so an app can have a rebuilt
// LibDeinterlace dropped in, as LGPL-2.1 asks of the FFmpeg-derived code it carries; it links dynamically, as
// FFmpegBuild's frameworks do.
let package = Package(
    name: "LibDeinterlace",
    platforms: [
        .iOS(.v16),
        .tvOS(.v16),
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "LibDeinterlace", targets: ["LibDeinterlace", "LibDeinterlaceLinkage"]),
    ],
    dependencies: [
        .package(url: "https://github.com/TheFireKahuna/FFmpegBuild", "3.10.1-fire.1"..<"3.11.0"),
    ],
    targets: [
        .binaryTarget(name: "LibDeinterlace", path: "Frameworks/LibDeinterlace.xcframework"),
        // The framework's interface imports AetherLibavutil and its binary links it: this brings FFmpegBuild's into
        // the app with it.
        .target(
            name: "LibDeinterlaceLinkage",
            dependencies: [.product(name: "AetherLibavutil", package: "FFmpegBuild")],
            path: "Linkage"
        ),
    ]
)
