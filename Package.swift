// swift-tools-version: 6.0

import PackageDescription

// A dynamic library on purpose: it carries code derived from FFmpeg under LGPL-2.1-or-later, and an app that links it
// dynamically can have it replaced, as with FFmpegBuild's frameworks.
let package = Package(
    name: "LibDeinterlace",
    platforms: [
        .iOS(.v16),
        .tvOS(.v16),
        .macOS(.v14),
        .visionOS(.v1),
    ],
    products: [
        .library(name: "LibDeinterlace", type: .dynamic, targets: ["LibDeinterlace"]),
    ],
    dependencies: [
        .package(url: "https://github.com/TheFireKahuna/FFmpegBuild", .upToNextMinor(from: "3.10.1")),
    ],
    targets: [
        .target(
            name: "LibDeinterlace",
            dependencies: [
                .product(name: "AetherLibavutil", package: "FFmpegBuild"),
            ],
            path: "Sources/LibDeinterlace"
        ),
        .testTarget(
            name: "LibDeinterlaceTests",
            dependencies: [
                "LibDeinterlace",
                // The umbrella product: libavfilter, the reference, needs the rest of FFmpeg at run time.
                .product(name: "AetherFFmpegBuild", package: "FFmpegBuild"),
            ],
            path: "Tests/LibDeinterlaceTests"
        ),
    ]
)
