# LibDeinterlace

FFmpeg's deinterlacer and inverse telecine on the GPU, for Apple platforms. The video filters of FFmpeg's
libavfilter (`bwdif`, `fieldmatch`, `decimate`) transcribed to Metal compute kernels, with their scheduling and
decisions in Swift, so a player that keeps decoded pictures in Metal buffers can deinterlace them, or recover the film
frames of telecined video, without a CPU pass over the pixels.

The C is the reference. None of these filters has a specification beyond FFmpeg's source, so the arithmetic is the same
integer arithmetic, sample for sample, and the tests require every output (how many, their timestamps, every sample of
every plane) to equal FFmpeg's on the same frames.

## What it does

| Part | Transcribes | What it gives |
|---|---|---|
| `DeinterlaceKernels.encodeBwdif` | `libavfilter/bwdifdsp.c`, `vf_bwdif.c` | one deinterlaced field, encoded into the caller's compute pass |
| `BwdifWindow` | `libavfilter/yadif_common.c` | which pictures, parities and timestamps each field takes (`deint=interlaced`, `parity=auto`, field or frame rate) |
| `InverseTelecine` | `libavfilter/vf_fieldmatch.c` (`mode=pc_n`, `combmatch=sc`), `vf_decimate.c` (`cycle=5`), in FFmpeg's documented recipe `fieldmatch,bwdif=mode=send_frame:deint=interlaced,decimate` | film frames back out of 3:2 or 2:2 carriage, every decision taken on the GPU in the caller's batch |

Pictures cross the interface as `PlaneSet` (a Metal buffer, plane offsets and linesizes) described by a `PictureLayout`
(FFmpeg's component descriptors), so planar, semi-planar and packed layouts all read as their planar equivalent.
Pictures are held, never copied: a decoded picture by its frame reference, a woven or deinterlaced one in a buffer the
caller supplies.

The library owns no command queue and makes no pipeline itself. Its GPU work goes through the caller's submission
(`DeinterlaceSubmission`, with `DeinterlaceBatch` and `DeinterlacePass`): `encodeBwdif` encodes into a pass of the
caller's batch, so a deinterlaced field can share one submission with the caller's own passes, and every pipeline is
the caller's to make from a `DeinterlacePipelineRequest` (the precompiled library, a function, its constants, the
binary archive). The protocols are bindings by index, pipelines, small constant arguments, fills, dispatches (direct
and indirect), barriers and fences, so a caller can implement them over Metal 4 or Metal 3.

Inverse telecine never waits for the GPU. fieldmatch's decisions (compare_fields, the scene-change test, checkmm),
bwdif's window at frame rate and decimate's choice of the picture to drop run as kernels in the same pass as the
per-pixel work they decide on, each picture's verdict in a record and the chain's running state (fieldmatch's
scene-change memory, bwdif's field-end, decimate's timestamps and cycle) in a buffer of its own. `push(_:into:)`
encodes one picture's work into two passes of the caller's batch and returns the film pictures it releases, and every
picture the work reads, which the caller holds until the work has completed; `resolve(_:)` then says whether each goes
out, and at what timestamp, and `shown(_:)` where its pixels are. The first pass reads only decoded pictures, so it
runs beside earlier batches' work; fences order the rest, and a caller's pass reading the outputs waits on
`InverseTelecine.fence`. Comb scores are counted only on a scene change, a picture bwdif's window passes through is not
copied (readers choose it by its record), and decimate's dropped picture is marked for the caller's work to skip
(`Output.kept`), all decided on the GPU.

Scope is FFmpeg's: bwdif on the formats `vf_bwdif.c` takes (8 to 16 bits, 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:0, 4:1:0,
grey 8 and 16; `BwdifWindow.referenceFormats`), field matching and decimation on 8-bit 4:4:4, 4:2:2, 4:2:0, 4:1:1 and
4:1:0 (`InverseTelecine.takes`).

## Using it

Depend on this repository: its root package ships `Frameworks/LibDeinterlace.xcframework` (tvOS, iOS, macOS and
visionOS, devices and simulators) and brings FFmpegBuild's `AetherLibavutil`, which the library's interface uses, with
it. The framework is built with library evolution: its `.swiftinterface` is a stable ABI, so an app can have a rebuilt
LibDeinterlace dropped in, which LGPL-2.1 asks for. Public enums are `@frozen` only where no further case can exist.

Shaders are precompiled: each slice's bundle carries `default.metallib`, and the tvOS device slice also carries a Metal 4
binary archive of every pipeline the library prewarms, translated ahead of time for the A15 (Apple TV 4K, 3rd
generation), so a Metal 4 compiler given the request's archive takes those pipelines from it instead of compiling them.
Metal takes an archive only on the OS release it was built for (tvOS 27); elsewhere, and on other GPUs, pipelines
compile. Call `DeinterlaceKernels.prewarm(_:)` once, early, off the main thread, for the formats a stream will bring,
so no picture waits for a pipeline.

## Developing and building

The source package is `Development/`: `cd Development && swift test`, and `Development/Scripts/build-xcframework.sh`
(Xcode 27.2, macOS 26 or later) rebuilds `Frameworks/LibDeinterlace.xcframework`: every slice with library evolution,
the shaders compiled with the IEEE math the C uses (`#pragma METAL fp math_mode(safe)`), and the archive recorded by a
Metal 4 compiler from the tvOS slice's own `default.metallib` (`Sources/Harvest`) and translated by `metal-tt`.

## Tests

`cd Development && swift test` on a Mac with a Metal device. The tests submit through a Metal 3 implementation of the
protocols (`Metal3Submission`), from the source package, with the shaders precompiled as they ship.

- `PipelineTests`: every pipeline of the prewarmed formats is made from the precompiled library, and the archive's
  requests are prewarm's.

- `BwdifReferenceTests`: libavfilter's bwdif (from FFmpegBuild, in process) against the GPU on 30 cases: 8 to 16 bits,
  every subsampling, grey, NV12, P010, UYVY and big-endian sources, both field orders, field and frame rate, odd and
  minimal sizes, progressive and repeat-field pictures, a missing timestamp, one- and two-picture streams.
- `InverseTelecineReferenceTests`: the `ffmpeg` command-line tool on the PATH (FFmpegBuild does not build fieldmatch or
  decimate) against the GPU: 3:2 in both field orders, 4:2:2, a scene cut, an edit that breaks the cycle, a stream that
  ends mid-cycle, and 2:2. Skipped without `ffmpeg`. FFmpeg 9.0.2's sources of all five filters are byte for byte
  n8.1.3's.

## License

LGPL-2.1-or-later, the license of the FFmpeg sources it transcribes (see `LICENSE`). The product is a dynamic library so
an application that links it can have it replaced, as with FFmpegBuild's frameworks.

The transcribed sources and their authors:

- `libavfilter/bwdifdsp.c`, `vf_bwdif.c`: Copyright (C) 2016 Thomas Mundt; Copyright (C) 2006-2011 Michael Niedermayer;
  Copyright (C) 2012 British Broadcasting Corporation
- `libavfilter/yadif_common.c`: Copyright (C) 2006-2011 Michael Niedermayer; 2010 James Darnley
- `libavfilter/vf_fieldmatch.c`, `vf_decimate.c`: Copyright (c) 2012 Fredrik Mellbin; Copyright (c) 2013 Clément Bœsch
