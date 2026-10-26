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
| `FieldMatcher` | `libavfilter/vf_fieldmatch.c` | the film frame each interlaced frame weaves into (defaults: `mode=pc_n`, `combmatch=sc`) |
| `Decimator` | `libavfilter/vf_decimate.c` | one picture of five dropped, timestamps on the film rate (defaults: `cycle=5`) |
| `InverseTelecine` | FFmpeg's documented recipe `fieldmatch,bwdif=mode=send_frame:deint=interlaced,decimate` | film frames back out of 3:2 or 2:2 carriage |

Pictures cross the interface as `PlaneSet` (a Metal buffer, plane offsets and linesizes) described by a `PictureLayout`
(FFmpeg's component descriptors), so planar, semi-planar and packed layouts all read as their planar equivalent.
Pictures are held, never copied: a decoded picture by its frame reference, a woven or deinterlaced one in a buffer the
caller supplies.

The library owns no command queue. Its GPU work goes through the caller's submission (`DeinterlaceSubmission`, with
`DeinterlaceBatch` and `DeinterlacePass`): `encodeBwdif` encodes into a pass of the caller's batch, so a deinterlaced
field can share one submission with the caller's own passes, and the calls that return results (the field matcher's and
decimator's sums, a woven picture) commit a batch of their own and wait for it. The protocols are bindings by index,
pipelines, small constant arguments, dispatches and barriers, so a caller can implement them over Metal 4 or Metal 3.

Scope is FFmpeg's: bwdif on the formats `vf_bwdif.c` takes (8 to 16 bits, 4:2:0, 4:2:2, 4:4:4, 4:1:1, 4:4:0, 4:1:0,
grey 8 and 16; `BwdifWindow.referenceFormats`), field matching and decimation on 8-bit 4:4:4, 4:2:2, 4:2:0, 4:1:1 and
4:1:0 (`InverseTelecine.takes`).

## Tests

`swift test` on a Mac with a Metal device. The tests submit through a Metal 3 implementation of the protocols
(`Metal3Submission`).

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
