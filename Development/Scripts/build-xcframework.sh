#!/bin/zsh
# build-xcframework.sh: Frameworks/LibDeinterlace.xcframework at the repository's root, from this package.
#
# Every platform's framework, built with library evolution (a stable ABI and a .swiftinterface, so a rebuilt library
# drops into an app as LGPL-2.1 asks), carries its precompiled shaders (the module's bundle, default.metallib). The tvOS
# device slice also carries a binary archive of every pipeline the library prewarms, translated ahead of time for the
# A15 (Apple TV 4K, 3rd generation, Metal 4) and the A12 (2nd generation, Metal 3) and lzfse-compressed (LibDeinterlace-pipelines.metallib.lzfse): its compiler takes those pipelines from the archive instead of compiling
# them. Metal takes an archive only on the OS release it was built for (TVOS_ARCHIVE_OS); on any other release, and on
# GPUs it was not built for, the pipelines are compiled when the player prewarms them.
#
# Needs Xcode 27.2 (metal-tt) on macOS 26 or later.
set -euo pipefail

HERE=${0:A:h}
DEV=${HERE:h}
ROOT=${DEV:h}
NAME=LibDeinterlace
DD=${DD:-$DEV/.build/xcframework}
TVOS_ARCHIVE_OS=${TVOS_ARCHIVE_OS:-27.0}
METAL4_GPU=applegpu_g14p
METAL3_GPU=applegpu_g11p

# destination : products directory suffix
SLICES=(
  "tvOS:-appletvos"
  "tvOS Simulator:-appletvsimulator"
  "iOS:-iphoneos"
  "iOS Simulator:-iphonesimulator"
  "macOS:"
  "visionOS:-xros"
  "visionOS Simulator:-xrsimulator"
)

cd $DEV
mkdir -p $DD/logs
for slice in $SLICES; do
  destination=${slice%%:*}
  log=$DD/logs/${destination// /-}.log
  print "== $destination"
  # The visionOS simulator is arm64 alone, as FFmpegBuild's slice is.
  archs=()
  [[ $destination == "visionOS Simulator" ]] && archs=(ARCHS=arm64)
  if ! xcodebuild build -scheme $NAME -destination "generic/platform=$destination" -configuration Release \
      -derivedDataPath $DD BUILD_LIBRARY_FOR_DISTRIBUTION=YES SKIP_INSTALL=NO $archs > $log 2>&1; then
    grep -E "error:" $log | sort -u | head -20
    print "failed: $log"
    exit 1
  fi
done

# Each slice's framework, with its module (interfaces) and resource bundle inside, where Bundle.module finds it.
ASSEMBLED=$DD/assembled
mkdir -p $ASSEMBLED
arguments=()
for slice in $SLICES; do
  suffix=${slice#*:}
  products=$DD/Build/Products/Release$suffix
  slice_dir=$ASSEMBLED/release$suffix
  if [[ -e $slice_dir ]]; then rm -rf "${slice_dir:?}"; fi
  mkdir -p $slice_dir
  cp -R $products/PackageFrameworks/$NAME.framework $slice_dir/
  framework=$slice_dir/$NAME.framework
  if [[ -d $framework/Versions ]]; then
    contents=$framework/Versions/A
    mkdir -p $contents/Modules $contents/Resources
    cp -R $products/$NAME.swiftmodule $contents/Modules/
    cp -R $products/${NAME}_$NAME.bundle $contents/Resources/
    ln -sfh Versions/Current/Modules $framework/Modules
  else
    mkdir -p $framework/Modules
    cp -R $products/$NAME.swiftmodule $framework/Modules/
    cp -R $products/${NAME}_$NAME.bundle $framework/
  fi
  arguments+=(-framework $framework -debug-symbols $products/$NAME.framework.dSYM)
done

# The tvOS device slice's archive: the prewarm's pipelines, specialised from the very default.metallib the slice ships
# (an archive entry names its library's functions), recorded by a Metal 4 compiler on this Mac
# and written again as a Metal 3 script (a Metal 4 script translates for no GPU without Metal 4), each translated for its
# GPU, the two joined into one archive from which Metal takes the slice of the GPU it runs on, through either API.
print "== archive (tvOS $TVOS_ARCHIVE_OS: $METAL4_GPU from the Metal 4 script, $METAL3_GPU from the Metal 3 one)"
swift build -c release --product Harvest -q
bundle=$ASSEMBLED/release-appletvos/$NAME.framework/${NAME}_$NAME.bundle
script=$DD/pipelines.mtl4-json
$DEV/.build/release/Harvest $bundle/default.metallib $script
# Shipped lzfse-compressed: a raw archive is a Mach-O GPU binary, which App Store signing cannot sign.
raw=$DD/$NAME-pipelines.metallib
xcrun metal-tt -arch $METAL4_GPU -tvos-version-min $TVOS_ARCHIVE_OS $bundle/default.metallib $script -o $raw.metal4
xcrun metal-tt -arch $METAL3_GPU -tvos-version-min $TVOS_ARCHIVE_OS ${script%.mtl4-json}.mtlp-json -o $raw.metal3
xcrun metal-lipo -create $raw.metal4 $raw.metal3 -output $raw
compression_tool -encode -a lzfse -i $raw -o $bundle/$NAME-pipelines.metallib.lzfse
print "archive: $(stat -f %z $raw) bytes unpacked ($(xcrun metal-lipo -info $raw))"

mkdir -p $ROOT/Frameworks
output=$ROOT/Frameworks/$NAME.xcframework
if [[ -e $output ]]; then rm -rf "${output:?}"; fi
xcodebuild -create-xcframework $arguments -output $output
print "wrote $output"
