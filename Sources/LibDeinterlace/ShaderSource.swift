// The kernels in this file are transcriptions of FFmpeg's libavfilter: bwdifdsp.c and vf_bwdif.c's row selection
//   Copyright (C) 2016 Thomas Mundt <loudmax@yahoo.de>
//   Copyright (C) 2006-2011 Michael Niedermayer <michaelni@gmx.at>
//   Copyright (C) 2012 British Broadcasting Corporation, All Rights Reserved
// and vf_fieldmatch.c, vf_decimate.c
//   Copyright (c) 2012 Fredrik Mellbin
//   Copyright (c) 2013 Clément Bœsch
// This library is free software; you can redistribute it and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either version 2.1 of the License, or (at your option)
// any later version. See LICENSE.

import Foundation

/// Metal source for the library's kernels. bwdif has no specification beyond FFmpeg's C, and neither do fieldmatch and
/// decimate, so the C is the reference: the same integer arithmetic, sample for sample, read through a descriptor
/// reader specialised per layout with function constants (so a semi-planar or packed picture reads as its planar
/// equivalent).
enum ShaderSource {
    static let source = """
#include <metal_stdlib>
using namespace metal;

// MARK: - Specialisation: the picture layout (PictureLayout.specialisation)

constant uint LD_C0_PLANE [[function_constant(0)]];  constant uint LD_C0_STEP [[function_constant(1)]];
constant uint LD_C0_OFFSET [[function_constant(2)]]; constant uint LD_C0_SHIFT [[function_constant(3)]];
constant uint LD_C0_DEPTH [[function_constant(4)]];
constant uint LD_C1_PLANE [[function_constant(5)]];  constant uint LD_C1_STEP [[function_constant(6)]];
constant uint LD_C1_OFFSET [[function_constant(7)]]; constant uint LD_C1_SHIFT [[function_constant(8)]];
constant uint LD_C1_DEPTH [[function_constant(9)]];
constant uint LD_C2_PLANE [[function_constant(10)]]; constant uint LD_C2_STEP [[function_constant(11)]];
constant uint LD_C2_OFFSET [[function_constant(12)]]; constant uint LD_C2_SHIFT [[function_constant(13)]];
constant uint LD_C2_DEPTH [[function_constant(14)]];
constant bool LD_BIG_ENDIAN [[function_constant(15)]];

// One component at column x of a plane row: the bytes its step and offset name, shifted and masked to its depth.
static inline uint ld_read_row(device const uchar* row, uint step, uint offset, uint shift, uint depth, uint x) {
    device const uchar* q = row + x * step + offset;
    uint bits = shift + depth;
    uint v;
    if (bits <= 8u) {
        v = uint(q[0]);
    } else if (bits <= 16u) {
        v = LD_BIG_ENDIAN ? (uint(q[0]) << 8 | uint(q[1])) : (uint(q[1]) << 8 | uint(q[0]));
    } else {
        v = LD_BIG_ENDIAN ? (uint(q[0]) << 24 | uint(q[1]) << 16 | uint(q[2]) << 8 | uint(q[3]))
                          : (uint(q[3]) << 24 | uint(q[2]) << 16 | uint(q[1]) << 8 | uint(q[0]));
    }
    return (v >> shift) & ((1u << depth) - 1u);
}


// MARK: - Deinterlacing (bwdif)

struct PFPicture {
    uint plane_offset[4];
    uint linesize[4];
};

struct PFBwdif {
    PFPicture prev;
    PFPicture cur;
    PFPicture next;
    uint width;          // the component's plane
    uint height;
    uint component;      // 0, 1, 2
    uint parity;         // rows with (y ^ parity) & 1 are interpolated, the others are the current field's
    uint field_parity;   // vf_bwdif's parity ^ tff: prev2 / next2 are prev / cur when 1, cur / next when 0
    uint intra;          // the stream's first or last field: FILTER_INTRA, the current field alone
    uint clip_max;       // (1 << depth) - 1
    uint out_offset;     // the scratch plane
    uint out_linesize;
    uint out_wide;       // two bytes a sample
};

// bwdifdsp.c's coefficients, over 2^13.
constant int PF_BW_LF0 = 4309, PF_BW_LF1 = 213;
constant int PF_BW_HF0 = 5570, PF_BW_HF1 = 3801, PF_BW_HF2 = 1016;
constant int PF_BW_SP0 = 5077, PF_BW_SP1 = 981;

static inline int pf_bw_read(device const uchar* s, constant PFPicture& p, uint c, uint x, int y) {
    uint plane = c == 0u ? LD_C0_PLANE : (c == 1u ? LD_C1_PLANE : LD_C2_PLANE);
    device const uchar* row = s + p.plane_offset[plane] + uint(y) * p.linesize[plane];
    if (c == 0u) { return int(ld_read_row(row, LD_C0_STEP, LD_C0_OFFSET, LD_C0_SHIFT, LD_C0_DEPTH, x)); }
    if (c == 1u) { return int(ld_read_row(row, LD_C1_STEP, LD_C1_OFFSET, LD_C1_SHIFT, LD_C1_DEPTH, x)); }
    return int(ld_read_row(row, LD_C2_STEP, LD_C2_OFFSET, LD_C2_SHIFT, LD_C2_DEPTH, x));
}

// One interpolated sample at (x, y). The reference offsets are vf_bwdif.c's: mirrored at the picture's top and
// bottom for intra and edge rows, plain +-1 ... +-4 rows elsewhere.
static inline int pf_bw_sample(device const uchar* prev, device const uchar* cur, device const uchar* next,
                               constant PFBwdif& b, uint x, int y) {
    int h = int(b.height);
    uint c = b.component;
    int clip_max = int(b.clip_max);
    #define CUR(r)  pf_bw_read(cur, b.cur, c, x, y + (r))
    #define PREV(r) pf_bw_read(prev, b.prev, c, x, y + (r))
    #define NEXT(r) pf_bw_read(next, b.next, c, x, y + (r))
    #define P2(r)   (b.field_parity ? PREV(r) : CUR(r))
    #define N2(r)   (b.field_parity ? CUR(r) : NEXT(r))
    if (b.intra != 0u) {
        int prefs = y + 1 < h ? 1 : -1, mrefs = y > 0 ? -1 : 1;
        int prefs3 = y + 3 < h ? 3 : -1, mrefs3 = y > 2 ? -3 : 1;
        int interpol = (PF_BW_SP0 * (CUR(mrefs) + CUR(prefs)) - PF_BW_SP1 * (CUR(mrefs3) + CUR(prefs3))) >> 13;
        return clamp(interpol, 0, clip_max);
    }
    bool edge = y < 4 || y + 5 > h;
    int prefs = edge && y + 1 >= h ? -1 : 1;
    int mrefs = edge && y <= 0 ? 1 : -1;
    // FILTER1
    int c0 = CUR(mrefs);
    int p2 = P2(0), n2 = N2(0);
    int d = (p2 + n2) >> 1;
    int e = CUR(prefs);
    int temporal_diff0 = abs(p2 - n2);
    int temporal_diff1 = (abs(PREV(mrefs) - c0) + abs(PREV(prefs) - e)) >> 1;
    int temporal_diff2 = (abs(NEXT(mrefs) - c0) + abs(NEXT(prefs) - e)) >> 1;
    int diff = max(max(temporal_diff0 >> 1, temporal_diff1), temporal_diff2);
    if (diff == 0) { return d; }
    // SPAT_CHECK: every line row, and edge rows away from the outermost two (vf_bwdif.c's `spat`).
    if (!edge || !(y < 2 || y + 3 > h)) {
        int bb = ((P2(-2) + N2(-2)) >> 1) - c0;
        int f = ((P2(2) + N2(2)) >> 1) - e;
        int dc = d - c0;
        int de = d - e;
        int mx = max(max(de, dc), min(bb, f));
        int mn = min(min(de, dc), max(bb, f));
        diff = max(max(diff, mn), -mx);
    }
    int interpol;
    if (edge) {
        // FILTER_EDGE
        interpol = (c0 + e) >> 1;
    } else if (abs(c0 - e) > temporal_diff0) {
        // FILTER_LINE, high-frequency branch
        interpol = (((PF_BW_HF0 * (p2 + n2)
                      - PF_BW_HF1 * (P2(-2) + N2(-2) + P2(2) + N2(2))
                      + PF_BW_HF2 * (P2(-4) + N2(-4) + P2(4) + N2(4))) >> 2)
                    + PF_BW_LF0 * (c0 + e) - PF_BW_LF1 * (CUR(-3) + CUR(3))) >> 13;
    } else {
        interpol = (PF_BW_SP0 * (c0 + e) - PF_BW_SP1 * (CUR(-3) + CUR(3))) >> 13;
    }
    #undef CUR
    #undef PREV
    #undef NEXT
    #undef P2
    #undef N2
    // FILTER2
    if (interpol > d + diff) {
        interpol = d + diff;
    } else if (interpol < d - diff) {
        interpol = d - diff;
    }
    return clamp(interpol, 0, clip_max);
}

static inline void pf_bw_write(device uchar* out, constant PFBwdif& b, uint x, uint y, int v) {
    device uchar* row = out + b.out_offset + y * b.out_linesize;
    if (b.out_wide != 0u) {
        *(device ushort*)(row + 2u * x) = ushort(v);
    } else {
        row[x] = uchar(v);
    }
}

// One thread per four columns of a row pair: the current field's row, copied, and the row between, interpolated.
// Four adjacent samples share every row of their window, so its loads come from the same cache lines.
kernel void pf_bwdif(device const uchar*  prev  [[buffer(0)]],
                     constant PFBwdif&    b     [[buffer(1)]],
                     device const uchar*  cur   [[buffer(5)]],
                     device const uchar*  next  [[buffer(6)]],
                     device uchar*        out   [[buffer(7)]],
                     uint2                gid   [[thread_position_in_grid]]) {
    uint x0 = gid.x * 4u;
    if (x0 >= b.width) { return; }
    uint kept = 2u * gid.y + b.parity;
    uint interpolated = 2u * gid.y + (b.parity ^ 1u);
    uint n = min(4u, b.width - x0);
    for (uint i = 0u; i < n; ++i) {
        uint x = x0 + i;
        if (kept < b.height) {
            pf_bw_write(out, b, x, kept, pf_bw_read(cur, b.cur, b.component, x, int(kept)));
        }
        if (interpolated < b.height) {
            pf_bw_write(out, b, x, interpolated, pf_bw_sample(prev, cur, next, b, x, int(interpolated)));
        }
    }
}


// MARK: - Inverse telecine (fieldmatch, decimate)

struct FMPlane {
    uint width;          // the component's plane
    uint height;
    uint component;
};

static inline int fm_read(device const uchar* s, constant PFPicture& p, uint c, int x, int y) {
    return pf_bw_read(s, p, c, uint(x), y);
}

// build_diff_map: dp(k, u) is row k of the C's tbuffer, |prv - nxt| on the field rows base - 2 + 2k (base2 for nxt).
struct FMDiffMap {
    PFPicture prv;
    PFPicture nxt;
    FMPlane plane;
    uint base;           // the row the C passes as prvp
    uint base2;          // the row the C passes as nxtp
    uint map_start;      // the map row the C passes as dstp
    uint map_offset;     // the plane's map (one byte a sample, width-wide rows)
};

static inline int fm_dp(device const uchar* prv, device const uchar* nxt, constant FMDiffMap& m, int k, int u) {
    return abs(fm_read(prv, m.prv, m.plane.component, u, int(m.base) - 2 + 2 * k)
               - fm_read(nxt, m.nxt, m.plane.component, u, int(m.base2) - 2 + 2 * k));
}

// One thread per map sample of the rows y = 2, 4, ... < height - 2 (i = (y - 2) / 2), x = 1 ... width - 2.
kernel void fm_diffmap(device const uchar*  prv  [[buffer(0)]],
                       constant FMDiffMap&  m    [[buffer(1)]],
                       device const uchar*  nxt  [[buffer(5)]],
                       device uchar*        map  [[buffer(7)]],
                       uint2                gid  [[thread_position_in_grid]]) {
    int width = int(m.plane.width), height = int(m.plane.height);
    int x = int(gid.x) + 1;
    int i = int(gid.y);
    int y = 2 + 2 * i;
    if (x >= width - 1 || y >= height - 2) { return; }
    #define DP(k, u) fm_dp(prv, nxt, m, (k), (u))
    int diff = DP(i + 1, x);
    uchar v = 0;
    if (diff > 3) {
        int count = 0;
        for (int u = x - 1; u < x + 2 && count < 2; u++) {
            count += DP(i, u) > 3;
            count += DP(i + 1, u) > 3;
            count += DP(i + 2, u) > 3;
        }
        if (count > 1) {
            v = 1;
            if (diff > 19) {
                int upper = 0, lower = 0;
                count = 0;
                for (int u = x - 1; u < x + 2 && count < 6; u++) {
                    if (DP(i, u) > 19) { count++; upper = 1; }
                    if (DP(i + 1, u) > 19) { count++; }
                    if (DP(i + 2, u) > 19) { count++; lower = 1; }
                }
                if (count > 3) {
                    if (upper && lower) {
                        v |= 1 << 1;
                    } else {
                        int upper2 = 0, lower2 = 0;
                        for (int u = max(x - 4, 0); u < min(x + 5, width); u++) {
                            if (y != 2 && DP(i - 1, u) > 19) { upper2 = 1; }
                            if (DP(i, u) > 19) { upper = 1; }
                            if (DP(i + 2, u) > 19) { lower = 1; }
                            if (y != height - 4 && DP(i + 3, u) > 19) { lower2 = 1; }
                        }
                        if ((upper && (lower || upper2)) || (lower && (upper || lower2))) {
                            v |= 1 << 1;
                        } else if (count > 5) {
                            v |= 1 << 2;
                        }
                    }
                }
            }
        }
    }
    #undef DP
    map[m.map_offset + uint(int(m.map_start) + 2 * i) * uint(width) + uint(x)] = v;
}

// compare_fields' accumulations: accumPc, accumPm, accumPml, accumNc, accumNm, accumNml.
struct FMCompare {
    PFPicture src;
    PFPicture prev;      // select_frame(match1)
    PFPicture next;      // select_frame(match2)
    FMPlane plane;
    uint fbase1;         // get_field_base(match1, field)
    uint fbase2;         // get_field_base(match2, field)
    uint startx;
    uint map_offset;
    uint partial_offset; // this dispatch's slots in the partials buffer, six a threadgroup
};

// One thread per sample of rows y = 2, 4, ... < height - 2 (k = (y - 2) / 2), x = startx ... width - startx - 1.
kernel void fm_accumulate(device const uchar*  srcb    [[buffer(0)]],
                          constant FMCompare&  c       [[buffer(1)]],
                          device const uchar*  prevb   [[buffer(5)]],
                          device const uchar*  nextb   [[buffer(6)]],
                          device const uchar*  map     [[buffer(7)]],
                          device uint*         partial [[buffer(8)]],
                          uint2                gid     [[thread_position_in_grid]],
                          uint2                tgid    [[threadgroup_position_in_grid]],
                          uint2                tgcount [[threadgroups_per_grid]],
                          uint                 tindex  [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint sums[6];
    if (tindex < 6u) { atomic_store_explicit(&sums[tindex], 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int width = int(c.plane.width), height = int(c.plane.height);
    int x = int(c.startx) + int(gid.x);
    int k = int(gid.y);
    int y = 2 + 2 * k;
    if (x < width - int(c.startx) && y < height - 2) {
        uint comp = c.plane.component;
        int m0 = map[c.map_offset + uint(int(c.fbase1) + 2 * k) * uint(width) + uint(x)];
        int m1 = map[c.map_offset + uint(int(c.fbase1) + 2 * k + 2) * uint(width) + uint(x)];
        if (m0 > 0 || m1 > 0) {
            int f1 = int(c.fbase1);
            int temp1 = fm_read(srcb, c.src, comp, x, f1 - 1 + 2 * k) + (fm_read(srcb, c.src, comp, x, f1 + 1 + 2 * k) << 2)
                      + fm_read(srcb, c.src, comp, x, f1 + 3 + 2 * k);
            int prvpf = fm_read(prevb, c.prev, comp, x, f1 + 2 * k), prvnf = fm_read(prevb, c.prev, comp, x, f1 + 2 + 2 * k);
            int f2 = int(c.fbase2);
            int nxtpf = fm_read(nextb, c.next, comp, x, f2 + 2 * k), nxtnf = fm_read(nextb, c.next, comp, x, f2 + 2 + 2 * k);
            int t = abs(3 * (prvpf + prvnf) - temp1);
            if (t > 23 && ((m0 & 1) || (m1 & 1))) { atomic_fetch_add_explicit(&sums[0], uint(t), memory_order_relaxed); }
            if (t > 42) {
                if ((m0 & 2) || (m1 & 2)) { atomic_fetch_add_explicit(&sums[1], uint(t), memory_order_relaxed); }
                if ((m0 & 4) || (m1 & 4)) { atomic_fetch_add_explicit(&sums[2], uint(t), memory_order_relaxed); }
            }
            t = abs(3 * (nxtpf + nxtnf) - temp1);
            if (t > 23 && ((m0 & 1) || (m1 & 1))) { atomic_fetch_add_explicit(&sums[3], uint(t), memory_order_relaxed); }
            if (t > 42) {
                if ((m0 & 2) || (m1 & 2)) { atomic_fetch_add_explicit(&sums[4], uint(t), memory_order_relaxed); }
                if ((m0 & 4) || (m1 & 4)) { atomic_fetch_add_explicit(&sums[5], uint(t), memory_order_relaxed); }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tindex < 6u) {
        uint group = tgid.y * tgcount.x + tgid.x;
        partial[c.partial_offset + group * 6u + tindex] = atomic_load_explicit(&sums[tindex], memory_order_relaxed);
    }
}

// calc_combed_score on a weave of two pictures (luma, the default chroma=0): the comb mask, then pixels combed on three
// consecutive rows, counted per 8 x 8 cell (half the default 16 x 16 block); the CPU takes the largest 2 x 2 window.
struct FMComb {
    PFPicture src;
    PFPicture other;
    FMPlane plane;
    uint src_parity;     // rows of this parity come from src, the others from `other`
    int cthresh;
    uint cells_x;        // cells a row
};

static inline int fm_weave(device const uchar* srcb, device const uchar* otherb, constant FMComb& b, int x, int y) {
    return (uint(y) & 1u) == b.src_parity ? fm_read(srcb, b.src, 0u, x, y) : fm_read(otherb, b.other, 0u, x, y);
}

static inline bool fm_cmask(device const uchar* srcb, device const uchar* otherb, constant FMComb& b, int x, int y) {
    int h = int(b.plane.height);
    int ct = b.cthresh, ct6 = ct * 6;
    #define W(r) fm_weave(srcb, otherb, b, x, (r))
    #define FILTER(xm2, xm1, xp1, xp2) (abs(4 * W(y) - 3 * (W(y + (xm1)) + W(y + (xp1))) + (W(y + (xm2)) + W(y + (xp2)))) > ct6)
    bool on;
    if (y == 0) {
        on = abs(W(0) - W(1)) > ct && FILTER(2, 1, 1, 2);
    } else if (y == 1) {
        on = abs(W(1) - W(0)) > ct && abs(W(1) - W(2)) > ct && FILTER(2, -1, 1, 2);
    } else if (y < h - 2) {
        on = abs(W(y) - W(y - 1)) > ct && abs(W(y) - W(y + 1)) > ct && FILTER(-2, -1, 1, 2);
    } else if (y == h - 2) {
        on = abs(W(y) - W(y - 1)) > ct && abs(W(y) - W(y + 1)) > ct && FILTER(-2, -1, 1, -2);
    } else {
        on = abs(W(y) - W(y - 1)) > ct && FILTER(-2, -1, -1, -2);
    }
    #undef FILTER
    #undef W
    return on;
}

// One thread per pixel of rows 1 ... height - 2.
kernel void fm_comb(device const uchar*  srcb   [[buffer(0)]],
                    constant FMComb&     b      [[buffer(1)]],
                    device const uchar*  otherb [[buffer(5)]],
                    device atomic_uint*  cells  [[buffer(8)]],
                    uint2                gid    [[thread_position_in_grid]]) {
    int x = int(gid.x), y = int(gid.y) + 1;
    if (x >= int(b.plane.width) || y >= int(b.plane.height) - 1) { return; }
    if (b.cthresh >= 0 && !(fm_cmask(srcb, otherb, b, x, y - 1) && fm_cmask(srcb, otherb, b, x, y)
                            && fm_cmask(srcb, otherb, b, x, y + 1))) { return; }
    atomic_fetch_add_explicit(&cells[uint(y >> 3) * b.cells_x + uint(x >> 3)], 1u, memory_order_relaxed);
}

// luma_abs_diff, and calc_diffs' block sums: |a - b| summed per block of `block_w` x `block_h` samples of one plane
// (block_w 0: the whole plane in per-threadgroup partials).
struct FMDiff {
    PFPicture a;
    PFPicture b;
    FMPlane plane;
    uint block_w;
    uint block_h;
    uint blocks_x;       // blocks a row of the luma grid
};

kernel void fm_absdiff(device const uchar*  ab    [[buffer(0)]],
                       constant FMDiff&     d     [[buffer(1)]],
                       device const uchar*  bb    [[buffer(5)]],
                       device atomic_uint*  out   [[buffer(8)]],
                       uint2                gid   [[thread_position_in_grid]],
                       uint2                tgid  [[threadgroup_position_in_grid]],
                       uint2                tgcount [[threadgroups_per_grid]],
                       uint                 tindex [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint sum;
    if (tindex == 0u) { atomic_store_explicit(&sum, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int x = int(gid.x), y = int(gid.y);
    uint v = 0u;
    if (x < int(d.plane.width) && y < int(d.plane.height)) {
        v = uint(abs(fm_read(ab, d.a, d.plane.component, x, y) - fm_read(bb, d.b, d.plane.component, x, y)));
    }
    if (d.block_w == 0u) {
        if (v) { atomic_fetch_add_explicit(&sum, v, memory_order_relaxed); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tindex == 0u) {
            atomic_store_explicit(&out[tgid.y * tgcount.x + tgid.x], atomic_load_explicit(&sum, memory_order_relaxed),
                                  memory_order_relaxed);
        }
    } else if (v) {
        atomic_fetch_add_explicit(&out[uint(y) / d.block_h * d.blocks_x + uint(x) / d.block_w], v, memory_order_relaxed);
    }
}

// create_weave_frame: rows of `src_parity` from src, the others from `other`, written planar at 8 bits.
struct FMWeave {
    PFPicture src;
    PFPicture other;
    FMPlane plane;
    uint src_parity;
    uint out_offset;
    uint out_linesize;
};

kernel void fm_weave_out(device const uchar*  srcb   [[buffer(0)]],
                         constant FMWeave&    w      [[buffer(1)]],
                         device const uchar*  otherb [[buffer(5)]],
                         device uchar*        out    [[buffer(7)]],
                         uint2                gid    [[thread_position_in_grid]]) {
    if (gid.x >= w.plane.width || gid.y >= w.plane.height) { return; }
    int x = int(gid.x), y = int(gid.y);
    int v = (gid.y & 1u) == w.src_parity ? fm_read(srcb, w.src, w.plane.component, x, y)
                                         : fm_read(otherb, w.other, w.plane.component, x, y);
    out[w.out_offset + gid.y * w.out_linesize + gid.x] = uchar(v);
}
"""
}
