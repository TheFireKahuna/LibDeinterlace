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
    uint decided;        // inverse telecine: whether to deinterlace, and intra, come from `record` (iv_window_decide)
    uint record;
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
                               constant PFBwdif& b, uint intra, uint x, int y) {
    int h = int(b.height);
    uint c = b.component;
    int clip_max = int(b.clip_max);
    #define CUR(r)  pf_bw_read(cur, b.cur, c, x, y + (r))
    #define PREV(r) pf_bw_read(prev, b.prev, c, x, y + (r))
    #define NEXT(r) pf_bw_read(next, b.next, c, x, y + (r))
    #define P2(r)   (b.field_parity ? PREV(r) : CUR(r))
    #define N2(r)   (b.field_parity ? CUR(r) : NEXT(r))
    if (intra != 0u) {
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
// Inverse telecine's per-picture decisions, one record per matched picture in a ring the caller indexes: the match
// fieldmatch chose (iv_match_decide), whether the weave is still combed, and what bwdif's window made of it
// (iv_window_decide).
struct IVRecord {
    uint match;          // 0 p, 1 c, 2 n
    uint combed;         // fieldmatch's combed verdict: the weave goes out flagged interlaced
    uint passthrough;    // bwdif left the picture as it is
    uint intra;          // bwdif's first field after the start: FILTER_INTRA
};

// One thread per four columns of a row pair: the current field's row, copied, and the row between, interpolated.
// Four adjacent samples share every row of their window, so its loads come from the same cache lines. A decided
// dispatch (inverse telecine) copies the current picture where the window decision left it as it is.
//
// Every component in one dispatch, the component the grid's third axis.
struct PFBwdifs {
    PFBwdif component[3];
};

kernel void pf_bwdif(device const uchar*     prev    [[buffer(0)]],
                     constant PFBwdifs&      bs      [[buffer(1)]],
                     device const IVRecord*  records [[buffer(2)]],
                     device const uchar*     cur     [[buffer(5)]],
                     device const uchar*     next    [[buffer(6)]],
                     device uchar*           out     [[buffer(7)]],
                     uint3                   gid     [[thread_position_in_grid]]) {
    constant PFBwdif& b = bs.component[gid.z];
    uint x0 = gid.x * 4u;
    if (x0 >= b.width) { return; }
    uint kept = 2u * gid.y + b.parity;
    uint interpolated = 2u * gid.y + (b.parity ^ 1u);
    uint n = min(4u, b.width - x0);
    bool copy = b.decided != 0u && records[b.record].passthrough != 0u;
    uint intra = b.decided != 0u ? records[b.record].intra : b.intra;
    for (uint i = 0u; i < n; ++i) {
        uint x = x0 + i;
        if (kept < b.height) {
            pf_bw_write(out, b, x, kept, pf_bw_read(cur, b.cur, b.component, x, int(kept)));
        }
        if (interpolated < b.height) {
            int v = copy ? pf_bw_read(cur, b.cur, b.component, x, int(interpolated))
                         : pf_bw_sample(prev, cur, next, b, intra, x, int(interpolated));
            pf_bw_write(out, b, x, interpolated, v);
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

// Every plane's in one dispatch, the plane the grid's third axis.
struct FMDiffMaps {
    FMDiffMap plane[3];
};

static inline int fm_dp(device const uchar* prv, device const uchar* nxt, constant FMDiffMap& m, int k, int u) {
    return abs(fm_read(prv, m.prv, m.plane.component, u, int(m.base) - 2 + 2 * k)
               - fm_read(nxt, m.nxt, m.plane.component, u, int(m.base2) - 2 + 2 * k));
}

// One thread per map sample of the rows y = 2, 4, ... < height - 2 (i = (y - 2) / 2), x = 1 ... width - 2.
kernel void fm_diffmap(device const uchar*   prv  [[buffer(0)]],
                       constant FMDiffMaps&  maps [[buffer(1)]],
                       device const uchar*   nxt  [[buffer(5)]],
                       device uchar*         map  [[buffer(7)]],
                       uint3                 gid  [[thread_position_in_grid]]) {
    constant FMDiffMap& m = maps.plane[gid.z];
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
    uint partial_offset; // this plane's slots in the partials buffer, six a threadgroup of the grid
};

struct FMCompares {
    FMCompare plane[3];
};

// One thread per sample of rows y = 2, 4, ... < height - 2 (k = (y - 2) / 2), x = startx ... width - startx - 1, every
// plane in one dispatch. Every threadgroup writes its six sums, zero where it covers nothing.
kernel void fm_accumulate(device const uchar*   srcb    [[buffer(0)]],
                          constant FMCompares&  cs      [[buffer(1)]],
                          device const uchar*   prevb   [[buffer(5)]],
                          device const uchar*   nextb   [[buffer(6)]],
                          device const uchar*   map     [[buffer(7)]],
                          device uint*          partial [[buffer(8)]],
                          uint3                 gid     [[thread_position_in_grid]],
                          uint3                 tgid    [[threadgroup_position_in_grid]],
                          uint3                 tgcount [[threadgroups_per_grid]],
                          uint                  tindex  [[thread_index_in_threadgroup]]) {
    constant FMCompare& c = cs.plane[gid.z];
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

// calc_combed_score's comb mask (luma, the default chroma=0) on the weave whose rows of `parity` come from `src` and
// the others from `other`. Kept inline: outlined, the compiler makes these calls with a stack frame of their own.
static inline __attribute__((always_inline)) int fm_weave(device const uchar* srcb, constant PFPicture& src, device const uchar* otherb,
                           constant PFPicture& other, uint parity, int x, int y) {
    return (uint(y) & 1u) == parity ? fm_read(srcb, src, 0u, x, y) : fm_read(otherb, other, 0u, x, y);
}

static inline __attribute__((always_inline)) bool fm_cmask(device const uchar* srcb, constant PFPicture& src, device const uchar* otherb,
                            constant PFPicture& other, uint parity, int h, int ct, int x, int y) {
    int ct6 = ct * 6;
    #define W(r) fm_weave(srcb, src, otherb, other, parity, x, (r))
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

// A pixel of rows 1 ... height - 2 is combed when the mask is on at it and the rows above and below.
static inline __attribute__((always_inline)) bool fm_combed(device const uchar* srcb, constant PFPicture& src, device const uchar* otherb,
                             constant PFPicture& other, uint parity, int w, int h, int ct, int x, int y) {
    if (x >= w || y < 1 || y >= h - 1) { return false; }
    return ct < 0 || (fm_cmask(srcb, src, otherb, other, parity, h, ct, x, y - 1)
                      && fm_cmask(srcb, src, otherb, other, parity, h, ct, x, y)
                      && fm_cmask(srcb, src, otherb, other, parity, h, ct, x, y + 1));
}

// calc_combed_score's per-cell counts for one weave (the load probe's question: is this picture combed): pixels
// combed, per 8 x 8 cell (half the default 16 x 16 block); the caller takes the largest 2 x 2 window.
struct FMComb {
    PFPicture src;
    PFPicture other;
    FMPlane plane;
    uint src_parity;     // rows of this parity come from src, the others from `other`
    int cthresh;
    uint cells_x;        // cells a row
};

kernel void fm_comb(device const uchar*  srcb   [[buffer(0)]],
                    constant FMComb&     b      [[buffer(1)]],
                    device const uchar*  otherb [[buffer(5)]],
                    device atomic_uint*  cells  [[buffer(8)]],
                    uint2                gid    [[thread_position_in_grid]]) {
    int x = int(gid.x), y = int(gid.y);
    if (!fm_combed(srcb, b.src, otherb, b.other, b.src_parity, int(b.plane.width), int(b.plane.height), b.cthresh, x, y)) {
        return;
    }
    atomic_fetch_add_explicit(&cells[uint(y >> 3) * b.cells_x + uint(x >> 3)], 1u, memory_order_relaxed);
}

// MARK: - Inverse telecine's decisions on the GPU

// The chain's state, carried from picture to picture in one buffer of 64-bit words (InverseTelecine.State).
constant uint IV_LASTSCDIFF = 0, IV_LASTN = 1, IV_FIELD_END = 2, IV_START_SET = 3, IV_START_PTS = 4, IV_LAST_PTS = 5,
              IV_LAST_DURATION = 6, IV_CYCLES = 7, IV_CYCLES_WITH_DUPLICATE = 8, IV_MAXBDIFF = 9, IV_TOTDIFF = 14;
constant uint IV_THREADS = 256;

// One picture's luma statistics for fieldmatch, read once from its rows: |prv - src| and |src - nxt| summed per 8 x 8
// cell (luma_abs_diff), and, from iv_comb, calc_combed_score's cell counts for the weaves p, c and n. A threadgroup is
// one cell, so each cell's values are written once, whole.
struct IVLuma {
    PFPicture prv;
    PFPicture src;
    PFPicture nxt;
    uint width;
    uint height;
    uint pn_parity;      // the weaves p and n take their rows of this parity from src (1 - field)
    int cthresh;
    uint cells_x;
    uint cells_y;
};

kernel void iv_luma(device const uchar*  prvb  [[buffer(0)]],
                    constant IVLuma&     l     [[buffer(1)]],
                    device const uchar*  srcb  [[buffer(5)]],
                    device const uchar*  nxtb  [[buffer(6)]],
                    device uint*         luma  [[buffer(7)]],
                    uint2                gid   [[thread_position_in_grid]],
                    uint2                tgid  [[threadgroup_position_in_grid]],
                    uint                 tindex [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint sums[2];
    if (tindex < 2u) { atomic_store_explicit(&sums[tindex], 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int x = int(gid.x), y = int(gid.y);
    if (x < int(l.width) && y < int(l.height)) {
        int s = fm_read(srcb, l.src, 0u, x, y);
        uint dp = uint(abs(fm_read(prvb, l.prv, 0u, x, y) - s)), dn = uint(abs(s - fm_read(nxtb, l.nxt, 0u, x, y)));
        if (dp) { atomic_fetch_add_explicit(&sums[0], dp, memory_order_relaxed); }
        if (dn) { atomic_fetch_add_explicit(&sums[1], dn, memory_order_relaxed); }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tindex < 2u) {
        luma[tindex * l.cells_x * l.cells_y + tgid.y * l.cells_x + tgid.x] = atomic_load_explicit(&sums[tindex], memory_order_relaxed);
    }
}

struct IVScene {
    uint cells_x;
    uint cells_y;
    uint reset;          // the chain starts over (a seek): fieldmatch's scene-change state, bwdif's, decimate's
    uint pad;
    long emitted;        // the pictures fieldmatch has put out before this one
    long scthresh;
};

// fieldmatch's scene-change test (combmatch=sc) from the luma differences. Comb scores are read only on a scene change,
// so the test is also the comb kernel's dispatch: its threadgroups, every cell or none, then the flag the decision
// reads. One threadgroup of IV_THREADS.
kernel void iv_scene(constant IVScene&   s     [[buffer(1)]],
                     device const uint*  luma  [[buffer(5)]],
                     device long*        state [[buffer(7)]],
                     device uint*        scene [[buffer(8)]],
                     uint                tid   [[thread_index_in_threadgroup]]) {
    threadgroup ulong sums[2][IV_THREADS];
    ulong prevSum = 0, nextSum = 0;
    uint cellCount = s.cells_x * s.cells_y;
    for (uint i = tid; i < cellCount; i += IV_THREADS) {
        prevSum += ulong(luma[i]);
        nextSum += ulong(luma[cellCount + i]);
    }
    sums[0][tid] = prevSum;
    sums[1][tid] = nextSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = IV_THREADS / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            sums[0][tid] += sums[0][tid + stride];
            sums[1][tid] += sums[1][tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid != 0u) { return; }
    if (s.reset != 0u) {
        state[IV_LASTSCDIFF] = 0; state[IV_LASTN] = 0; state[IV_FIELD_END] = 0;
        state[IV_START_SET] = 0; state[IV_START_PTS] = 0; state[IV_LAST_PTS] = 0; state[IV_LAST_DURATION] = 0;
    }
    long prevDiff = long(sums[0][0]), nextDiff = long(sums[1][0]);
    bool sc = false;
    if (state[IV_LASTN] == s.emitted - 1) {
        if (state[IV_LASTSCDIFF] > s.scthresh) { sc = true; }
    } else if (prevDiff > s.scthresh) {
        sc = true;
    }
    if (!sc) {
        state[IV_LASTN] = s.emitted;
        state[IV_LASTSCDIFF] = nextDiff;
        sc = nextDiff > s.scthresh;
    }
    scene[0] = sc ? s.cells_x : 0u;
    scene[1] = sc ? s.cells_y : 0u;
    scene[2] = 1u;
    scene[3] = sc ? 1u : 0u;
}

// calc_combed_score's cell counts for the weaves p, c and n, on a scene change alone (iv_scene's dispatch).
kernel void iv_comb(device const uchar*  prvb  [[buffer(0)]],
                    constant IVLuma&     l     [[buffer(1)]],
                    device const uchar*  srcb  [[buffer(5)]],
                    device const uchar*  nxtb  [[buffer(6)]],
                    device uint*         cells [[buffer(8)]],
                    uint2                gid   [[thread_position_in_grid]],
                    uint2                tgid  [[threadgroup_position_in_grid]],
                    uint                 tindex [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint sums[3];
    if (tindex < 3u) { atomic_store_explicit(&sums[tindex], 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int x = int(gid.x), y = int(gid.y), w = int(l.width), h = int(l.height);
    if (x < w && y < h) {
        // Weaves by match: p (rows not of pn_parity from prv), c (src alone), n (from nxt).
        if (fm_combed(srcb, l.src, prvb, l.prv, l.pn_parity, w, h, l.cthresh, x, y)) {
            atomic_fetch_add_explicit(&sums[0], 1u, memory_order_relaxed);
        }
        if (fm_combed(srcb, l.src, srcb, l.src, 0u, w, h, l.cthresh, x, y)) {
            atomic_fetch_add_explicit(&sums[1], 1u, memory_order_relaxed);
        }
        if (fm_combed(srcb, l.src, nxtb, l.nxt, l.pn_parity, w, h, l.cthresh, x, y)) {
            atomic_fetch_add_explicit(&sums[2], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tindex < 3u) {
        cells[tindex * l.cells_x * l.cells_y + tgid.y * l.cells_x + tgid.x] = atomic_load_explicit(&sums[tindex], memory_order_relaxed);
    }
}

struct IVMatch {
    uint partials;       // fm_accumulate's words: six a threadgroup, every plane
    uint cells_x;        // iv_comb's cells: a comb grid per weave p, c, n
    uint cells_y;
    uint record;
    uint window;         // bwdif's window has a picture to decide: window_prev / window_cur, with this one as next
    uint window_prev;
    uint window_cur;
    uint prev_repeat;
    uint next_repeat;
    uint window_start;   // the window's first output since the start: its first deinterlaced field is intra
};

// vf_fieldmatch.c compare_fields' decision from its six accumulations, between match1 and match2.
static inline uint iv_compare(thread const ulong* a, uint match1, uint match2) {
    ulong accumPm = a[1], accumNm = a[4];
    if (accumPm < 500 && accumNm < 500 && (a[2] >= 500 || a[5] >= 500) && max(a[2], a[5]) > 3 * min(a[2], a[5])) {
        accumPm = a[2];
        accumNm = a[5];
    }
    int norm1 = int((float(a[0]) / 6.0f) + 0.5f), norm2 = int((float(a[3]) / 6.0f) + 0.5f);
    int mtn1 = int((float(accumPm) / 6.0f) + 0.5f), mtn2 = int((float(accumNm) / 6.0f) + 0.5f);
    float c1 = float(max(norm1, norm2)) / float(max(min(norm1, norm2), 1));
    float c2 = float(max(mtn1, mtn2)) / float(max(min(mtn1, mtn2), 1));
    float mr = float(max(mtn1, mtn2)) / float(max(max(norm1, norm2), 1));
    if (((mtn1 >= 500 || mtn2 >= 500) && (mtn1 * 2 < mtn2 * 1 || mtn2 * 2 < mtn1 * 1))
        || ((mtn1 >= 1000 || mtn2 >= 1000) && (mtn1 * 3 < mtn2 * 2 || mtn2 * 3 < mtn1 * 2))
        || ((mtn1 >= 2000 || mtn2 >= 2000) && (mtn1 * 5 < mtn2 * 4 || mtn2 * 5 < mtn1 * 4))
        || ((mtn1 >= 4000 || mtn2 >= 4000) && c2 > c1)) {
        return mtn1 > mtn2 ? match2 : match1;
    }
    // The C compares mr, a float, with the double 0.005: 0.005f lies just below 0.005, so `mr > 0.005f` is the same test.
    if (mr > 0.005f && max(mtn1, mtn2) > 150 && (mtn1 * 2 < mtn2 * 1 || mtn2 * 2 < mtn1 * 1)) {
        return mtn1 > mtn2 ? match2 : match1;
    }
    return norm1 > norm2 ? match2 : match1;
}

// bwdif's window at frame rate (yadif_common.c ff_yadif_filter_frame, deint=interlaced) for the matched picture
// `cur`: as it is when it, or a progressive repeat-field neighbour, is not combed, else its first field deinterlaced,
// intra while the stream's start is pending (YADIF_FIELD_END, which the window's first picture sets; the end of the
// stream replaces it, YADIF_FIELD_BACK_END).
static inline void iv_window(device long* state, device IVRecord* records, uint prev, uint cur, uint next,
                             bool prev_repeat, bool next_repeat, bool start, bool finishing) {
    if (start) { state[IV_FIELD_END] = 1; }
    if (finishing) { state[IV_FIELD_END] = 0; }
    bool pass = records[cur].combed == 0u || (records[prev].combed == 0u && prev_repeat)
             || (records[next].combed == 0u && next_repeat);
    records[cur].passthrough = pass ? 1u : 0u;
    records[cur].intra = !pass && state[IV_FIELD_END] != 0 ? 1u : 0u;
    if (!pass) { state[IV_FIELD_END] = 0; }
}

static inline int iv_cell(device const uint* cells, constant IVMatch& m, uint grid, int y, int x) {
    if (y < 0 || x < 0 || y >= int(m.cells_y) || x >= int(m.cells_x)) { return 0; }
    return int(cells[grid * m.cells_x * m.cells_y + uint(y) * m.cells_x + uint(x)]);
}

// One picture's fieldmatch decision (filter_frame at the defaults): the sums reduced, compare_fields(c, p), then on
// iv_scene's scene change checkmm(match, n) over the comb counts and the combed verdict, into the picture's record; then,
// with this picture as bwdif's next, the window's decision for the one before it. One threadgroup of IV_THREADS.
kernel void iv_match_decide(device const uint*  partials [[buffer(0)]],
                            constant IVMatch&   m        [[buffer(1)]],
                            device const uint*  cells    [[buffer(5)]],
                            device const uint*  scene    [[buffer(6)]],
                            device long*        state    [[buffer(7)]],
                            device IVRecord*    records  [[buffer(8)]],
                            uint                tid      [[thread_index_in_threadgroup]]) {
    threadgroup ulong sums[6][IV_THREADS];
    threadgroup int combs[3][IV_THREADS];
    bool sc = scene[3] != 0u;
    ulong own[6] = {0, 0, 0, 0, 0, 0};
    for (uint i = tid; i < m.partials; i += IV_THREADS) { own[i % 6u] += ulong(partials[i]); }
    int best[3] = {0, 0, 0};
    // The comb counts exist on a scene change alone; otherwise every score is fieldmatch's -1, unread.
    uint windows = sc ? (m.cells_y + 1u) * (m.cells_x + 1u) : 0u;
    for (uint w = tid; w < windows; w += IV_THREADS) {
        int y = int(w / (m.cells_x + 1u)) - 1, x = int(w % (m.cells_x + 1u)) - 1;
        for (uint g = 0; g < 3u; ++g) {
            int v = iv_cell(cells, m, g, y, x) + iv_cell(cells, m, g, y, x + 1) + iv_cell(cells, m, g, y + 1, x)
                  + iv_cell(cells, m, g, y + 1, x + 1);
            best[g] = max(best[g], v);
        }
    }
    for (uint k = 0; k < 6u; ++k) { sums[k][tid] = own[k]; }
    for (uint g = 0; g < 3u; ++g) { combs[g][tid] = best[g]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = IV_THREADS / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            for (uint k = 0; k < 6u; ++k) { sums[k][tid] += sums[k][tid + stride]; }
            for (uint g = 0; g < 3u; ++g) { combs[g][tid] = max(combs[g][tid], combs[g][tid + stride]); }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid != 0u) { return; }
    ulong a[6] = {sums[0][0], sums[1][0], sums[2][0], sums[3][0], sums[4][0], sums[5][0]};
    // Matches: 0 p, 1 c, 2 n.
    uint match = iv_compare(a, 1u, 0u);
    bool combed = false;
    if (sc) {
        // checkmm(match, n): the comb grids are the weaves p, c, n, by match.
        int c1 = combs[match][0], c2 = combs[2][0];
        if ((c2 * 3 < c1 || (c2 * 2 < c1 && c1 > 80)) && abs(c2 - c1) >= 30 && c2 < 80) { match = 2u; }
        combed = combs[match][0] >= 80;
    }
    records[m.record] = IVRecord{match, combed ? 1u : 0u, 0u, 0u};
    if (m.window != 0u) {
        iv_window(state, records, m.window_prev, m.window_cur, m.record, m.prev_repeat != 0u, m.next_repeat != 0u,
                  m.window_start != 0u, false);
    }
}

// The window's decision for the stream's last picture, its own successor standing in for the next (end of stream).
struct IVWindowEnd {
    uint prev;
    uint cur;
    uint prev_repeat;
    uint cur_repeat;
    uint start;
};

kernel void iv_window_end(constant IVWindowEnd&  w       [[buffer(1)]],
                          device long*           state   [[buffer(7)]],
                          device IVRecord*       records [[buffer(8)]]) {
    iv_window(state, records, w.prev, w.cur, w.cur, w.prev_repeat != 0u, w.cur_repeat != 0u, w.start != 0u, true);
}

// create_weave_frame of the match the record holds, every plane in one dispatch: c is the picture itself, p and n take
// the rows not of `src_parity` from the previous or next picture.
struct IVWeavePlane {
    FMPlane plane;
    uint out_offset;
    uint out_linesize;
};

struct IVWeave {
    PFPicture prv;
    PFPicture src;
    PFPicture nxt;
    IVWeavePlane plane[3];
    uint src_parity;     // 1 - field
    uint record;
};

kernel void iv_weave(device const uchar*     prvb    [[buffer(0)]],
                     constant IVWeave&       w       [[buffer(1)]],
                     device const IVRecord*  records [[buffer(2)]],
                     device const uchar*     srcb    [[buffer(5)]],
                     device const uchar*     nxtb    [[buffer(6)]],
                     device uchar*           out     [[buffer(7)]],
                     uint3                   gid     [[thread_position_in_grid]]) {
    constant IVWeavePlane& p = w.plane[gid.z];
    if (gid.x >= p.plane.width || gid.y >= p.plane.height) { return; }
    int x = int(gid.x), y = int(gid.y);
    uint match = records[w.record].match;
    int v;
    if (match == 1u || (gid.y & 1u) == w.src_parity) {
        v = fm_read(srcb, w.src, p.plane.component, x, y);
    } else if (match == 0u) {
        v = fm_read(prvb, w.prv, p.plane.component, x, y);
    } else {
        v = fm_read(nxtb, w.nxt, p.plane.component, x, y);
    }
    out[p.out_offset + gid.y * p.out_linesize + gid.x] = uchar(v);
}

// calc_diffs' block sums, |a - b| over every plane (chroma on): one threadgroup per block of half blockx x half blocky
// luma samples with its chroma, written whole.
struct IVBlocksPlane {
    FMPlane plane;
    uint block_w;
    uint block_h;
};

struct IVBlocks {
    PFPicture a;
    PFPicture b;
    IVBlocksPlane plane[3];
    uint planes;
    uint blocks_x;
};

kernel void iv_blocks(device const uchar*  ab     [[buffer(0)]],
                      constant IVBlocks&   d      [[buffer(1)]],
                      device const uchar*  bb     [[buffer(5)]],
                      device uint*         blocks [[buffer(8)]],
                      uint2                tid2   [[thread_position_in_threadgroup]],
                      uint2                tgid   [[threadgroup_position_in_grid]],
                      uint                 tindex [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint sum;
    if (tindex == 0u) { atomic_store_explicit(&sum, 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint v = 0u;
    for (uint c = 0; c < d.planes; ++c) {
        constant IVBlocksPlane& p = d.plane[c];
        if (tid2.x >= p.block_w || tid2.y >= p.block_h) { continue; }
        uint x = tgid.x * p.block_w + tid2.x, y = tgid.y * p.block_h + tid2.y;
        if (x < p.plane.width && y < p.plane.height) {
            v += uint(abs(fm_read(ab, d.a, p.plane.component, int(x), int(y)) - fm_read(bb, d.b, p.plane.component, int(x), int(y))));
        }
    }
    if (v) { atomic_fetch_add_explicit(&sum, v, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tindex == 0u) { blocks[tgid.y * d.blocks_x + tgid.x] = atomic_load_explicit(&sum, memory_order_relaxed); }
}

// vf_decimate.c: calc_diffs' result for the cycle's slot (the largest sum over 2 x 2 neighbouring blocks, and the
// total; none for the stream's first picture, which has no predecessor), then, at the end of a cycle, filter_frame's
// choice of the picture to drop and every other one's timestamp, into `results` ({pts, kept} a picture). `rescaled` is
// each picture's pts on the output base (av_rescale_q, the caller's), `valid` whether it had one. One threadgroup.
struct IVDecimate {
    uint blocks_x;
    uint blocks_y;
    uint slot;           // the cycle's slot of this picture's differences
    uint diffs;          // whether this picture brings differences (0 at the end of the stream)
    uint none;           // no predecessor
    uint count;          // pictures in the cycle when it ends here: 5, or fewer at the end of the stream; 0 otherwise
    uint full;           // a full cycle drops one
    uint result;         // the cycle's first result slot
    long dupthresh;
    long scthresh;
    long rescaled[5];
    uint valid[5];
};

kernel void iv_decimate(constant IVDecimate&  d       [[buffer(1)]],
                        device const uint*    blocks  [[buffer(5)]],
                        device long*          state   [[buffer(7)]],
                        device long2*         results [[buffer(8)]],
                        uint                  tid     [[thread_index_in_threadgroup]]) {
    threadgroup long maxima[IV_THREADS];
    threadgroup long totals[IV_THREADS];
    long best = -1, total = 0;
    uint nx = d.blocks_x, ny = d.blocks_y;
    if (d.diffs != 0u && d.none == 0u) {
        for (uint i = tid; i < nx * ny; i += IV_THREADS) {
            total += long(blocks[i]);
            uint y = i / nx, x = i % nx;
            if (y + 1u < ny && x + 1u < nx) {
                best = max(best, long(blocks[i]) + long(blocks[i + 1u]) + long(blocks[i + nx]) + long(blocks[i + nx + 1u]));
            }
        }
    }
    maxima[tid] = best;
    totals[tid] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = IV_THREADS / 2u; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            maxima[tid] = max(maxima[tid], maxima[tid + stride]);
            totals[tid] += totals[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid != 0u) { return; }
    if (d.diffs != 0u) {
        long none = 0x7fffffffffffffff;
        state[IV_MAXBDIFF + d.slot] = d.none != 0u ? none : maxima[0];
        state[IV_TOTDIFF + d.slot] = d.none != 0u ? none : totals[0];
    }
    if (d.count == 0u) { return; }
    int drop = -1;
    if (d.full != 0u) {
        int scpos = -1, duppos = -1, lowest = 0;
        for (int i = 0; i < 5; ++i) {
            if (state[IV_TOTDIFF + i] > d.scthresh) { scpos = i; }
            if (state[IV_MAXBDIFF + i] < state[IV_MAXBDIFF + lowest]) { lowest = i; }
        }
        if (state[IV_MAXBDIFF + lowest] < d.dupthresh) { duppos = lowest; }
        state[IV_CYCLES] += 1;
        if (duppos >= 0) { state[IV_CYCLES_WITH_DUPLICATE] += 1; }
        drop = scpos >= 0 && duppos < 0 ? scpos : lowest;
    }
    for (uint i = 0; i < d.count; ++i) {
        if (int(i) == drop) {
            results[d.result + i] = long2(0, 0);
            continue;
        }
        if (d.valid[i] != 0u && state[IV_START_SET] == 0) {
            state[IV_START_PTS] = d.rescaled[i];
            state[IV_START_SET] = 1;
        }
        long pts = state[IV_LAST_DURATION] != 0 ? state[IV_LAST_PTS] + state[IV_LAST_DURATION] : state[IV_START_PTS];
        state[IV_LAST_DURATION] = 1;
        state[IV_LAST_PTS] = pts;
        results[d.result + i] = long2(pts, 1);
    }
}
"""
}
