/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include "../common/bits.h" /* ZSTD_highbit32 */
#include "../common/compiler.h" /* ZSTD_ALIGNOF */
#include "../common/mem.h" /* S64 */
#include "../common/zstd_deps.h" /* ZSTD_memset */
#include "../common/zstd_internal.h" /* ZSTD_STATIC_ASSERT */
#include "hist.h" /* HIST_add */
#include "zstd_preSplit.h"

#if defined(__SSE2__) && !defined(ZSTD_NO_INTRINSICS)
#include <emmintrin.h>
#endif


#define BLOCKSIZE_MIN 3500
#define THRESHOLD_PENALTY_RATE 16
#define THRESHOLD_BASE (THRESHOLD_PENALTY_RATE - 2)
#define THRESHOLD_PENALTY 3

#define HASHLENGTH 2
#define HASHLOG_MAX 10
#define HASHTABLESIZE (1 << HASHLOG_MAX)
#define HASHMASK (HASHTABLESIZE - 1)
#define KNUTH 0x9e3779b9

/* for hashLog > 8, hash 2 bytes.
 * for hashLog == 8, just take the byte, no hashing.
 * The speed of this method relies on compile-time constant propagation */
FORCE_INLINE_TEMPLATE unsigned hash2(const void *p, unsigned hashLog)
{
    assert(hashLog >= 8);
    if (hashLog == 8) return (U32)((const BYTE*)p)[0];
    assert(hashLog <= HASHLOG_MAX);
    return (U32)(MEM_read16(p)) * KNUTH >> (32 - hashLog);
}


typedef struct {
  unsigned events[HASHTABLESIZE];
  size_t nbEvents;
} Fingerprint;
typedef struct {
    Fingerprint pastEvents;
    Fingerprint newEvents;
} FPStats;

static void initStats(FPStats* fpstats)
{
    ZSTD_memset(fpstats, 0, sizeof(FPStats));
}

FORCE_INLINE_TEMPLATE void
addEvents_generic(Fingerprint* fp, const void* src, size_t srcSize, size_t samplingRate, unsigned hashLog)
{
    const char* p = (const char*)src;
    size_t limit = srcSize - HASHLENGTH + 1;
    size_t n;
    assert(srcSize >= HASHLENGTH);
    for (n = 0; n < limit; n+=samplingRate) {
        fp->events[hash2(p+n, hashLog)]++;
    }
    fp->nbEvents += limit/samplingRate;
}

FORCE_INLINE_TEMPLATE void
recordFingerprint_generic(Fingerprint* fp, const void* src, size_t srcSize, size_t samplingRate, unsigned hashLog)
{
    ZSTD_memset(fp, 0, sizeof(unsigned) * ((size_t)1 << hashLog));
    fp->nbEvents = 0;
    addEvents_generic(fp, src, srcSize, samplingRate, hashLog);
}

typedef void (*RecordEvents_f)(Fingerprint* fp, const void* src, size_t srcSize);

#define FP_RECORD(_rate) ZSTD_recordFingerprint_##_rate

#define ZSTD_GEN_RECORD_FINGERPRINT(_rate, _hSize)                                 \
    static void FP_RECORD(_rate)(Fingerprint* fp, const void* src, size_t srcSize) \
    {                                                                              \
        recordFingerprint_generic(fp, src, srcSize, _rate, _hSize);                \
    }

ZSTD_GEN_RECORD_FINGERPRINT(1, 10)
ZSTD_GEN_RECORD_FINGERPRINT(5, 10)
ZSTD_GEN_RECORD_FINGERPRINT(11, 9)
ZSTD_GEN_RECORD_FINGERPRINT(43, 8)


static U64 abs64(S64 s64) { return (U64)((s64 < 0) ? -s64 : s64); }

static U64 fpDistance(const Fingerprint* fp1, const Fingerprint* fp2, unsigned hashLog)
{
    U64 distance = 0;
    size_t n;
    assert(hashLog <= HASHLOG_MAX);
    for (n = 0; n < ((size_t)1 << hashLog); n++) {
        distance +=
            abs64((S64)fp1->events[n] * (S64)fp2->nbEvents - (S64)fp2->events[n] * (S64)fp1->nbEvents);
    }
    return distance;
}

/* Compare newEvents with pastEvents
 * return 1 when considered "too different"
 */
static int compareFingerprints(const Fingerprint* ref,
                            const Fingerprint* newfp,
                            int penalty,
                            unsigned hashLog)
{
    assert(ref->nbEvents > 0);
    assert(newfp->nbEvents > 0);
    {   U64 p50 = (U64)ref->nbEvents * (U64)newfp->nbEvents;
        U64 deviation = fpDistance(ref, newfp, hashLog);
        U64 threshold = p50 * (U64)(THRESHOLD_BASE + penalty) / THRESHOLD_PENALTY_RATE;
        return deviation >= threshold;
    }
}

static void mergeEvents(Fingerprint* acc, const Fingerprint* newfp)
{
    size_t n;
    for (n = 0; n < HASHTABLESIZE; n++) {
        acc->events[n] += newfp->events[n];
    }
    acc->nbEvents += newfp->nbEvents;
}

static void flushEvents(FPStats* fpstats)
{
    size_t n;
    for (n = 0; n < HASHTABLESIZE; n++) {
        fpstats->pastEvents.events[n] = fpstats->newEvents.events[n];
    }
    fpstats->pastEvents.nbEvents = fpstats->newEvents.nbEvents;
    ZSTD_memset(&fpstats->newEvents, 0, sizeof(fpstats->newEvents));
}

static void removeEvents(Fingerprint* acc, const Fingerprint* slice)
{
    size_t n;
    for (n = 0; n < HASHTABLESIZE; n++) {
        assert(acc->events[n] >= slice->events[n]);
        acc->events[n] -= slice->events[n];
    }
    acc->nbEvents -= slice->nbEvents;
}

#define CHUNKSIZE (8 << 10)
static size_t ZSTD_splitBlock_byChunks(const void* blockStart, size_t blockSize,
                        int level,
                        void* workspace, size_t wkspSize)
{
    static const RecordEvents_f records_fs[] = {
        FP_RECORD(43), FP_RECORD(11), FP_RECORD(5), FP_RECORD(1)
    };
    static const unsigned hashParams[] = { 8, 9, 10, 10 };
    const RecordEvents_f record_f = (assert(0<=level && level<=3), records_fs[level]);
    FPStats* const fpstats = (FPStats*)workspace;
    const char* p = (const char*)blockStart;
    int penalty = THRESHOLD_PENALTY;
    size_t pos = 0;
    assert(blockSize == (128 << 10));
    assert(workspace != NULL);
    assert((size_t)workspace % ZSTD_ALIGNOF(FPStats) == 0);
    ZSTD_STATIC_ASSERT(ZSTD_SLIPBLOCK_WORKSPACESIZE >= sizeof(FPStats));
    assert(wkspSize >= sizeof(FPStats)); (void)wkspSize;

    initStats(fpstats);
    record_f(&fpstats->pastEvents, p, CHUNKSIZE);
    for (pos = CHUNKSIZE; pos <= blockSize - CHUNKSIZE; pos += CHUNKSIZE) {
        record_f(&fpstats->newEvents, p + pos, CHUNKSIZE);
        if (compareFingerprints(&fpstats->pastEvents, &fpstats->newEvents, penalty, hashParams[level])) {
            return pos;
        } else {
            mergeEvents(&fpstats->pastEvents, &fpstats->newEvents);
            if (penalty > 0) penalty--;
        }
    }
    assert(pos == blockSize);
    return blockSize;
    (void)flushEvents; (void)removeEvents;
}

/* ZSTD_splitBlock_fromBorders(): very fast strategy :
 * compare fingerprint from beginning and end of the block,
 * derive from their difference if it's preferable to split in the middle,
 * repeat the process a second time, for finer grained decision.
 * 3 times did not brought improvements, so I stopped at 2.
 * Benefits are good enough for a cheap heuristic.
 * More accurate splitting saves more, but speed impact is also more perceptible.
 * For better accuracy, use more elaborate variant *_byChunks.
 */
static size_t ZSTD_splitBlock_fromBorders(const void* blockStart, size_t blockSize,
                        void* workspace, size_t wkspSize)
{
#define SEGMENT_SIZE 512
    FPStats* const fpstats = (FPStats*)workspace;
    Fingerprint* middleEvents = (Fingerprint*)(void*)((char*)workspace + 512 * sizeof(unsigned));
    assert(blockSize == (128 << 10));
    assert(workspace != NULL);
    assert((size_t)workspace % ZSTD_ALIGNOF(FPStats) == 0);
    ZSTD_STATIC_ASSERT(ZSTD_SLIPBLOCK_WORKSPACESIZE >= sizeof(FPStats));
    assert(wkspSize >= sizeof(FPStats)); (void)wkspSize;

    initStats(fpstats);
    HIST_add(fpstats->pastEvents.events, blockStart, SEGMENT_SIZE);
    HIST_add(fpstats->newEvents.events, (const char*)blockStart + blockSize - SEGMENT_SIZE, SEGMENT_SIZE);
    fpstats->pastEvents.nbEvents = fpstats->newEvents.nbEvents = SEGMENT_SIZE;
    if (!compareFingerprints(&fpstats->pastEvents, &fpstats->newEvents, 0, 8))
        return blockSize;

    HIST_add(middleEvents->events, (const char*)blockStart + blockSize/2 - SEGMENT_SIZE/2, SEGMENT_SIZE);
    middleEvents->nbEvents = SEGMENT_SIZE;
    {   U64 const distFromBegin = fpDistance(&fpstats->pastEvents, middleEvents, 8);
        U64 const distFromEnd = fpDistance(&fpstats->newEvents, middleEvents, 8);
        U64 const minDistance = SEGMENT_SIZE * SEGMENT_SIZE / 3;
        if (abs64((S64)distFromBegin - (S64)distFromEnd) < minDistance)
            return 64 KB;
        return (distFromBegin > distFromEnd) ? 32 KB : 96 KB;
    }
}

/* round(256 * log2(1 + i/256)), for i in [0, 255]. */
static const BYTE ZSTD_splitLogFraction[256] = {
    0, 1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 16, 17, 18, 20, 21,
    22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42,
    44, 45, 46, 47, 49, 50, 51, 52, 54, 55, 56, 57, 59, 60, 61, 62,
    63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81,
    82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
    134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    150, 151, 152, 153, 154, 155, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 168, 169, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 178,
    179, 180, 181, 182, 183, 184, 185, 185, 186, 187, 188, 189, 190, 191, 192, 192,
    193, 194, 195, 196, 197, 198, 198, 199, 200, 201, 202, 203, 203, 204, 205, 206,
    207, 208, 208, 209, 210, 211, 212, 212, 213, 214, 215, 216, 216, 217, 218, 219,
    220, 220, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228, 229, 230, 231, 231,
    232, 233, 234, 234, 235, 236, 237, 238, 238, 239, 240, 241, 241, 242, 243, 244,
    244, 245, 246, 247, 247, 248, 249, 249, 250, 251, 252, 252, 253, 254, 255, 255,
};

/* Fixed-point log2 with eight fractional bits. All arguments are small
 * histogram counts, so the normalization shift cannot overflow. */
MEM_STATIC int ZSTD_splitLog256(U32 value)
{
    U32 log;
    assert(value > 0 && value < (1U << 24));
    log = ZSTD_highbit32(value);
    return (int)(log*256 + ZSTD_splitLogFraction[((value<<8)>>log)-256]);
}

/* Locate the transition more precisely than the 8 KiB fingerprint grid.
 * Compare the likelihood of samples on either side of the proposed split
 * under the past and current distributions. One-count smoothing keeps
 * missing symbols finite and regularizes these small fingerprints.
 * The best boundary minimizes the cumulative cost of using the past model.
 * Require four bits of improvement over the grid boundary, and eight bits
 * of evidence after the minimum before accepting it.
 *
 * ref already includes the current histogram: subtract it to recover the
 * past distribution. The price cache uses unused space after the four
 * temporary histograms, within the existing splitting workspace. */
static size_t ZSTD_refineSplitFast(const BYTE* src, size_t pos,
        const U16* ref, U16* hist, U32 k)
{
    enum { span = CHUNKSIZE/2, priceUnset = -32640 };
    U32 const samples = (CHUNKSIZE+41)/43;
    S16* const prices = (S16*)(hist + 4*256);
    int const constant = ZSTD_splitLog256(k*samples+256)
                       - ZSTD_splitLog256(samples+256);
    int sum = 0, minimum = 0, original = 0, seen = 0, confirmed = 0;
    size_t best = 0, n;
    const BYTE* const p = src+pos-span;
    ZSTD_STATIC_ASSERT(CHUNKSIZE == 8 KB);
    ZSTD_STATIC_ASSERT(CHUNKSIZE/2 >= BLOCKSIZE_MIN);
    assert(pos >= CHUNKSIZE && pos <= 128 KB-CHUNKSIZE);
    /* Signed 0x8080 is outside the range of any valid price. */
    ZSTD_memset(prices, 0x80, 256*sizeof(*prices));
    for (n = 0; n*43 < 2*span; ++n) {
        unsigned const symbol = p[n*43];
        int price = prices[symbol];
        if (price == priceUnset) {
            U32 const current = hist[symbol] + hist[256+symbol]
                              + hist[512+symbol] + hist[768+symbol];
            U32 const previous = ref[symbol] - current;
            assert(ref[symbol] >= current);
            price = constant + ZSTD_splitLog256(current+1)
                             - ZSTD_splitLog256(previous+1);
            assert(price > priceUnset && price < 32768);
            prices[symbol] = (S16)price;
        }
        sum += price;
        if (sum < minimum) { minimum = sum; best = n+1; }
        if (!seen && (n+1)*43 >= span) { original = sum; seen = 1; }
        if (seen && sum-minimum >= 8*256) { confirmed = 1; break; }
    }
    if (confirmed && original-minimum >= 4*256) {
        /* Midpoint between the last old and first new sample, rounded to
         * 64 bytes. Even at either end, leave at least 4 KiB in each block. */
        size_t const delta = (best*43+11) & ~(size_t)63;
        assert(delta <= 2*span);
        return pos-span+delta;
    }
    return pos;
}

/* The byte fingerprints at sampling rate 43 have fewer than 3072 samples
 * in a full block. 16-bit counters are sufficient, including the weighted
 * differences and their sum. Cancel the common nbEvents factor before
 * computing the distance; this gives the same decision as the full-width
 * fingerprints. Four independent histograms shorten dependency chains on
 * common bytes (notably spaces and zeroes).
 *
 * This is scratch space only. Merging a chunk before testing the distance
 * is harmless even when that chunk turns out to be the split point. */
static size_t
ZSTD_splitBlock_fastChunks(const void* blockStart, size_t blockSize,
                          void* workspace, size_t wkspSize)
{
    const BYTE* const p = (const BYTE*)blockStart;
    U16* const ref = (U16*)workspace;
    U16* const hist = ref+256;
    size_t pos,n;
    U32 k=1;
    int penalty=THRESHOLD_PENALTY;
    /* Match recordFingerprint_generic's convention (floor, not ceil). */
    const U32 events = (CHUNKSIZE-1)/43;
    ZSTD_STATIC_ASSERT(HASHLENGTH == 2);
    ZSTD_STATIC_ASSERT(ZSTD_SLIPBLOCK_WORKSPACESIZE >= 6*256*sizeof(U16));
    ZSTD_STATIC_ASSERT(2 * (128 KB / 43 + 128 KB / CHUNKSIZE) < 32768);
    assert(blockSize >= 16 KB && blockSize <= 128 KB);
    assert(wkspSize >= (4+2)*256*sizeof(U16));
    (void)wkspSize;
    ZSTD_memset(ref,0,256*sizeof(U16));
    for (n=0; n<CHUNKSIZE-1; n+=43) ref[p[n]]++;
    for (pos=CHUNKSIZE; pos<=blockSize-CHUNKSIZE; pos+=CHUNKSIZE,++k) {
        U32 distance=0;
        /* nbEvents >= THRESHOLD_PENALTY_RATE makes this rounded-up,
         * cancelled threshold equivalent to the original comparison. */
        U32 const threshold = (k*events*(THRESHOLD_BASE+(U32)penalty)
                              + THRESHOLD_PENALTY_RATE-1) / THRESHOLD_PENALTY_RATE;
        const BYTE* const src=p+pos;
        size_t i;
        ZSTD_memset(hist,0,4*256*sizeof(U16));
        for (n=0; n+4<=(CHUNKSIZE+41)/43; n+=4) {
            hist[0*256 + src[(n+0)*43]]++;
            hist[1*256 + src[(n+1)*43]]++;
            hist[2*256 + src[(n+2)*43]]++;
            hist[3*256 + src[(n+3)*43]]++;
        }
        for (; n<(CHUNKSIZE+41)/43; ++n) hist[src[n*43]]++;
#if defined(__SSE2__) && !defined(ZSTD_NO_INTRINSICS)
        {
        __m128i acc = _mm_setzero_si128();
        __m128i const mul = _mm_set1_epi16((short)k);
        __m128i sum;
        for (i=0;i<256;i+=8) {
            __m128i const prev = _mm_loadu_si128((const __m128i*)(ref+i));
            __m128i cur = _mm_loadu_si128((const __m128i*)(hist+i));
            __m128i diff,sign;
            cur = _mm_add_epi16(cur, _mm_loadu_si128((const __m128i*)(hist+1*256+i)));
            cur = _mm_add_epi16(cur, _mm_loadu_si128((const __m128i*)(hist+2*256+i)));
            cur = _mm_add_epi16(cur, _mm_loadu_si128((const __m128i*)(hist+3*256+i)));
            diff = _mm_sub_epi16(prev,_mm_mullo_epi16(cur,mul));
            sign = _mm_srai_epi16(diff,15);
            acc = _mm_add_epi16(acc,_mm_sub_epi16(_mm_xor_si128(diff,sign),sign));
            _mm_storeu_si128((__m128i*)(ref+i),_mm_add_epi16(prev,cur));
        }
        sum = _mm_madd_epi16(acc,_mm_set1_epi16(1));
        sum = _mm_add_epi32(sum,_mm_srli_si128(sum,8));
        sum = _mm_add_epi32(sum,_mm_srli_si128(sum,4));
        distance = (U32)_mm_cvtsi128_si32(sum);
        }
#else
        for (i=0;i<256;++i) {
            U32 const cur = hist[0*256+i] + hist[1*256+i] + hist[2*256+i] + hist[3*256+i];
            S32 const diff = (S32)ref[i] - (S32)(cur*k);
            distance += (U32)(diff<0 ? -diff : diff);
            ref[i] = (U16)(ref[i]+cur);
        }
#endif
        if (distance >= threshold) return ZSTD_refineSplitFast(p,pos,ref,hist,k);
        if (penalty>0) --penalty;
    }
    return blockSize;
}

/* This gate only decides whether to run the finer test, not where to
 * split. Its threshold is deliberately less strict than the chunk test.
 * Use small counters and omit the middle fingerprint needed by the
 * standalone border splitter: that fingerprint cannot affect this gate. */
static int ZSTD_bordersDiffer(const void* blockStart, size_t blockSize,
                             void* workspace, size_t wkspSize)
{
    U16* const a = (U16*)workspace;
    U16* const b = a + 256;
    const BYTE* const p = (const BYTE*)blockStart;
    const BYTE* const q = p + blockSize - 256;
    U32 distance = 0;
    size_t n, i;
    assert(blockSize >= 16 KB && blockSize <= 128 KB);
    assert(wkspSize >= 2*256*sizeof(U16));
    (void)wkspSize;
    ZSTD_memset(workspace, 0, 2*256*sizeof(U16));
    for (n = 0; n < 256; ++n) {
        a[p[n]]++;
        b[q[n]]++;
    }
#if defined(__SSE2__) && !defined(ZSTD_NO_INTRINSICS)
    {
        __m128i acc = _mm_setzero_si128(), sum;
        for (i = 0; i < 256; i += 8) {
            __m128i const aa = _mm_loadu_si128((const __m128i*)(a+i));
            __m128i const bb = _mm_loadu_si128((const __m128i*)(b+i));
            __m128i const diff = _mm_sub_epi16(aa, bb);
            __m128i const sign = _mm_srai_epi16(diff, 15);
            acc = _mm_add_epi16(acc, _mm_sub_epi16(_mm_xor_si128(diff, sign), sign));
        }
        /* The total distance is at most 512, so signed 16-bit sums suffice. */
        sum = _mm_madd_epi16(acc, _mm_set1_epi16(1));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 8));
        sum = _mm_add_epi32(sum, _mm_srli_si128(sum, 4));
        distance = (U32)_mm_cvtsi128_si32(sum);
    }
#else
    for (i = 0; i < 256; ++i) {
        int const d = (int)a[i] - (int)b[i];
        distance += (U32)(d < 0 ? -d : d);
    }
#endif
    return distance >= 256*12/16;
}

/* Refine the cheap estimate only when the borders indicate a change. */
size_t ZSTD_splitBlock_fast(const void* blockStart, size_t blockSize,
                           void* workspace, size_t wkspSize)
{
    if (!ZSTD_bordersDiffer(blockStart, blockSize, workspace, wkspSize))
        return blockSize;
    return ZSTD_splitBlock_fastChunks(blockStart, blockSize, workspace, wkspSize);
}

size_t ZSTD_splitBlock(const void* blockStart, size_t blockSize,
                    int level,
                    void* workspace, size_t wkspSize)
{
    DEBUGLOG(6, "ZSTD_splitBlock (level=%i)", level);
    assert(0<=level && level<=4);
    if (level == 0)
        return ZSTD_splitBlock_fromBorders(blockStart, blockSize, workspace, wkspSize);
    /* level >= 1*/
    return ZSTD_splitBlock_byChunks(blockStart, blockSize, level-1, workspace, wkspSize);
}

/* The first block may mix a file header with its payload while retaining
 * similar borders. Use the fine scan directly in that case. */
size_t ZSTD_splitBlock_fastScan(const void* blockStart, size_t blockSize,
                                void* workspace, size_t wkspSize)
{
    return ZSTD_splitBlock_fastChunks(blockStart, blockSize, workspace, wkspSize);
}
