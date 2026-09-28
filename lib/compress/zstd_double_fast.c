/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

#include "zstd_compress_internal.h"
#include "zstd_double_fast.h"

#ifndef ZSTD_EXCLUDE_DFAST_BLOCK_COMPRESSOR

#if defined(__SSE2__) && !defined(ZSTD_NO_INTRINSICS)
#include <emmintrin.h>
#endif
/* Keep the word-sized fast exit for short matches. For long-match blocks,
 * count sixteen bytes at a time, without reading beyond the input limit. */
FORCE_INLINE_TEMPLATE size_t ZSTD_dfastCount(const BYTE* pIn, const BYTE* pMatch, const BYTE* const pInLimit, int const useSimd)
{
    const BYTE* const pStart = pIn;
    if (!useSimd) return ZSTD_count(pIn,pMatch,pInLimit);
    if ((size_t)(pInLimit-pIn) >= sizeof(size_t)) {
        size_t const diff = MEM_readST(pMatch) ^ MEM_readST(pIn);
        if (diff) return ZSTD_NbCommonBytes(diff);
        pIn+=sizeof(size_t); pMatch+=sizeof(size_t);
    }
#if defined(__SSE2__) && !defined(ZSTD_NO_INTRINSICS)
    while ((size_t)(pInLimit-pIn) >= 16) {
        __m128i const a = _mm_loadu_si128((const __m128i*)pIn);
        __m128i const b = _mm_loadu_si128((const __m128i*)pMatch);
        U32 const mask = (U32)_mm_movemask_epi8(_mm_cmpeq_epi8(a,b));
        if (mask != 0xffff) return (size_t)(pIn-pStart) + (size_t)ZSTD_countTrailingZeros32(mask ^ 0xffff);
        pIn+=16; pMatch+=16;
    }
#endif
    return (size_t)(pIn-pStart) + ZSTD_count(pIn,pMatch,pInLimit);
}


static
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
void ZSTD_fillDoubleHashTableForCDict(ZSTD_MatchState_t* ms,
                              void const* end, ZSTD_dictTableLoadMethod_e dtlm)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const hashLarge = ms->hashTable;
    U32  const hBitsL = cParams->hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    U32  const mls = cParams->minMatch;
    U32* const hashSmall = ms->chainTable;
    U32  const hBitsS = cParams->chainLog + ZSTD_SHORT_CACHE_TAG_BITS;
    const BYTE* const base = ms->window.base;
    const BYTE* ip = base + ms->nextToUpdate;
    const BYTE* const iend = ((const BYTE*)end) - HASH_READ_SIZE;
    const U32 fastHashFillStep = 3;

    /* Always insert every fastHashFillStep position into the hash tables.
     * Insert the other positions into the large hash table if their entry
     * is empty.
     */
    for (; ip + fastHashFillStep - 1 <= iend; ip += fastHashFillStep) {
        U32 const curr = (U32)(ip - base);
        U32 i;
        for (i = 0; i < fastHashFillStep; ++i) {
            size_t const smHashAndTag = ZSTD_hashPtr(ip + i, hBitsS, mls);
            size_t const lgHashAndTag = ZSTD_hashPtr(ip + i, hBitsL, 8);
            if (i == 0) {
                ZSTD_writeTaggedIndex(hashSmall, smHashAndTag, curr + i);
            }
            if (i == 0 || hashLarge[lgHashAndTag >> ZSTD_SHORT_CACHE_TAG_BITS] == 0) {
                ZSTD_writeTaggedIndex(hashLarge, lgHashAndTag, curr + i);
            }
            /* Only load extra positions for ZSTD_dtlm_full */
            if (dtlm == ZSTD_dtlm_fast)
                break;
    }   }
}

static
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
void ZSTD_fillDoubleHashTableForCCtx(ZSTD_MatchState_t* ms,
                              void const* end, ZSTD_dictTableLoadMethod_e dtlm)
{
    const ZSTD_compressionParameters* const cParams = &ms->cParams;
    U32* const hashLarge = ms->hashTable;
    U32  const hBitsL = cParams->hashLog;
    U32  const mls = cParams->minMatch;
    U32* const hashSmall = ms->chainTable;
    U32  const hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* ip = base + ms->nextToUpdate;
    const BYTE* const iend = ((const BYTE*)end) - HASH_READ_SIZE;
    const U32 fastHashFillStep = 3;

    /* Always insert every fastHashFillStep position into the hash tables.
     * Insert the other positions into the large hash table if their entry
     * is empty.
     */
    for (; ip + fastHashFillStep - 1 <= iend; ip += fastHashFillStep) {
        U32 const curr = (U32)(ip - base);
        U32 i;
        for (i = 0; i < fastHashFillStep; ++i) {
            size_t const smHash = ZSTD_hashPtr(ip + i, hBitsS, mls);
            size_t const lgHash = ZSTD_hashPtr(ip + i, hBitsL, 8);
            if (i == 0)
                hashSmall[smHash] = curr + i;
            if (i == 0 || hashLarge[lgHash] == 0)
                hashLarge[lgHash] = curr + i;
            /* Only load extra positions for ZSTD_dtlm_full */
            if (dtlm == ZSTD_dtlm_fast)
                break;
        }   }
}

void ZSTD_fillDoubleHashTable(ZSTD_MatchState_t* ms,
                        const void* const end,
                        ZSTD_dictTableLoadMethod_e dtlm,
                        ZSTD_tableFillPurpose_e tfp)
{
    if (tfp == ZSTD_tfp_forCDict) {
        ZSTD_fillDoubleHashTableForCDict(ms, end, dtlm);
    } else {
        ZSTD_fillDoubleHashTableForCCtx(ms, end, dtlm);
    }
}


FORCE_INLINE_TEMPLATE
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_compressBlock_doubleFast_noDict_generic(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize, U32 const mls /* template */,
        U32 const hBitsL, U32 const hBitsS,
        int const swapTables, int const longMode, int const tuned)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = swapTables ? ms->chainTable : ms->hashTable;
    U32* const hashSmall = swapTables ? ms->hashTable : ms->chainTable;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* anchor = istart;
    const U32 endIndex = (U32)((size_t)(istart - base) + srcSize);
    /* presumes that, if there is a dictionary, it must be using Attach mode */
    const U32 prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const prefixLowest = base + prefixLowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=rep[0], offset_2=rep[1];
    U32 offsetSaved1 = 0, offsetSaved2 = 0;

    size_t mLength;
    U32 offset;
    U32 curr;

    /* how many positions to search before increasing step size */
    const size_t kStepIncr = (size_t)1 << (tuned ? 7 : kSearchStrength);
    /* the position at which to increment the step size if no match is found */
    const BYTE* nextStep;
    size_t step; /* the current step size */

    size_t hl0; /* the long hash at ip */
    size_t hl1; /* the long hash at ip1 */

    U32 idxl0; /* the long match index for ip */
    U32 idxl1; /* the long match index for ip1 */

    const BYTE* matchl0; /* the long match for ip */
    const BYTE* matchs0; /* the short match for ip */
    const BYTE* matchl1; /* the long match for ip1 */
    const BYTE* matchs0_safe; /* matchs0 or safe address */

    const BYTE* ip = istart; /* the current position */
    const BYTE* ip1; /* the next position */
    /* Array of ~random data, should have low probability of matching data
     * we load from here instead of from tables, if matchl0/matchl1 are
     * invalid indices. Used to avoid unpredictable branches. */
    const BYTE dummy[] = {0x12,0x34,0x56,0x78,0x9a,0xbc,0xde,0xf0,0xe2,0xb4};

    DEBUGLOG(5, "ZSTD_compressBlock_doubleFast_noDict_generic");

    /* init */
    ip += ((ip - prefixLowest) == 0);
    {
        U32 const current = (U32)(ip - base);
        U32 const windowLow = ZSTD_getLowestPrefixIndex(ms, current, cParams->windowLog);
        U32 const maxRep = current - windowLow;
        if (offset_2 > maxRep) offsetSaved2 = offset_2, offset_2 = 0;
        if (offset_1 > maxRep) offsetSaved1 = offset_1, offset_1 = 0;
    }

    /* Outer Loop: one iteration per match found and stored */
    while (1) {
        step = 1;
        nextStep = ip + kStepIncr;
        ip1 = ip + step;

        if (ip1 > ilimit) {
            goto _cleanup;
        }

        hl0 = ZSTD_hashPtr(ip, hBitsL, 8);
        idxl0 = hashLong[hl0];
        matchl0 = base + idxl0;

        /* Inner Loop: one iteration per search / position */
        do {
            const size_t hs0 = ZSTD_hashPtr(ip, hBitsS, mls);
            const U32 idxs0 = hashSmall[hs0];
            curr = (U32)(ip-base);
            matchs0 = base + idxs0;

            hashLong[hl0] = hashSmall[hs0] = curr;   /* update hash tables */

            /* check noDict repcode */
            if ((offset_1 > 0) & (MEM_read32(ip+1-offset_1) == MEM_read32(ip+1))) {
                mLength = ZSTD_dfastCount(ip+1+4, ip+1+4-offset_1, iend, longMode) + 4;
                /* A cheap repeat is not necessarily the best match when a
                 * substantially longer regular match starts one byte earlier. */
                if (tuned && mLength <= 32 && idxl0 > prefixLowestIndex && MEM_read64(matchl0) == MEM_read64(ip)) {
                    size_t const len = ZSTD_dfastCount(ip+8, matchl0+8, iend, longMode) + 8;
                    offset = (U32)(ip-matchl0);
                    /* Multiplication keeps the comparison integral without
                     * a division for the binary literal price. */
                    if (((int)len-(int)mLength-1)*(swapTables ? 6 : 4) > (int)ZSTD_highbit32(offset+1)) {
                        mLength = len;
                        hl1 = ZSTD_hashPtr(ip1, hBitsL, 8);
                        while (((ip>anchor) & (matchl0>prefixLowest)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; }
                        goto _match_found;
                    }
                }
                ip++;
                ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
                goto _match_stored;
            }

            hl1 = ZSTD_hashPtr(ip1, hBitsL, 8);

            /* idxl0 > prefixLowestIndex is a (somewhat) unpredictable branch.
             * However expression below complies into conditional move. Since
             * match is unlikely and we only *branch* on idxl0 > prefixLowestIndex
             * if there is a match, all branches become predictable. */
            {   const BYTE*  const matchl0_safe = ZSTD_selectAddr(idxl0, prefixLowestIndex, matchl0, &dummy[0]);

                /* check prefix long match */
                if ((!tuned || longMode) ? (MEM_read64(matchl0_safe) == MEM_read64(ip) && matchl0_safe == matchl0) : (idxl0 >= prefixLowestIndex && MEM_read64(matchl0) == MEM_read64(ip))) {
                    mLength = ZSTD_dfastCount(ip+8, matchl0+8, iend, longMode) + 8;
                    offset = (U32)(ip-matchl0);
                    /* Include the current text match's backward extension
                     * in the lazy comparison. It increases matched length
                     * and reduces literals, just like the next candidate's
                     * look-behind credit below. */
                    if (tuned && !swapTables) {
                        while (((ip>anchor) & (matchl0>prefixLowest)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; }
                    }
                    /* One lazy step, charging for the extra literals and for
                     * any increase in the offset's encoding cost. A byte that
                     * can be caught up backwards both extends the match and
                     * removes a literal, so credit it twice in this estimate.
                     * Avoid recounting long candidates when the existing
                     * match is already long. Binary data benefits from a
                     * tighter limit than text. */
                    if (tuned && (mLength <= (swapTables ? 16 : 32))) {
                        idxl1 = hashLong[hl1];
                        matchl1 = base + idxl1;
                        if (idxl1 > prefixLowestIndex && MEM_read64(matchl1) == MEM_read64(ip1)) {
                            size_t const len = ZSTD_dfastCount(ip1+8, matchl1+8, iend, longMode) + 8;
                            if (((int)len-(int)mLength)*(swapTables ? 4 : 3) >
                                (int)ZSTD_highbit32((U32)(ip1-matchl1)+3) - (int)ZSTD_highbit32(offset+3)
                                + ((int)(ip1-ip)-2*(int)(ip1[-1] == matchl1[-1]))*(swapTables ? 4 : 3)) {
                                ip = ip1;
                                mLength = len;
                                matchl0 = matchl1;
                                offset = (U32)(ip - matchl0);
                                if (tuned && !swapTables) {
                                    while (((ip>anchor) & (matchl0>prefixLowest)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; }
                                }
                            }
                        }
                    }
                    if (!tuned || swapTables) {
                        while (((ip>anchor) & (matchl0>prefixLowest)) && (ip[-1] == matchl0[-1])) { ip--; matchl0--; mLength++; }
                    }
                    goto _match_found;
            }   }

            idxl1 = hashLong[hl1];
            matchl1 = base + idxl1;

            /* Same optimization as matchl0 above */
            matchs0_safe = ZSTD_selectAddr(idxs0, prefixLowestIndex, matchs0, &dummy[0]);

            /* check prefix short match */
            if((!tuned || longMode) ? (MEM_read32(matchs0_safe) == MEM_read32(ip) && matchs0_safe == matchs0) : (idxs0 >= prefixLowestIndex && MEM_read32(matchs0) == MEM_read32(ip))) {
                  goto _search_next_long;
            }

            if (ip1 >= nextStep) {
                PREFETCH_L1(ip1 + 64);
                PREFETCH_L1(ip1 + 128);
                step++;
                nextStep += kStepIncr;
            }
            ip = ip1;
            ip1 += step;

            hl0 = hl1;
            idxl0 = idxl1;
            matchl0 = matchl1;
    #if defined(__aarch64__)
            PREFETCH_L1(ip+256);
    #endif
        } while (ip1 <= ilimit);

_cleanup:
        /* If offset_1 started invalid (offsetSaved1 != 0) and became valid (offset_1 != 0),
         * rotate saved offsets. See comment in ZSTD_compressBlock_fast_noDict for more context. */
        offsetSaved2 = ((offsetSaved1 != 0) && (offset_1 != 0)) ? offsetSaved1 : offsetSaved2;

        /* save reps for next block */
        rep[0] = offset_1 ? offset_1 : offsetSaved1;
        rep[1] = offset_2 ? offset_2 : offsetSaved2;

        /* Return the last literals size */
        return (size_t)(iend - anchor);

_search_next_long:

        /* short match found: let's check for a longer one */
        mLength = ZSTD_dfastCount(ip+4, matchs0+4, iend, longMode) + 4;
        offset = (U32)(ip - matchs0);

        /* check long match at +1 position */
        if ((idxl1 > prefixLowestIndex) && (MEM_read64(matchl1) == MEM_read64(ip1))) {
            size_t const l1len = ZSTD_dfastCount(ip1+8, matchl1+8, iend, longMode) + 8;
            if (tuned ? (((int)l1len-(int)mLength)*(swapTables ? 4 : 3) >
                    (int)ZSTD_highbit32((U32)(ip1-matchl1)+3) - (int)ZSTD_highbit32(offset+3) + ((int)(ip1-ip)-2*(int)(ip1[-1] == matchl1[-1]))*(swapTables ? 4 : 3)) : l1len > mLength) {
                /* use the long match instead */
                ip = ip1;
                mLength = l1len;
                offset = (U32)(ip-matchl1);
                matchs0 = matchl1;
            }
        }

        while (((ip>anchor) & (matchs0>prefixLowest)) && (ip[-1] == matchs0[-1])) { ip--; matchs0--; mLength++; } /* complete backward */

        /* fall-through */

_match_found: /* requires ip, offset, mLength */
        offset_2 = offset_1;
        offset_1 = offset;

        if (step < 4) {
            /* It is unsafe to write this value back to the hashtable when ip1 is
             * greater than or equal to the new ip we will have after we're done
             * processing this match. Rather than perform that test directly
             * (ip1 >= ip + mLength), which costs speed in practice, we do a simpler
             * more predictable test. The minmatch even if we take a short match is
             * 4 bytes, so as long as step, the distance between ip and ip1
             * (initially) is less than 4, we know ip1 < new ip. */
            hashLong[hl1] = (U32)(ip1 - base);
        }

        ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

_match_stored:
        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr+2;
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
                /* Fill a few more positions at the beginning of the match.
                 * Text benefits from denser insertion. For binary inputs,
                 * reserve the extra positions for matches longer than eight
                 * bytes; short matches already have enough hash coverage.
                 * Never insert at or beyond the next search position.
                 * The enclosing ilimit check makes these hash reads safe. */
                if (tuned) {
                    hashLong[ZSTD_hashPtr(base+curr+3, hBitsL, 8)] = curr+3;
                    if (base+curr+4 < ip) {
                        size_t const hl = ZSTD_hashPtr(base+curr+4, hBitsL, 8);
                        size_t const hs = ZSTD_hashPtr(base+curr+4, hBitsS, mls);
                        hashLong[hl] = curr+4;
                        hashSmall[hs] = curr+4;
                    }
                    if ((!swapTables || (size_t)(ip-(base+curr))>8) && base+curr+5 < ip) {
                        hashLong[ZSTD_hashPtr(base+curr+5, hBitsL, 8)] = curr+5;
                    }
                    if ((!swapTables || (size_t)(ip-(base+curr))>8) && base+curr+6 < ip) {
                        size_t const hl = ZSTD_hashPtr(base+curr+6, hBitsL, 8);
                        size_t const hs = ZSTD_hashPtr(base+curr+6, hBitsS, mls);
                        hashLong[hl] = curr+6;
                        hashSmall[hs] = curr+6;
                    }
                }
                /* Give the end-of-match candidate priority when a
                 * complementary insertion maps to the same long-hash bucket. */
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
            }

            /* check immediate repcode */
            while ( (ip <= ilimit)
                 && ( (offset_2>0)
                    & (MEM_read32(ip) == MEM_read32(ip - offset_2)) )) {
                /* store sequence */
                size_t const rLength = ZSTD_dfastCount(ip+4, ip+4-offset_2, iend, longMode) + 4;
                U32 const tmpOff = offset_2; offset_2 = offset_1; offset_1 = tmpOff;  /* swap offset_2 <=> offset_1 */
                hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = (U32)(ip-base);
                hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = (U32)(ip-base);
                ZSTD_storeSeqOnly(seqStore, 0, REPCODE1_TO_OFFBASE, rLength);
                ip += rLength;
                anchor = ip;
                continue;   /* faster when present ... (?) */
            }
        }
    }
}


FORCE_INLINE_TEMPLATE
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_compressBlock_doubleFast_dictMatchState_generic(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    const U32 hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    const U32 hBitsS = cParams->chainLog;
    const BYTE* const base = ms->window.base;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const U32 endIndex = (U32)((size_t)(istart - base) + srcSize);
    /* presumes that, if there is a dictionary, it must be using Attach mode */
    const U32 prefixLowestIndex = ZSTD_getLowestPrefixIndex(ms, endIndex, cParams->windowLog);
    const BYTE* const prefixLowest = base + prefixLowestIndex;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - HASH_READ_SIZE;
    U32 offset_1=rep[0], offset_2=rep[1];

    const ZSTD_MatchState_t* const dms = ms->dictMatchState;
    const ZSTD_compressionParameters* const dictCParams = &dms->cParams;
    const U32* const dictHashLong  = dms->hashTable;
    const U32* const dictHashSmall = dms->chainTable;
    const U32 dictStartIndex       = dms->window.dictLimit;
    const BYTE* const dictBase     = dms->window.base;
    const BYTE* const dictStart    = dictBase + dictStartIndex;
    const BYTE* const dictEnd      = dms->window.nextSrc;
    const U32 dictIndexDelta       = prefixLowestIndex - (U32)(dictEnd - dictBase);
    const U32 dictHBitsL           = dictCParams->hashLog + ZSTD_SHORT_CACHE_TAG_BITS;
    const U32 dictHBitsS           = dictCParams->chainLog + ZSTD_SHORT_CACHE_TAG_BITS;
    const U32 dictAndPrefixLength  = (U32)((ip - prefixLowest) + (dictEnd - dictStart));

    DEBUGLOG(5, "ZSTD_compressBlock_doubleFast_dictMatchState_generic");

    /* if a dictionary is attached, it must be within window range */
    assert(ms->window.dictLimit + (1U << cParams->windowLog) >= endIndex);

    if (ms->prefetchCDictTables) {
        size_t const hashTableBytes = (((size_t)1) << dictCParams->hashLog) * sizeof(U32);
        size_t const chainTableBytes = (((size_t)1) << dictCParams->chainLog) * sizeof(U32);
        PREFETCH_AREA(dictHashLong, hashTableBytes);
        PREFETCH_AREA(dictHashSmall, chainTableBytes);
    }

    /* init */
    ip += (dictAndPrefixLength == 0);

    /* dictMatchState repCode checks don't currently handle repCode == 0
     * disabling. */
    assert(offset_1 <= dictAndPrefixLength);
    assert(offset_2 <= dictAndPrefixLength);

    /* Main Search Loop */
    while (ip < ilimit) {   /* < instead of <=, because repcode check at (ip+1) */
        size_t mLength;
        U32 offset;
        size_t const h2 = ZSTD_hashPtr(ip, hBitsL, 8);
        size_t const h = ZSTD_hashPtr(ip, hBitsS, mls);
        size_t const dictHashAndTagL = ZSTD_hashPtr(ip, dictHBitsL, 8);
        size_t const dictHashAndTagS = ZSTD_hashPtr(ip, dictHBitsS, mls);
        U32 const dictMatchIndexAndTagL = dictHashLong[dictHashAndTagL >> ZSTD_SHORT_CACHE_TAG_BITS];
        U32 const dictMatchIndexAndTagS = dictHashSmall[dictHashAndTagS >> ZSTD_SHORT_CACHE_TAG_BITS];
        int const dictTagsMatchL = ZSTD_comparePackedTags(dictMatchIndexAndTagL, dictHashAndTagL);
        int const dictTagsMatchS = ZSTD_comparePackedTags(dictMatchIndexAndTagS, dictHashAndTagS);
        U32 const curr = (U32)(ip-base);
        U32 const matchIndexL = hashLong[h2];
        U32 matchIndexS = hashSmall[h];
        const BYTE* matchLong = base + matchIndexL;
        const BYTE* match = base + matchIndexS;
        const U32 repIndex = curr + 1 - offset_1;
        const BYTE* repMatch = (repIndex < prefixLowestIndex) ?
                               dictBase + (repIndex - dictIndexDelta) :
                               base + repIndex;
        hashLong[h2] = hashSmall[h] = curr;   /* update hash tables */

        /* check repcode */
        if ((ZSTD_index_overlap_check(prefixLowestIndex, repIndex))
            && (MEM_read32(repMatch) == MEM_read32(ip+1)) ) {
            const BYTE* repMatchEnd = repIndex < prefixLowestIndex ? dictEnd : iend;
            mLength = ZSTD_count_2segments(ip+1+4, repMatch+4, iend, repMatchEnd, prefixLowest) + 4;
            ip++;
            ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
            goto _match_stored;
        }

        if ((matchIndexL >= prefixLowestIndex) && (MEM_read64(matchLong) == MEM_read64(ip))) {
            /* check prefix long match */
            mLength = ZSTD_count(ip+8, matchLong+8, iend) + 8;
            offset = (U32)(ip-matchLong);
            while (((ip>anchor) & (matchLong>prefixLowest)) && (ip[-1] == matchLong[-1])) { ip--; matchLong--; mLength++; } /* catch up */
            goto _match_found;
        } else if (dictTagsMatchL) {
            /* check dictMatchState long match */
            U32 const dictMatchIndexL = dictMatchIndexAndTagL >> ZSTD_SHORT_CACHE_TAG_BITS;
            const BYTE* dictMatchL = dictBase + dictMatchIndexL;
            assert(dictMatchL < dictEnd);

            if (dictMatchL > dictStart && MEM_read64(dictMatchL) == MEM_read64(ip)) {
                mLength = ZSTD_count_2segments(ip+8, dictMatchL+8, iend, dictEnd, prefixLowest) + 8;
                offset = (U32)(curr - dictMatchIndexL - dictIndexDelta);
                while (((ip>anchor) & (dictMatchL>dictStart)) && (ip[-1] == dictMatchL[-1])) { ip--; dictMatchL--; mLength++; } /* catch up */
                goto _match_found;
        }   }

        if (matchIndexS > prefixLowestIndex) {
            /* short match  candidate */
            if (MEM_read32(match) == MEM_read32(ip)) {
                goto _search_next_long;
            }
        } else if (dictTagsMatchS) {
            /* check dictMatchState short match */
            U32 const dictMatchIndexS = dictMatchIndexAndTagS >> ZSTD_SHORT_CACHE_TAG_BITS;
            match = dictBase + dictMatchIndexS;
            matchIndexS = dictMatchIndexS + dictIndexDelta;

            if (match > dictStart && MEM_read32(match) == MEM_read32(ip)) {
                goto _search_next_long;
        }   }

        ip += ((ip-anchor) >> kSearchStrength) + 1;
#if defined(__aarch64__)
        PREFETCH_L1(ip+256);
#endif
        continue;

_search_next_long:
        {   size_t const hl3 = ZSTD_hashPtr(ip+1, hBitsL, 8);
            size_t const dictHashAndTagL3 = ZSTD_hashPtr(ip+1, dictHBitsL, 8);
            U32 const matchIndexL3 = hashLong[hl3];
            U32 const dictMatchIndexAndTagL3 = dictHashLong[dictHashAndTagL3 >> ZSTD_SHORT_CACHE_TAG_BITS];
            int const dictTagsMatchL3 = ZSTD_comparePackedTags(dictMatchIndexAndTagL3, dictHashAndTagL3);
            const BYTE* matchL3 = base + matchIndexL3;
            hashLong[hl3] = curr + 1;

            /* check prefix long +1 match */
            if ((matchIndexL3 >= prefixLowestIndex) && (MEM_read64(matchL3) == MEM_read64(ip+1))) {
                mLength = ZSTD_count(ip+9, matchL3+8, iend) + 8;
                ip++;
                offset = (U32)(ip-matchL3);
                while (((ip>anchor) & (matchL3>prefixLowest)) && (ip[-1] == matchL3[-1])) { ip--; matchL3--; mLength++; } /* catch up */
                goto _match_found;
            } else if (dictTagsMatchL3) {
                /* check dict long +1 match */
                U32 const dictMatchIndexL3 = dictMatchIndexAndTagL3 >> ZSTD_SHORT_CACHE_TAG_BITS;
                const BYTE* dictMatchL3 = dictBase + dictMatchIndexL3;
                assert(dictMatchL3 < dictEnd);
                if (dictMatchL3 > dictStart && MEM_read64(dictMatchL3) == MEM_read64(ip+1)) {
                    mLength = ZSTD_count_2segments(ip+1+8, dictMatchL3+8, iend, dictEnd, prefixLowest) + 8;
                    ip++;
                    offset = (U32)(curr + 1 - dictMatchIndexL3 - dictIndexDelta);
                    while (((ip>anchor) & (dictMatchL3>dictStart)) && (ip[-1] == dictMatchL3[-1])) { ip--; dictMatchL3--; mLength++; } /* catch up */
                    goto _match_found;
        }   }   }

        /* if no long +1 match, explore the short match we found */
        if (matchIndexS < prefixLowestIndex) {
            mLength = ZSTD_count_2segments(ip+4, match+4, iend, dictEnd, prefixLowest) + 4;
            offset = (U32)(curr - matchIndexS);
            while (((ip>anchor) & (match>dictStart)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; } /* catch up */
        } else {
            mLength = ZSTD_count(ip+4, match+4, iend) + 4;
            offset = (U32)(ip - match);
            while (((ip>anchor) & (match>prefixLowest)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; } /* catch up */
        }

_match_found:
        offset_2 = offset_1;
        offset_1 = offset;

        ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

_match_stored:
        /* match found */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr+2;
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
            }

            /* check immediate repcode */
            while (ip <= ilimit) {
                U32 const current2 = (U32)(ip-base);
                U32 const repIndex2 = current2 - offset_2;
                const BYTE* repMatch2 = repIndex2 < prefixLowestIndex ?
                        dictBase + repIndex2 - dictIndexDelta :
                        base + repIndex2;
                if ( (ZSTD_index_overlap_check(prefixLowestIndex, repIndex2))
                   && (MEM_read32(repMatch2) == MEM_read32(ip)) ) {
                    const BYTE* const repEnd2 = repIndex2 < prefixLowestIndex ? dictEnd : iend;
                    size_t const repLength2 = ZSTD_count_2segments(ip+4, repMatch2+4, iend, repEnd2, prefixLowest) + 4;
                    U32 tmpOffset = offset_2; offset_2 = offset_1; offset_1 = tmpOffset;   /* swap offset_2 <=> offset_1 */
                    ZSTD_storeSeq(seqStore, 0, anchor, iend, REPCODE1_TO_OFFBASE, repLength2);
                    hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = current2;
                    hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
            }
        }
    }   /* while (ip < ilimit) */

    /* save reps for next block */
    rep[0] = offset_1;
    rep[1] = offset_2;

    /* Return the last literals size */
    return (size_t)(iend - anchor);
}

#define ZSTD_GEN_DFAST_FN(dictMode, mls)                                                                 \
    static size_t ZSTD_compressBlock_doubleFast_##dictMode##_##mls(                                      \
            ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],                          \
            void const* src, size_t srcSize)                                                             \
    {                                                                                                    \
        return ZSTD_compressBlock_doubleFast_##dictMode##_generic(ms, seqStore, rep, src, srcSize, mls); \
    }

#define ZSTD_GEN_DFAST_NODICT(mls) \
    static size_t ZSTD_compressBlock_doubleFast_noDict_##mls( \
        ZSTD_MatchState_t* ms, SeqStore_t* ss, U32 rep[ZSTD_REP_NUM], \
        void const* src, size_t size) { \
        return ZSTD_compressBlock_doubleFast_noDict_generic(ms,ss,rep,src,size,mls,ms->cParams.hashLog,ms->cParams.chainLog,0,0,0); \
    }
ZSTD_GEN_DFAST_NODICT(4)
ZSTD_GEN_DFAST_NODICT(5)
ZSTD_GEN_DFAST_NODICT(6)
ZSTD_GEN_DFAST_NODICT(7)


ZSTD_GEN_DFAST_FN(dictMatchState, 4)
ZSTD_GEN_DFAST_FN(dictMatchState, 5)
ZSTD_GEN_DFAST_FN(dictMatchState, 6)
ZSTD_GEN_DFAST_FN(dictMatchState, 7)


/* Constant table sizes remove variable shifts and reduce register pressure.
 * Both layouts use exactly the originally allocated 2^17 + 2^16 entries. */
FORCE_NOINLINE size_t ZSTD_compressBlock_doubleFast_special(
        ZSTD_MatchState_t* ms, SeqStore_t* ss, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t size) {
    return ZSTD_compressBlock_doubleFast_noDict_generic(ms,ss,rep,src,size,5,17,16,0,0,1);
}
FORCE_NOINLINE size_t ZSTD_compressBlock_doubleFast_specialLong(
        ZSTD_MatchState_t* ms, SeqStore_t* ss, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t size) {
    return ZSTD_compressBlock_doubleFast_noDict_generic(ms,ss,rep,src,size,5,17,16,0,1,1);
}

FORCE_NOINLINE size_t ZSTD_compressBlock_doubleFast_text(
        ZSTD_MatchState_t* ms, SeqStore_t* ss, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t size) {
    return ZSTD_compressBlock_doubleFast_noDict_generic(ms,ss,rep,src,size,6,17,16,0,0,1);
}
FORCE_NOINLINE size_t ZSTD_compressBlock_doubleFast_textLong(
        ZSTD_MatchState_t* ms, SeqStore_t* ss, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t size) {
    return ZSTD_compressBlock_doubleFast_noDict_generic(ms,ss,rep,src,size,6,17,16,0,1,1);
}
FORCE_NOINLINE size_t ZSTD_compressBlock_doubleFast_binary(
        ZSTD_MatchState_t* ms, SeqStore_t* ss, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t size) {
    return ZSTD_compressBlock_doubleFast_noDict_generic(ms,ss,rep,src,size,5,16,17,1,0,1);
}
FORCE_NOINLINE size_t ZSTD_compressBlock_doubleFast_binaryLong(
        ZSTD_MatchState_t* ms, SeqStore_t* ss, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t size) {
    return ZSTD_compressBlock_doubleFast_noDict_generic(ms,ss,rep,src,size,5,16,17,1,1,1);
}

size_t ZSTD_compressBlock_doubleFast(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    const U32 mls = ms->cParams.minMatch;
    if (mls==5 && ms->cParams.hashLog==17 && ms->cParams.chainLog==16) {
        /* Choose a short hash and table layout once per frame. Keep this
         * choice stable, including when a later block has different contents.
         * Dictionary tables retain their original interpretation. */
        if (ms->dfastMode == 0) {
            ms->dfastMode = 1;
            if (!ms->loadedDictEnd && !ms->dictMatchState && ms->window.dictLimit == ms->window.lowLimit && srcSize >= 4096) {
                const BYTE* const p = (const BYTE*)src;
                /* An odd stride avoids sampling just one column of records. */
                size_t const stride = (srcSize / 2048) | 1;
                size_t i, samples=0, text=0;
                for (i=0; i<srcSize; i+=stride) {
                    U32 const c=p[i];
                    text += (c>=32 && c<127) || c==9 || c==10 || c==13;
                    ++samples;
                }
                ms->dfastMode = text*4 >= samples*3 ? 2 : 3;
            }
        }
        {
            U32 const mode = ms->dfastMode & 3;
            U32 const longMode = ms->dfastMode & 4;
            SeqDef* const seqBegin = seqStore->sequences;
            BYTE* const litBegin = seqStore->lit;
            size_t result;
            if (mode == 2) result = longMode ?
                ZSTD_compressBlock_doubleFast_textLong(ms,seqStore,rep,src,srcSize) :
                ZSTD_compressBlock_doubleFast_text(ms,seqStore,rep,src,srcSize);
            else if (mode == 3) result = longMode ?
                ZSTD_compressBlock_doubleFast_binaryLong(ms,seqStore,rep,src,srcSize) :
                ZSTD_compressBlock_doubleFast_binary(ms,seqStore,rep,src,srcSize);
            else result = longMode ?
                ZSTD_compressBlock_doubleFast_specialLong(ms,seqStore,rep,src,srcSize) :
                ZSTD_compressBlock_doubleFast_special(ms,seqStore,rep,src,srcSize);
            {
                size_t const n = (size_t)(seqStore->sequences - seqBegin);
                size_t const matched = srcSize - result - (size_t)(seqStore->lit - litBegin);
                /* Long-match blocks leave more stale hash entries. Use the
                 * branchless validity checks and wide counter in that case.
                 * This selection affects speed only, not the parse. */
                ms->dfastMode = mode | ((n && matched >= n*24) ? 4 : 0);
            }
            return result;
        }
    }
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_doubleFast_noDict_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return ZSTD_compressBlock_doubleFast_noDict_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return ZSTD_compressBlock_doubleFast_noDict_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return ZSTD_compressBlock_doubleFast_noDict_7(ms, seqStore, rep, src, srcSize);
    }
}


size_t ZSTD_compressBlock_doubleFast_dictMatchState(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    const U32 mls = ms->cParams.minMatch;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return ZSTD_compressBlock_doubleFast_dictMatchState_7(ms, seqStore, rep, src, srcSize);
    }
}


static
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_compressBlock_doubleFast_extDict_generic(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize,
        U32 const mls /* template */)
{
    ZSTD_compressionParameters const* cParams = &ms->cParams;
    U32* const hashLong = ms->hashTable;
    U32  const hBitsL = cParams->hashLog;
    U32* const hashSmall = ms->chainTable;
    U32  const hBitsS = cParams->chainLog;
    const BYTE* const istart = (const BYTE*)src;
    const BYTE* ip = istart;
    const BYTE* anchor = istart;
    const BYTE* const iend = istart + srcSize;
    const BYTE* const ilimit = iend - 8;
    const BYTE* const base = ms->window.base;
    const U32   endIndex = (U32)((size_t)(istart - base) + srcSize);
    const U32   lowLimit = ZSTD_getLowestMatchIndex(ms, endIndex, cParams->windowLog);
    const U32   dictStartIndex = lowLimit;
    const U32   dictLimit = ms->window.dictLimit;
    const U32   prefixStartIndex = (dictLimit > lowLimit) ? dictLimit : lowLimit;
    const BYTE* const prefixStart = base + prefixStartIndex;
    const BYTE* const dictBase = ms->window.dictBase;
    const BYTE* const dictStart = dictBase + dictStartIndex;
    const BYTE* const dictEnd = dictBase + prefixStartIndex;
    U32 offset_1=rep[0], offset_2=rep[1];

    DEBUGLOG(5, "ZSTD_compressBlock_doubleFast_extDict_generic (srcSize=%zu)", srcSize);

    /* if extDict is invalidated due to maxDistance, switch to "regular" variant */
    if (prefixStartIndex == dictStartIndex)
        return ZSTD_compressBlock_doubleFast(ms, seqStore, rep, src, srcSize);

    /* Search Loop */
    while (ip < ilimit) {  /* < instead of <=, because (ip+1) */
        const size_t hSmall = ZSTD_hashPtr(ip, hBitsS, mls);
        const U32 matchIndex = hashSmall[hSmall];
        const BYTE* const matchBase = matchIndex < prefixStartIndex ? dictBase : base;
        const BYTE* match = matchBase + matchIndex;

        const size_t hLong = ZSTD_hashPtr(ip, hBitsL, 8);
        const U32 matchLongIndex = hashLong[hLong];
        const BYTE* const matchLongBase = matchLongIndex < prefixStartIndex ? dictBase : base;
        const BYTE* matchLong = matchLongBase + matchLongIndex;

        const U32 curr = (U32)(ip-base);
        const U32 repIndex = curr + 1 - offset_1;   /* offset_1 expected <= curr +1 */
        const BYTE* const repBase = repIndex < prefixStartIndex ? dictBase : base;
        const BYTE* const repMatch = repBase + repIndex;
        size_t mLength;
        hashSmall[hSmall] = hashLong[hLong] = curr;   /* update hash table */

        if (((ZSTD_index_overlap_check(prefixStartIndex, repIndex))
            & (offset_1 <= curr+1 - dictStartIndex)) /* note: we are searching at curr+1 */
          && (MEM_read32(repMatch) == MEM_read32(ip+1)) ) {
            const BYTE* repMatchEnd = repIndex < prefixStartIndex ? dictEnd : iend;
            mLength = ZSTD_count_2segments(ip+1+4, repMatch+4, iend, repMatchEnd, prefixStart) + 4;
            ip++;
            ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, REPCODE1_TO_OFFBASE, mLength);
        } else {
            if ((matchLongIndex > dictStartIndex) && (MEM_read64(matchLong) == MEM_read64(ip))) {
                const BYTE* const matchEnd = matchLongIndex < prefixStartIndex ? dictEnd : iend;
                const BYTE* const lowMatchPtr = matchLongIndex < prefixStartIndex ? dictStart : prefixStart;
                U32 offset;
                mLength = ZSTD_count_2segments(ip+8, matchLong+8, iend, matchEnd, prefixStart) + 8;
                offset = curr - matchLongIndex;
                while (((ip>anchor) & (matchLong>lowMatchPtr)) && (ip[-1] == matchLong[-1])) { ip--; matchLong--; mLength++; }   /* catch up */
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

            } else if ((matchIndex > dictStartIndex) && (MEM_read32(match) == MEM_read32(ip))) {
                size_t const h3 = ZSTD_hashPtr(ip+1, hBitsL, 8);
                U32 const matchIndex3 = hashLong[h3];
                const BYTE* const match3Base = matchIndex3 < prefixStartIndex ? dictBase : base;
                const BYTE* match3 = match3Base + matchIndex3;
                U32 offset;
                hashLong[h3] = curr + 1;
                if ( (matchIndex3 > dictStartIndex) && (MEM_read64(match3) == MEM_read64(ip+1)) ) {
                    const BYTE* const matchEnd = matchIndex3 < prefixStartIndex ? dictEnd : iend;
                    const BYTE* const lowMatchPtr = matchIndex3 < prefixStartIndex ? dictStart : prefixStart;
                    mLength = ZSTD_count_2segments(ip+9, match3+8, iend, matchEnd, prefixStart) + 8;
                    ip++;
                    offset = curr+1 - matchIndex3;
                    while (((ip>anchor) & (match3>lowMatchPtr)) && (ip[-1] == match3[-1])) { ip--; match3--; mLength++; } /* catch up */
                } else {
                    const BYTE* const matchEnd = matchIndex < prefixStartIndex ? dictEnd : iend;
                    const BYTE* const lowMatchPtr = matchIndex < prefixStartIndex ? dictStart : prefixStart;
                    mLength = ZSTD_count_2segments(ip+4, match+4, iend, matchEnd, prefixStart) + 4;
                    offset = curr - matchIndex;
                    while (((ip>anchor) & (match>lowMatchPtr)) && (ip[-1] == match[-1])) { ip--; match--; mLength++; }   /* catch up */
                }
                offset_2 = offset_1;
                offset_1 = offset;
                ZSTD_storeSeq(seqStore, (size_t)(ip-anchor), anchor, iend, OFFSET_TO_OFFBASE(offset), mLength);

            } else {
                ip += ((ip-anchor) >> kSearchStrength) + 1;
                continue;
        }   }

        /* move to next sequence start */
        ip += mLength;
        anchor = ip;

        if (ip <= ilimit) {
            /* Complementary insertion */
            /* done after iLimit test, as candidates could be > iend-8 */
            {   U32 const indexToInsert = curr+2;
                hashLong[ZSTD_hashPtr(base+indexToInsert, hBitsL, 8)] = indexToInsert;
                hashLong[ZSTD_hashPtr(ip-2, hBitsL, 8)] = (U32)(ip-2-base);
                hashSmall[ZSTD_hashPtr(base+indexToInsert, hBitsS, mls)] = indexToInsert;
                hashSmall[ZSTD_hashPtr(ip-1, hBitsS, mls)] = (U32)(ip-1-base);
            }

            /* check immediate repcode */
            while (ip <= ilimit) {
                U32 const current2 = (U32)(ip-base);
                U32 const repIndex2 = current2 - offset_2;
                const BYTE* repMatch2 = repIndex2 < prefixStartIndex ? dictBase + repIndex2 : base + repIndex2;
                if ( ((ZSTD_index_overlap_check(prefixStartIndex, repIndex2))
                    & (offset_2 <= current2 - dictStartIndex))
                  && (MEM_read32(repMatch2) == MEM_read32(ip)) ) {
                    const BYTE* const repEnd2 = repIndex2 < prefixStartIndex ? dictEnd : iend;
                    size_t const repLength2 = ZSTD_count_2segments(ip+4, repMatch2+4, iend, repEnd2, prefixStart) + 4;
                    U32 const tmpOffset = offset_2; offset_2 = offset_1; offset_1 = tmpOffset;   /* swap offset_2 <=> offset_1 */
                    ZSTD_storeSeq(seqStore, 0, anchor, iend, REPCODE1_TO_OFFBASE, repLength2);
                    hashSmall[ZSTD_hashPtr(ip, hBitsS, mls)] = current2;
                    hashLong[ZSTD_hashPtr(ip, hBitsL, 8)] = current2;
                    ip += repLength2;
                    anchor = ip;
                    continue;
                }
                break;
    }   }   }

    /* save reps for next block */
    rep[0] = offset_1;
    rep[1] = offset_2;

    /* Return the last literals size */
    return (size_t)(iend - anchor);
}

ZSTD_GEN_DFAST_FN(extDict, 4)
ZSTD_GEN_DFAST_FN(extDict, 5)
ZSTD_GEN_DFAST_FN(extDict, 6)
ZSTD_GEN_DFAST_FN(extDict, 7)

size_t ZSTD_compressBlock_doubleFast_extDict(
        ZSTD_MatchState_t* ms, SeqStore_t* seqStore, U32 rep[ZSTD_REP_NUM],
        void const* src, size_t srcSize)
{
    U32 const mls = ms->cParams.minMatch;
    switch(mls)
    {
    default: /* includes case 3 */
    case 4 :
        return ZSTD_compressBlock_doubleFast_extDict_4(ms, seqStore, rep, src, srcSize);
    case 5 :
        return ZSTD_compressBlock_doubleFast_extDict_5(ms, seqStore, rep, src, srcSize);
    case 6 :
        return ZSTD_compressBlock_doubleFast_extDict_6(ms, seqStore, rep, src, srcSize);
    case 7 :
        return ZSTD_compressBlock_doubleFast_extDict_7(ms, seqStore, rep, src, srcSize);
    }
}

#endif /* ZSTD_EXCLUDE_DFAST_BLOCK_COMPRESSOR */
