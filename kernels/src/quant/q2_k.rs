//! q2_k.rs - Q2_K (ggml): 256 weights a block, two bits each, with a 4-bit scale and a 4-bit minimum shared by
//! each group of 16 and two fp16 block scales. The two-bit values are unsigned (0..3), so the block's minimum is
//! taken off once at the end against the activations' own group sums - which is why the dot has a second term.

use crate::{BlockQ8K, f16_to_f32};

/// Q2_K's block: 16 scale bytes, 64 bytes of two-bit values, and the fp16 d and dmin.
pub const Q2_K_BLOCK: usize = 84;

/// How many of the nv vectors are carried at a time. The block's values come out of their packed form once and are
/// then used for all of them, which is the point of the loop order below.
const VECTORS: usize = 16;

/// The sixteen values of one group against the same sixteen activations.
#[inline(always)]
fn dot_group(w: &[i8], a: &[i8]) -> i32 {
    super::dots::dot_w16u(w, a)
}

/// A block's 256 values, in the order the activations are in: group g's values at byte `g % 2 * 16` of its 32-byte
/// half, shifted `(g % 8) / 2 * 2` bits. On x86-64 that reading is four shifts and masks per sixteen source bytes,
/// each run going to its group as is (x86mm::unpack_q2_k); elsewhere it is one code at a time, as the format is packed.
#[inline(always)]
fn unpack(blk: &[u8], qv: &mut [i8; 256]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    unsafe {
        crate::x86mm::unpack_q2_k(blk.as_ptr(), qv.as_mut_ptr())
    };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c")))]
    for g in 0..16 {
        let (chunk, half, shift) = (32 * (g / 8), 16 * (g % 2), 2 * ((g % 8) / 2));
        let (src, dst) = (&blk[16 + chunk + half..], &mut qv[16 * g..]);
        for l in 0..16 {
            dst[l] = ((src[l] >> shift) & 3) as i8;
        }
    }
}

/// One block against one vector: the scale-weighted sum of the value dots, and the sums of the activations over the
/// groups weighted by the group minima. With a single vector nothing is shared, so this is the whole of reading the
/// block; where there is a vector unpack it uses that and the shared dots, elsewhere it takes the codes straight from
/// the block, which needs no buffer for a block one vector sees once.
///
/// AArch64 takes the unpacked form, as in q2_0.rs. The reading below happens to be vectorisable - the shift is fixed
/// for a group, so a lane group can take it - and it still runs at 9.4 GMAC/s against 16.2 at sixteen vectors: it
/// re-reads and re-shifts the block for every vector rather than once, and it keeps the values out of the shape the
/// shared dots want.
#[inline(always)]
fn ints_from_block(blk: &[u8], xa: &BlockQ8K) -> (i32, i32) {
    let sc = &blk[0..16];
    let (mut sumi, mut summs) = (0i32, 0i32);
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"),
        target_arch = "aarch64"
    ))]
    {
        let mut qv = [0i8; 256];
        unpack(blk, &mut qv);
        for g in 0..16 {
            sumi += (sc[g] & 0xF) as i32 * dot_group(&qv[16 * g..], &xa.qs[16 * g..]);
            summs += xa.bsums[g] as i32 * (sc[g] >> 4) as i32;
        }
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"),
        target_arch = "aarch64"
    )))]
    for g in 0..16 {
        let (chunk, half, shift) = (32 * (g / 8), 16 * (g % 2), 2 * ((g % 8) / 2));
        let mut isuml = 0i32;
        for l in 0..16 {
            isuml += xa.qs[16 * g + l] as i32 * (((blk[16 + chunk + half + l] >> shift) & 3) as i32);
        }
        sumi += (sc[g] & 0xF) as i32 * isuml;
        summs += xa.bsums[g] as i32 * (sc[g] >> 4) as i32;
    }
    (sumi, summs)
}

/// Rows of Q2_K against nv vectors, portably (ggml's ggml_vec_dot_q2_K_q8_K): per group of 16, the group's scale
/// times the integer dot of its values with the activations, summed; minus the block minimum times the activations'
/// sums over the same groups. y[v * y_stride + r] for v < nv and r < rows.
///
/// Group g's sixteen two-bit values sit in the g / 8th 32-byte half of qs, at byte g % 2 * 16 of it, shifted
/// (g % 8) / 2 * 2 bits - four shifts a half, which is how ggml packs a block without wasting bits.
///
/// The vectors are taken in chunks so that unpacking a block is paid once per chunk rather than once per vector:
/// over a prompt that is the difference between reading a block a thousand times and reading it sixty.
pub fn q2k_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[BlockQ8K],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * Q2_K_BLOCK && w.len() >= rows * row_bytes);
    // One vector, which is what a decoded token is: reading the block and dotting it in one pass needs nothing kept
    // aside, and the chunk below would pay for materialising a block a single vector uses once.
    if nv == 1 {
        for r in 0..rows {
            let wr = &w[r * row_bytes..];
            let mut acc = 0f32;
            for b in 0..nb {
                let (blk, xa) = (&wr[b * Q2_K_BLOCK..(b + 1) * Q2_K_BLOCK], &x[b]);
                let (sumi, summs) = ints_from_block(blk, xa);
                let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
                let dm = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
                acc = (xa.d * d).mul_add(sumi as f32, acc);
                acc -= (xa.d * dm) * summs as f32;
            }
            y[r] = acc;
        }
        return;
    }
    for r in 0..rows {
        let wr = &w[r * row_bytes..];
        for v0 in (0..nv).step_by(VECTORS) {
            let k = VECTORS.min(nv - v0);
            let mut acc = [0f32; VECTORS];
            for b in 0..nb {
                let blk = &wr[b * Q2_K_BLOCK..(b + 1) * Q2_K_BLOCK];
                let sc = &blk[0..16];
                // The block's values in the order the activations are in, once for every vector in this chunk.
                let mut qv = [0i8; 256];
                unpack(blk, &mut qv);
                let d = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
                let dm = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
                for j in 0..k {
                    let xa = &x[(v0 + j) * nb + b];
                    let (mut sumi, mut summs) = (0i32, 0i32);
                    for g in 0..16 {
                        sumi += (sc[g] & 0xF) as i32 * dot_group(&qv[16 * g..], &xa.qs[16 * g..]);
                        summs += xa.bsums[g] as i32 * (sc[g] >> 4) as i32;
                    }
                    acc[j] = (xa.d * d).mul_add(sumi as f32, acc[j]);
                    acc[j] -= (xa.d * dm) * summs as f32;
                }
            }
            for j in 0..k {
                y[(v0 + j) * y_stride + r] = acc[j];
            }
        }
    }
}
