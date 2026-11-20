//! q2_0.rs - Q2_0 (ggml): two bits per weight, one fp16 scale per 64, and no zero-point stored anywhere: the four
//! codes mean -1, 0, +1 and +2 times the scale. A plain format with no tables and no sub-block scales, which is what
//! makes it cheap to add; the cost is that a 2-bit linear quant needs weights that tolerate it, so it turns up in
//! per-tensor mixes (the GSQ/RCO family) rather than as a whole-file choice.

use crate::f16_to_f32;
use crate::quant::q8_k::Groups32;

/// Q2_0's block: an fp16 scale and 64 two-bit weights, four to a byte, low bits first.
pub const Q2_0_BLOCK: usize = 18;

/// How many of the nv vectors are carried at a time: a block's scale and its unpacked weights are read once and
/// applied to all of them.
const VECTORS: usize = 16;

/// Thirty-two weights against the same thirty-two activations.
#[inline(always)]
fn dot_group(w: &[i8], a: &[i8]) -> i32 {
    super::dots::dot_w32(w, a)
}

/// One block's 64 weights, unpacked to integers in -1..2: the codes offset by -1 as they are read, so the dots below
/// need no separate offset term. Weights 0-31 of a block belong to the vector's first activation group of the pair,
/// weights 32-63 to the second. On x86-64 the whole run comes out of one shift and mask per four codes
/// (x86mm::unpack_q2_0); elsewhere it is read one code at a time, as the format is packed.
#[inline(always)]
fn unpack(blk: &[u8], qv: &mut [i8; 64]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"))]
    unsafe {
        crate::x86mm::unpack_q2_0(blk.as_ptr(), qv.as_mut_ptr())
    };
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c")))]
    for j in 0..64 {
        qv[j] = ((blk[2 + j / 4] >> (2 * (j % 4))) & 3) as i8 - 1;
    }
}

/// The two integer dots of one block (64 weights) against one vector's two activation groups. Where there is only one
/// vector to serve there is no block shared between vectors, so this is the whole of reading it: where there is a
/// vector unpack it is that unpack and the shared dots, and elsewhere the codes are taken straight from the block,
/// which needs no buffer for a block only one vector sees.
///
/// AArch64 takes the unpacked form. The fused reading below cannot be vectorised: the index picks both the source byte
/// (`j / 4`) and the field within it (`j % 4`), so there is no fixed shift to apply across a lane group, and a decoded
/// token pays it once per block. Measured on a Snapdragon X core, that reading runs Q2_0 at 3.9 GMAC/s at one vector
/// against 20.6 at sixteen, and it is the whole of the difference.
#[inline(always)]
fn ints_from_block(blk: &[u8], xa: &Groups32, h: usize) -> (i32, i32) {
    #[cfg(any(
        all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"),
        target_arch = "aarch64"
    ))]
    {
        let mut qv = [0i8; 64];
        unpack(blk, &mut qv);
        (dot_group(&qv[..32], &xa.qs[64 * h..]), dot_group(&qv[32..], &xa.qs[64 * h + 32..]))
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma", target_feature = "f16c"),
        target_arch = "aarch64"
    )))]
    {
        let (mut s0, mut s1) = (0i32, 0i32);
        for j in 0..32 {
            let (q0, q1) = (blk[2 + j / 4], blk[10 + j / 4]);
            s0 += (((q0 >> (2 * (j % 4))) & 3) as i8 - 1) as i32 * xa.qs[64 * h + j] as i32;
            s1 += (((q1 >> (2 * (j % 4))) & 3) as i8 - 1) as i32 * xa.qs[64 * h + 32 + j] as i32;
        }
        (s0, s1)
    }
}

/// Rows of Q2_0 against nv vectors, portably (ggml's ggml_vec_dot_q2_0_q8_0). The activations are the 32-wide groups
/// of ggml's q8_0 side - one scale per 32, which is why a block of 64 weights spans two of them - and every group is
/// multiplied by its own pair of scales, as in every other format here. Both paths add the same groups in the same
/// order, so a row gives the same bits whichever one runs.
pub fn q2_0_matmul_scalar(
    w: &[u8],
    row_bytes: usize,
    rows: usize,
    x: &[Groups32],
    nv: usize,
    y: &mut [f32],
    y_stride: usize,
) {
    // A row is nb units of 256 numbers: four blocks of 64 weights against eight groups of 32 activations.
    let nb = x.len() / nv.max(1);
    assert!(nv >= 1 && x.len() == nv * nb && row_bytes >= nb * 4 * Q2_0_BLOCK && w.len() >= rows * row_bytes);
    // One vector, which is what a decoded token is: the weights are read and dotted in the same pass, so nothing is
    // materialised for a block that only one vector uses.
    if nv == 1 {
        for r in 0..rows {
            let wr = &w[r * row_bytes..];
            let mut acc = 0f32;
            for b in 0..nb {
                let xa = &x[b];
                for h in 0..4 {
                    let blk = &wr[(b * 4 + h) * Q2_0_BLOCK..(b * 4 + h) * Q2_0_BLOCK + Q2_0_BLOCK];
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let (s0, s1) = ints_from_block(blk, xa, h);
                    acc = (d * xa.d[2 * h]).mul_add(s0 as f32, acc);
                    acc = (d * xa.d[2 * h + 1]).mul_add(s1 as f32, acc);
                }
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
                // The scale and the 64 weights of each block, decoded once for the whole chunk of vectors.
                for h in 0..4 {
                    let blk = &wr[(b * 4 + h) * Q2_0_BLOCK..(b * 4 + h) * Q2_0_BLOCK + Q2_0_BLOCK];
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let mut qv = [0i8; 64];
                    unpack(blk, &mut qv);
                    for j in 0..k {
                        let xa = &x[(v0 + j) * nb + b];
                        let s0 = dot_group(&qv[..32], &xa.qs[64 * h..]);
                        acc[j] = (d * xa.d[2 * h]).mul_add(s0 as f32, acc[j]);
                        let s1 = dot_group(&qv[32..], &xa.qs[64 * h + 32..]);
                        acc[j] = (d * xa.d[2 * h + 1]).mul_add(s1 as f32, acc[j]);
                    }
                }
            }
            for j in 0..k {
                y[(v0 + j) * y_stride + r] = acc[j];
            }
        }
    }
}
