# BlurSort, formally: bit-reversal, polyphase sub-images, and why frame order matters

*Working note, 2026-10-25. Source read: `/Users/nico/Sites/blursort/blursort.js` (`blursort_base`, `extract_frames`, `upsample_order`) and `extract_64_frames.js`. Verified numerically in `bs_check.py` (scratch).*

## 1. What `blursort_base` computes

```js
function blursort_base(a) {            // a: flat array
  if (a.length <= 1) return a
  return blursort_base(evens(a)).concat(blursort_base(odds(a)))
}
```

Take the even-indexed elements, then the odd-indexed ones, and recurse on each half. For an array of length N = 2^k this is the **bit-reversal permutation**: output position p holds input index rev_k(p), where rev_k reverses the k bits of p.

Why: the first split sends every element whose index has LSB 0 to the top half and LSB 1 to the bottom half — i.e. the *lowest* bit of the input index becomes the *highest* bit of the output position. The recursion repeats this one bit at a time, so every bit lands at its mirror position.

Check (N = 8): `blursort_base([0..7]) = [0, 4, 2, 6, 1, 5, 3, 7]` = `[rev3(0), rev3(1), …, rev3(7)]`. Identical.

This recursion is literally the Cooley–Tukey radix-2 decimation-in-time FFT recursion with the butterflies removed: the FFT splits even/odd samples the same way, which is why FFT inputs are consumed in bit-reversed order.

## 2. What it does to a square 2^k × 2^k image

The image is flattened row-major, so pixel (x, y) has index i = y·W + x. With W = H = 2^k the index is 2k bits: the high k bits are y, the low k bits are x. Reversing all 2k bits gives

    rev_2k( y ‖ x ) = rev_k(x) ‖ rev_k(y)

so the pixel that was at (x, y) ends up at **row rev_k(x), column rev_k(y)**. BlurSort of a square power-of-two image is therefore *transpose + bit-reverse both coordinates*. (Verified for 8×8 in `bs_check.py`.)

The useful consequence is the quadrant structure. The output row's top bit is the input x's **lowest** bit, and the output column's top bit is y's lowest bit, so:

| output region | contains |
|---|---|
| top-left | pixels with x even, y even |
| top-right | x even, y odd |
| bottom-left | x odd, y even |
| bottom-right | x odd, y odd |

Each quadrant is one of the four 2× **polyphase sub-images** — the image subsampled by 2 in each axis at one of the four phase offsets — and each quadrant is itself BlurSorted (the recursion continues inside it). Applying the split s times gives 4^s blocks; block (R, C) in the 2^s × 2^s grid is the image subsampled by 2^s at phase (dx, dy) = (rev_s(R), rev_s(C)), with the block's own pixels still in bit-reversed order (hence `unblursort_arreglo` on each sub-image in `extract_frames`). Because the blocks sit in bit-reversed phase order rather than raster phase order, the hand-built `upsample_order` array in `extract_frames` exists to put the 64 frames back into a sensible sequence.

So the "frames" are not a blur and not a sort: they are the 4^s phase-shifted low-resolution copies of the image. Two frames differ by a known sub-pixel shift of (Δdx, Δdy)/2^s in low-res units. That is the exact definition of a multi-frame super-resolution input set, and of the sub-bands of the *lazy wavelet* (below).

**Why power-of-two matters (your 512/1024 recommendation).** The even/odd split works on any N, but only when W and H are powers of two does "flat index bit-reversal" factor into "per-coordinate bit-reversal". For other sizes the halves have unequal length and the split mixes rows, so the blocks stop being clean polyphase sub-images. 512×512 (s ≤ 9) and 1024×1024 are the right test sizes; Kodak (768×512) needs a 512×512 crop.

**Information-theoretically** BlurSort is a fixed, data-independent permutation: zero bits to describe, perfectly invertible, no arithmetic. Any change in compressed size comes entirely from the *order* in which a coder sees the pixels, i.e. from what context is available when each pixel is predicted. This is what makes it a clean instrument for the "order matters" question.

## 3. Why frame order matters — two different "good" orders

Your mixed results from the sub-frame video are, I think, explained by the fact that there are two incompatible notions of a good order:

**(a) Smooth order — for a video codec.** A video codec predicts frame t from frame t−1 (motion compensation). Consecutive frames should then differ by the *smallest* possible shift: one full-resolution pixel, i.e. adjacent phases. The right path is a snake/Gray-code/Hilbert walk through the 2^s × 2^s phase grid. The dyadic order BlurSort produces (phase 0, then 2^{s−1}, then 2^{s−2}, …) does the opposite: the early frames are as far apart in phase as possible. Prediction: ffmpeg lossless (`ffv1` or `libx264 -qp 0`) on the 64 frames in snake order beats the same frames in dyadic order. Cheap to test (minutes, CPU).

**(b) Dyadic order — for progressive / coarse-to-fine conditioning.** For an autoregressive model that conditions on *everything decoded so far*, the dyadic order is ideal: each new phase sits halfway between already-decoded phases, so every new pixel has decoded neighbours on *both* sides (an interpolation problem, much easier than raster extrapolation). This is the order used by every multiscale autoregressive image model (Reed 2017, Menick & Kalchbrenner 2019) and by progressive codecs (Adam7, FLIF, JPEG XL). Our raster prior currently has a one-sided (causal L-shaped) context; the two-scale experiment gives it a two-sided one for 3/4 of the pixels.

Our earlier 1D test (`test_blursort_ordering.py`: BlurSort-ordered 28×28 patches, 4.34 bpp vs raster 3.1 — raster wins by 40%) was a *third* thing: it fed the bit-reversed 1D sequence into a model that was still given raster 2D positions, so the model saw spatially scrambled input with no coarse-to-fine conditioning. That was the wrong test; the two-scale design below is the right one.

## 4. Prior art (what to cite; "to verify" = I am confident of the work, not of every detail)

**Bit-reversal / FFT.** Cooley & Tukey (1965), *An algorithm for the machine calculation of complex Fourier series*, Math. Comp. 19 — the even/odd recursion; the bit-reversal step is standard in radix-2 DIT (Gold & Rader 1969; Oppenheim & Schafer).

**Polyphase decomposition.** Bellanger, Bonnerot & Coudreuse (1976), *Digital filtering by polyphase network*, IEEE Trans. ASSP; Crochiere & Rabiner (1983), *Multirate Digital Signal Processing*; Vaidyanathan (1993), *Multirate Systems and Filter Banks*. The 2D 2×2 polyphase split of an image is a textbook object; BlurSort is its recursive form.

**Lazy wavelet and lifting.** Sweldens (1996), *The lifting scheme: a custom-design construction of biorthogonal wavelets*, Appl. Comput. Harmon. Anal.; Daubechies & Sweldens (1998), *Factoring wavelet transforms into lifting steps*, J. Fourier Anal. Appl. The "lazy wavelet" is exactly the even/odd split with no filtering; adding one predict step and one update step gives the Haar transform, and the 5/3 integer lifting wavelet is what JPEG 2000 uses for lossless (Taubman & Marcellin 2002). BlurSort recursed is the lazy wavelet *packet* decomposition (split every band, not just the low band). This is the most precise formal name for it.

**Interlacing / progressive transmission (the same decomposition used for compression).** PNG Adam7 interlacing (Costello; PNG spec 1996, RFC 2083 1997): seven passes of polyphase sub-images over an 8×8 phase grid — identical sub-images to BlurSort s = 3, in a hand-chosen order. GIF interlace (1987, rows only). JPEG progressive and hierarchical modes (ITU-T T.81, 1992). Burt & Adelson (1983), *The Laplacian pyramid as a compact image code* — coarse-to-fine residual coding. FLIF (Sneyers & Wuille, ICIP 2016) — interlaced lossless coding with context modelling (MANIAC trees) *conditioned on the already-decoded coarser passes*; this is the closest classical codec to the two-scale experiment. JPEG XL modular mode's **Squeeze** transform (Alakuijala et al. 2019; ISO/IEC 18181) — an integer Haar-like multiscale on exactly this split; JXL is the strongest lossless codec in our Kodak benchmark (8.688 bpp), so this is also the bar.

**Space-to-depth in CNNs.** Shi et al. (2016), *Real-time single image and video super-resolution using an efficient sub-pixel convolutional neural network* (ESPCN, CVPR) — PixelShuffle / pixel-unshuffle is the 2×2 polyphase split as a tensor op; the BlurSort frame extraction at scale s is `pixel_unshuffle(img, 2^s)` followed by a bit-reversal of the channel index.

**Autoregressive image models and pixel orderings (most directly relevant).**
- van den Oord et al. (2016), *Pixel Recurrent Neural Networks*, ICML — raster PixelRNN/PixelCNN, plus the *Multi-Scale PixelRNN* variant that conditions on a subsampled image.
- Reed et al. (2017), *Parallel Multiscale Autoregressive Density Estimation*, ICML — generates by repeated 2× upsampling; each new group of pixels is one polyphase phase conditioned on the previous ones; O(log N) sequential steps. This *is* BlurSort's dyadic order as a generative model.
- Menick & Kalchbrenner (2019), *Generating High Fidelity Images with Subscale Pixel Networks and Multidimensional Upscaling*, ICLR — "subscale ordering": slice the image into S×S spatially interleaved sub-images and generate them in sequence, conditioning on the earlier ones. This is precisely "BlurSort frames as a sequence", and it is the paper to cite for the idea.
- Jain, Abbeel & Pathak (2020), *Locally Masked Convolution for Autoregressive Models*, UAI — trains one model under several pixel orderings (raster, S-curve, Hilbert) and shows the ordering changes likelihood; the formal statement that "order matters" for compression.
- Salimans et al. (2017), PixelCNN++, ICLR — discretised logistic outputs, the standard likelihood model for lossless coding at this scale.

**Learned lossless compression with a coarse-to-fine structure.**
- Mentzer et al. (2019), *Practical Full Resolution Learned Lossless Image Compression* (L3C), CVPR — hierarchical latent, parallel decoding.
- Cao, Wu & Krähenbühl (2020), *Lossless Image Compression through Super-Resolution* (SReC), arXiv:2004.02872 — verified from the abstract (2026-10-25): "store a low resolution version of an image as raw pixels, followed by several iterations of lossless super-resolution", each iteration predicting the probability of the higher-resolution image conditioned on the lower one and entropy-coding it. That is the two-scale design below, iterated. (Number of levels and the exact downsampler — average pool vs subsample — still to confirm from the paper body.)
- Hoogeboom et al. (2019), *Integer Discrete Flows* (IDF), NeurIPS; Townsend et al. (2020), *HiLLoC*, ICLR; Kingma, Abbeel & Ho (2019), *Bit-Swap*, ICML — hierarchical latent-variable lossless coders, the other family.
- Zhang, Zhang, Kang & Li (2021), *iVPF: Numerical Invertible Volume Preserving Flow for Efficient Lossless Compression*, CVPR, arXiv:2103.16211 (verified 2026-10-25) — volume-preserving flows for lossless coding.

**Image-set compression / "set redundancy" (prior art for grouping and ordering frames to maximise compression — Nico's 2026-10-25 idea).** The question "given a set of similar images, in what order / tree should they be predicted from one another so the whole set compresses best" is a 1990s line of work from medical image databases (verified via Crossref 2026-10-25):
- Karadimitriou & Tyler (1997), *Min-max compression methods for medical image databases*, ACM SIGMOD Record (doi:10.1145/248603.248613); Karadimitriou & Fenstermacher (1997), *Image compression in medical image databases using set redundancy*, DCC'97; Karadimitriou & Tyler (1998), *The Centroid method for compressing sets of similar images*, Pattern Recognition Letters (doi:10.1016/S0167-8655(98)00033-6) — coined "set redundancy": compress the set against a shared centroid / min-max image, then code residuals.
- Nielsen & Li (2006), *MST for Lossy Compression of Image Sets*, DCC'06 (doi:10.1109/dcc.2006.48) — build a complete graph over the images with inter-image prediction cost as edge weight, take the **minimum spanning tree**, code each image from its tree parent. This is exactly the "search over groupings of frames" idea: MST is the optimal *tree* for pairwise prediction, computable in O(K² log K) for K frames; the greedy DFS/BFS over frames is a heuristic for the same objective. (Lossless MST variants exist in the same community; the earliest I could confirm by title is this lossy one.)
- Gergel, Cheng & Li (2006), *A Unified Framework for Lossless Image Set Compression*, DCC'06 (doi:10.1109/dcc.2006.4) — the lossless counterpart of the MST work, same group (U. Lethbridge; Cheng also has *Hierarchical Minimum Spanning Trees for Lossy Image Set Compression*, 2008). Found via Google Scholar through the real-Chrome route, 2026-10-25.
- **Sakurai & Inoue (2021), *Lossless Image Set Compression Using Animated FLIF*, ACM (doi:10.1145/3468081.3471130)** — codes a set of similar images as the frames of an *animated* FLIF file, i.e. exactly "run a still-image codec's multi-frame mode over the set". This is the closest prior art to the tile/interleave-then-JXL test; our variant differs only in the merge geometry (one big still image) and the codec (JXL/WebP). Abstract not yet read.
- Shi, Sun & Wu (2013), *Multi-model prediction for image set compression*, VCIP; Zhang, Zhang, Lin & Ma (2015), *An inter-image redundancy measure for image set compression*, ISCAS (doi:10.1109/iscas.2015.7168873); Wu et al. (2019), *3D mesh based inter-image prediction for image set compression*, ICME — the modern (photo-album, Internet-image) continuation: a cheap redundancy measure chooses the prediction structure, then geometric/photometric alignment before coding.
- Where a learned grouping would be new: these methods choose the prediction *tree* with a fixed measure; training a model to predict compressibility-of-the-group (or to emit the order directly) and comparing against MST on the same codec is, as far as I can find, not done for lossless sets of video frames. Our measurement says when it would pay: interleaving wins only where frames are more alike than neighbouring pixels (polyphase sub-images, near-static shots), so the search is really "find the near-static runs and the self-similar regions".

**Multi-frame super-resolution (the "pixel movie" reading).** Tsai & Huang (1984), *Multiframe image restoration and registration*; Irani & Peleg (1991), *Improving resolution by image registration*, CVGIP; Farsiu et al. (2004), *Fast and robust multiframe super resolution*, IEEE TIP. The classical MFSR problem is to reconstruct a high-resolution image from several low-resolution frames with *unknown* sub-pixel shifts. BlurSort frames are that input with the shifts *known exactly* and no blur — the degenerate, perfectly-posed case. Your "upsample the planet from video" idea is MFSR with real frames; the BlurSort frames are its ideal synthetic version.

**Space-filling curves (what BlurSort is *not*).** Morton (1966) Z-order; Hilbert (1891); Dafner, Cohen-Or & Matias (2000), *Context-based space filling curves*, Eurographics — curve orderings that *preserve* locality. Bit-reversal of the flattened index is the opposite: a dyadic order in which *sequence* neighbours are spatially far apart (which is why the 1D test with a local-context coder lost). To be precise about what is and is not lost (Nico, 2026-10-25): nothing — the permutation is deterministic and invertible, every pixel's destination is known, and a coder that models each sub-image given the earlier ones (the subscale / two-scale design) keeps full raster locality *inside* each sub-image and gains scale locality *across* them. "Locality-destroying" applies only to reading the bit-reversed stream with a 1D sliding window. Z-order interleaves the bits of x and y; BlurSort reverses them. Worth stating explicitly in a post because the two get confused.

**Where there may be novelty.** The pieces are all known: polyphase split, subscale ordering, conditioning a learned prior on coarser phases, lossless coding with a small model. What I have not seen is (i) the specific observation that the same decomposition has two opposite optimal orders depending on whether the coder conditions on one previous frame (video codec) or on all decoded content (AR model) — testable in an afternoon and a good blog-post figure; (ii) doing subscale coding with a *shared, ~200K-parameter* prior rather than a per-image or multi-million-parameter model; (iii) the "frames as a video" framing itself, which is a genuinely good way to *explain* progressive coding to a general reader. No claim stronger than that until the experiment runs.

## 5. The two-scale experiment (design; sizes to be ETA'd before launch)

Images: 512×512 crops (Kodak centre crops → 24 images; Pluto crop for figures; CLIC crops for training). Scale s = 1 first (4 phases), s = 2 if it pays.

1. **Baseline:** current raster prior (28×28 windows, L-shaped causal halo), Kodak-512 bpp. Already measured on full Kodak (9.33 bpp, 215K prior); re-measure on the crops.
2. **Two-scale, s = 1:** code phase (0,0) — the 256×256 half-res image — with the raster prior as-is (it is just a smaller image). Cost B0 ≈ 1/4 of the pixels at roughly the raster bpp (slightly higher: half-res images are less smooth).
3. Code phases (0,1), (1,0), (1,1) in that (dyadic) order with a **conditioned prior**: same PriorLM, but each target pixel's context gains the already-decoded neighbours from earlier phases — at phase (1,1) that is all four diagonal neighbours; at (0,1)/(1,0) the two neighbours along one axis. Implementation: extend the halo input from "L-shaped causal strip" to "L-shaped strip + decoded future neighbours", fed as extra tokens with their 2D RoPE positions (the model already handles arbitrary position sets). Train one model per phase class, or one model with a phase embedding (3 extra tokens) — the latter is the MVE.
4. **Metric:** total bpp = (B0 + B1 + B2 + B3)/N vs raster bpp, same 24 crops, same int8 prior budget; progressive-transmission curve (quality at 1/4, 1/2, 3/4 of the bits) as the bonus figure.
5. **Order test (cheap, CPU):** 64 frames of the 512×512 Pluto crop (s = 3) → `ffmpeg -c:v ffv1` and `libx264 -qp 0` under three orders: dyadic (BlurSort native), raster-phase, snake/Hilbert over the 8×8 phase grid. File sizes answer "does frame order matter for a video codec" directly and reproduce your sub-frame video experiment with a control.

Expected sign, from the literature: SReC/FLIF/JXL all gain from coarse-to-fine conditioning; the question for us is whether a 215K-parameter prior can exploit a two-sided context as well as it exploits the one-sided one. If it does, the Kodak number moves toward JXL with no decoder-size cost; if it does not, that is a capacity result worth reporting.


## 6. What Nico already tried (digest of `LanguageModel/`, `LanguageModel/blursort-predictor/`, `ImageProcessing/`; survey 2026-10-25)

- **Sub-frames as video (2023-2025).** Pipeline: BlurSort the image, cut the BlurSorted canvas into a grid of crops (2×2 → 4 frames, 8×8 → 64 frames), UnBlurSort each crop so it is a recognisable downsample, encode as MP4, reverse. Writing the raw BlurSorted tiles into the MP4 gave checkerboards (7.7 dB) — expected, since within a block the pixels are still bit-reversed. **Frame order was always row-major over the crop grid** (= dyadic phase order, since block (R,C) holds phase (rev R, rev C)); a 4-frame test tried sequential / random / "most-similar" orders: 68.8 / 70.3 / 74.5 KB at CRF23 — a ≤6% effect, with the similarity order *worst*. The planned "order by difference" step for 64 frames was never implemented, so the snake/Hilbert order has not been tested.
- **Measured compression (Pluto 1024², blursort-predictor, Aug-Oct 2025).** Lossless x264: 794-801 KB for 4/16/64 frames, 2.25× larger than the 354 KB JPEG — and *not actually lossless* (53.3 dB, 70% of pixels exact: yuv420 conversion). Lossy CRF18: 180-185 KB at ~39.3 dB regardless of frame count. The apparent "fps makes files smaller" gains (30→120 fps: 54 → 31 KB at the same CRF) are x264's rate control depending on frame rate, not a BlurSort property; against JPEG at matched PSNR the result was roughly a tie (29.5 KB JPEG ≈ 30.6 KB BlurSort-MP4 at ~29.4 dB). A claimed "64-frame recovery breakthrough" was retracted in the notes. All on n = 1 image, mostly black background.
- **Planet upsample from video (2024).** Fusing 64 real video frames (380², batches of 4 via BlurSort/concat/UnBlurSort) to 3040²: "can zoom more, less blocky, but not much more detail". A 400² image padded to 512² → 4 sub-frames → 48 KB video ≈ the same image as a 49 KB JPEG: **no gain**. This is multi-frame super-resolution without registration — Irani-Peleg 1991 is cited in `Ideas.md` but registration was never implemented, and for real video frames the sub-pixel shifts are unknown, so plain interleaving cannot recover detail.
- **Repetition analysis (DIV2K 512², Oct 2025).** 2×2 RGB patches after BlurSort: 0.78% repeats (65,022 unique of 65,536) — a block-matching/dictionary coder has nothing to work with; the redundancy is statistical, not literal. Supports the learned-prior route over LZ-style coding.
- **`ImageProcessing/` (Apr 2026)** is a different thing: a reproduction of dsp.stackexchange Q60106 (a "broken quicksort" permutation applied to pixels under several traversal orders). Nothing measured; every re-sort made the JPEG 9-14× larger — the generic result that scrambling locality kills transform coding.
- **Never tested:** any prediction/AR model on the frames; multi-image sets; two-level (coarse-to-fine) conditioning; exact RGB lossless round trips.

**Consequences for the design above.** (1) Verify exact invertibility in RGB (`libx264rgb -qp 0`, `ffv1 rgb24`), never yuv420 — done in `blursort_frame_order.py`. (2) Compare against PNG/WebP/JXL on the *original*, not against JPEG at one quality point. (3) Use ≥ 4 images (Kodak 512² crops + Pluto). (4) The frame-order test now includes the snake/Gray orders that were never run, plus an intra-only codec (ffv1) as the null control: if ffv1 sizes do not move with order and x264 sizes do, the effect is inter-frame prediction, cleanly.
