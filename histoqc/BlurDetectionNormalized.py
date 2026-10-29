"""
HistoQC blur step with a content-normalised, per-tile blur index (drop-in replacement for
BlurDetectionModule.identifyBlurryRegions).

Why: HistoQC's stock step thresholds the Gaussian-smoothed |Laplacian| of the 2.5x grey thumbnail at an absolute
value (radius 100, threshold 0.15 in config_v2.1). That value depends on scanner, stain intensity and tissue
texture, so pale / low-texture tissue (mucin, fat, stroma, necrosis) is called "blurry" and most of a slide can be
removed although it is in focus.

Here: the slide is read at `image_work_size` (default 5x) and cut into tiles of `tile_px` pixels. For each tile
the blur index of Crete et al. 2007 ("The blur effect: perception and estimation with a new no-reference
perceptual blur metric", SPIE 6492) is computed: the image is re-blurred with a 9-pixel box filter and the index
is how little the local gradients change under that extra blur, normalised by the gradients present -> 0 (sharp)
to 1 (already blurred), largely independent of how much texture or stain the tile holds. Tiles whose index is
above `blur_threshold` and that hold at least `min_tissue_frac` tissue are marked blurry.

Reported columns (same names as the stock step so downstream tooling is unchanged, plus two):
  blurry_removed_num_regions, blurry_removed_mean_area, blurry_removed_max_area, blurry_removed_percent
  blur_index_median, blur_index_p90   -- over tissue tiles, before thresholding (slide-level sharpness summary)

Config:
  [BlurDetectionNormalized.identifyBlurryRegionsNormalized]
  image_work_size = 5x
  tile_px = 128           # at 5x on a 0.25 um/px scan: 128 px = 256 um
  blur_threshold = 0.6
  min_tissue_frac = 0.5
"""
import logging
import os

import numpy as np
from scipy.ndimage import uniform_filter1d
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray

from histoqc.BaseImage import printMaskHelper, getMaskRegionsStats


def crete_blur_tiles(g: np.ndarray, tile: int, k: int = 9):
    """Per-tile Crete blur index of a 2-D float image g (0..1). Returns (n, m) array in [0, 1]; NaN where the tile
    has no gradient at all (flat)."""
    n, m = g.shape[0] // tile, g.shape[1] // tile
    if n == 0 or m == 0:
        return np.full((max(n, 1), max(m, 1)), np.nan, dtype=np.float32)
    P = k
    gp = np.pad(g[:n * tile, :m * tile].astype(np.float32), P, mode="reflect")
    out = np.full((n, m), np.nan, dtype=np.float32)
    for i in range(n):
        band = gp[i * tile:(i + 1) * tile + 2 * P, :]
        bh = uniform_filter1d(band, k, axis=1, mode="nearest")
        bv = uniform_filter1d(band, k, axis=0, mode="nearest")
        dfh = np.abs(np.diff(band, axis=1));  dbh = np.abs(np.diff(bh, axis=1))
        dfv = np.abs(np.diff(band, axis=0));  dbv = np.abs(np.diff(bv, axis=0))
        vh = np.clip(dfh - dbh, 0, None);     vv = np.clip(dfv - dbv, 0, None)
        sl = (slice(P, P + tile), slice(P, P + m * tile))
        def blk(a):
            return a[sl].reshape(tile, m, tile).sum(axis=(0, 2))
        sfh, svh, sfv, svv = blk(dfh), blk(vh), blk(dfv), blk(vv)
        with np.errstate(divide="ignore", invalid="ignore"):
            b_h = np.where(sfh > 0, (sfh - svh) / sfh, np.nan)
            b_v = np.where(sfv > 0, (sfv - svv) / sfv, np.nan)
        out[i] = np.fmax(b_h, b_v)
    return out


def tile_tissue_frac(mask: np.ndarray, shape_hi, tile: int):
    """Fraction of each (tile x tile at the high-res grid) tile covered by `mask` (given at a lower resolution)."""
    n, m = shape_hi[0] // tile, shape_hi[1] // tile
    ys = (np.arange(n * tile) * mask.shape[0] / shape_hi[0]).astype(int).clip(0, mask.shape[0] - 1)
    xs = (np.arange(m * tile) * mask.shape[1] / shape_hi[1]).astype(int).clip(0, mask.shape[1] - 1)
    hi = mask[np.ix_(ys, xs)].astype(np.float32)
    return hi.reshape(n, tile, m, tile).mean(axis=(1, 3))


def tiles_to_mask(flag: np.ndarray, shape_hi, tile: int, shape_lo):
    """Boolean tile grid -> full-resolution (shape_hi) -> nearest-neighbour to shape_lo."""
    n, m = flag.shape
    hi = np.zeros(shape_hi, dtype=bool)
    hi[:n * tile, :m * tile] = np.repeat(np.repeat(flag, tile, axis=0), tile, axis=1)
    ys = (np.arange(shape_lo[0]) * shape_hi[0] / shape_lo[0]).astype(int).clip(0, shape_hi[0] - 1)
    xs = (np.arange(shape_lo[1]) * shape_hi[1] / shape_lo[1]).astype(int).clip(0, shape_hi[1] - 1)
    return hi[np.ix_(ys, xs)]


def identifyBlurryRegionsNormalized(s, params):
    logging.info(f"{s['filename']} - \tidentifyBlurryRegionsNormalized")
    tile = int(params.get("tile_px", 128))
    thr = float(params.get("blur_threshold", 0.6))
    min_tissue = float(params.get("min_tissue_frac", 0.5))

    img = s.getImgThumb(params.get("image_work_size", "5x"))
    g = rgb2gray(img).astype(np.float32)
    del img
    prev_mask = s["img_mask_use"]
    bidx = crete_blur_tiles(g, tile)
    tfrac = tile_tissue_frac(prev_mask, g.shape, tile)
    del g
    tissue_tiles = tfrac >= min_tissue
    vals = bidx[tissue_tiles & np.isfinite(bidx)]
    med = float(np.median(vals)) if vals.size else float("nan")
    p90 = float(np.percentile(vals, 90)) if vals.size else float("nan")

    flag = tissue_tiles & np.isfinite(bidx) & (bidx > thr)
    mask = tiles_to_mask(flag, (bidx.shape[0] * tile, bidx.shape[1] * tile), tile, prev_mask.shape) & prev_mask

    io.imsave(s["outdir"] + os.sep + s["filename"] + "_blurry.png", img_as_ubyte(mask), check_contrast=False)
    s["img_mask_blurry"] = mask
    s["img_mask_use"] = prev_mask & ~mask

    stat_rs = getMaskRegionsStats(mask)
    s.addToPrintList("blurry_removed_num_regions", str(stat_rs.get("num", 0)))
    s.addToPrintList("blurry_removed_mean_area", str(stat_rs.get("area_mean", 0)))
    s.addToPrintList("blurry_removed_max_area", str(stat_rs.get("area_max", 0)))
    s.addToPrintList("blurry_removed_percent",
                     printMaskHelper(params.get("mask_statistics", s["mask_statistics"]), prev_mask, s["img_mask_use"]))
    s.addToPrintList("blur_index_median", f"{med:.4f}")
    s.addToPrintList("blur_index_p90", f"{p90:.4f}")

    if len(s["img_mask_use"].nonzero()[0]) == 0:
        logging.warning(f"{s['filename']} - After BlurDetectionNormalized NO tissue remains detectable!")
        s["warnings"].append("After BlurDetectionNormalized NO tissue remains detectable! Downstream modules likely to be incorrect/fail")
    return
