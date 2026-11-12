import argparse
import os
import time
import warnings
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data as data

from util import (eval_forward, evaluate_psnr, get_models, set_eval,
                  save_numpy_array_as_image, as_img_array, get_ms_ssim)
from torchvision import transforms
from dataset import get_loader
from mask_codec import bitrate_bytes, code_bitrate_bytes
from semantic_metrics import vmaf, lpips, lpips_map
from semantic_metrics import psnr_masked, msssim_bbox, lpips_roi

EvalResult = namedtuple('EvalResult', [
    'loss', 'msssim', 'psnr', 'psnr_ee1', 'psnr_ee2', 'psnr_ee3', 'psnr_ee4',
    'msssim_ee1', 'msssim_ee2', 'msssim_ee3', 'msssim_ee4',
    'psnr_roi', 'psnr_bg', 'psnr_edge', 'msssim_bbox',
    'lpips_roi', 'lpips_bg', 'lpips_edge', 'lpips_full',
    'mask_bytes', 'edge_bytes', 'mask_iou', 'edge_f1',
    'vmaf', 'lpips', 'total_bytes_per_frame',
    'prefix_visual_bytes_raw', 'prefix_visual_bytes', 'prefix_total_bytes',
])


def save_codes(name, codes):
  codes = (codes.astype(np.int8) + 1) // 2
  export = np.packbits(codes.reshape(-1))
  np.savez_compressed(
      name + '.codes',
      shape=codes.shape,
      codes=export)


def _eval_workers(args):
    """Number of threads for the CPU-bound MS-SSIM metric pass.

    ``--eval-workers 0`` (default) auto-selects ``min(cpu_count, 8)``, the
    measured sweet spot for MS-SSIM (it is FFT/convolution-bound, not BLAS-bound,
    and parallel speedup plateaus around 8 threads).
    """
    w = getattr(args, 'eval_workers', 0) or 0
    if w <= 0:
        w = min(os.cpu_count() or 1, 8)
    return max(1, int(w))


def _map_parallel(fn, indices, workers):
    """Map ``fn`` over ``indices`` on a thread pool when it is worth it."""
    if workers > 1 and len(indices) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, indices))
    return [fn(i) for i in indices]


def _nanmean_or_none(rows):
    """Column-wise mean that ignores NaN, or None when there are no rows."""
    if not rows:
        return None
    arr = np.array(rows, dtype=np.float64)
    if arr.size == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(arr, axis=0)


def _slot_visual_bytes(code_full, iterations, bits):
    """Cumulative raw packed bytes per progressive prefix for one code map.

    ``code_full`` is ``(iterations * bits, h, w)`` of values in {-1, 0, 1}. Each
    element is one bit, so a slot contributes ``n_transmitted_positions / 8``
    bytes. For unmasked maps (all values in {-1, 1}) one slot therefore equals
    ``bits * h * w / 8``. A zero entry is a masked (non-transmitted) position.
    """
    slots = code_full.reshape(iterations, bits,
                              code_full.shape[-2], code_full.shape[-1])
    bytes_per_slot = []
    for s in range(iterations):
        n_positions = int((slots[s] != 0).sum())
        bytes_per_slot.append(n_positions / 8.0)
    return np.cumsum(bytes_per_slot, dtype=np.float64)


def _prefix_visual_bytes_compressed(code_full, iterations, bits):
    """Compressed code bytes for each progressive prefix.

    This matches the accounting of ``code_bitrate_bytes`` (DEFLATE over
    ``np.packbits``) that ``total_bytes_per_frame`` uses. For prefix ``s`` the
    slots after ``s`` are zeroed so their positions are omitted, then the whole
    prefix map is compressed. The result is non-decreasing in ``s``.
    """
    out = []
    for s in range(iterations):
        prefix = np.zeros_like(code_full)
        prefix[:(s + 1) * bits] = code_full[:(s + 1) * bits]
        out.append(float(code_bitrate_bytes(prefix)))
    return np.asarray(out, dtype=np.float64)


def save_probe_frame(args, output_suffix, filenames, original, out_imgs,
                     out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4):
  """Save one fixed frame (first sample) with all reconstruction layers.

  The same frame is saved at every evaluation, keyed by ``output_suffix``
  (the training iteration), so image quality can be compared across
  evaluations and across the progressive/early-exit layers.
  """
  if getattr(args, 'rank', 0) != 0:
    return

  probe_dir = os.path.join(args.out_dir, 'probe')
  os.makedirs(probe_dir, exist_ok=True)

  def save(name, arr):
    save_numpy_array_as_image(
        os.path.join(probe_dir, '%s_%s.png' % (output_suffix, name)), arr)

  save('original', original[0])
  for step, img in enumerate(out_imgs):
    save('step%02d' % (step + 1), img[0])
  for idx, ee in enumerate((out_imgs_ee1, out_imgs_ee2,
                            out_imgs_ee3, out_imgs_ee4), 1):
    save('ee%d' % idx, ee[-1][0])

  print('Saved probe frame %s -> %s' % (filenames[0], probe_dir))


def finish_batch(args, filenames, original, out_imgs,
                 losses, code_batch, output_suffix):

  workers = _eval_workers(args)
  msssim_mode = getattr(args, 'eval_msssim', 'all')
  msssim_all, psnr_all = get_msssim_psnr(
      original, out_imgs, workers, msssim_mode)

  all_losses, all_msssim, all_psnr = [], [], []
  for ex_idx in range(original.shape[0]):
      if args.save_codes:
        rel = os.path.relpath(filenames[ex_idx], args.eval).replace(os.sep, '_')
        save_codes(
          os.path.join(args.out_dir, output_suffix, 'codes', rel[:-4]),
          code_batch[:, ex_idx, :, :, :]
        )

      all_losses.append(losses)
      all_msssim.append(msssim_all[ex_idx])
      all_psnr.append(psnr_all[ex_idx])

  return all_losses, all_msssim, all_psnr


def get_msssim_psnr(original, out_imgs, workers=1, msssim_mode='all'):
    """Return per-example ``(msssim, psnr)`` arrays over the progressive steps.

    PSNR is always computed for every step (cheap). ``msssim_mode`` controls the
    MS-SSIM steps: ``'all'`` (every step), ``'final'`` (last step only), or
    ``'off'`` (none). Non-computed MS-SSIM slots are NaN so per-step logging can
    skip them.
    """
    num_steps = len(out_imgs)
    if msssim_mode == 'final':
        step_idxs = [num_steps - 1]
    elif msssim_mode == 'off':
        step_idxs = []
    else:
        step_idxs = list(range(num_steps))

    def compute(ex_idx):
        orig = original[None, ex_idx]
        outs = [o[None, ex_idx] for o in out_imgs]
        psnr = evaluate_psnr(orig, outs)
        msssim = np.full(num_steps, np.nan)
        for s in step_idxs:
            msssim[s] = get_ms_ssim(orig, outs[s])
        return msssim, psnr

    results = _map_parallel(compute, list(range(original.shape[0])), workers)
    all_msssim = np.array([r[0] for r in results])
    all_psnr = np.array([r[1] for r in results])
    return all_msssim, all_psnr


def _region_flags(mask, edge):
    """Return ``(has_mask, has_bg, has_edge)`` for one sample.

    ROI/background metrics need an object mask. Background also needs at least
    one non-object pixel. Edge metrics use the edge map even when the object
    mask is empty, so edge-only frames are not dropped.
    """
    has_mask = bool(np.asarray(mask).sum() > 0)
    has_bg = has_mask and bool((1.0 - np.asarray(mask)).sum() > 0)
    has_edge = bool(np.asarray(edge).sum() > 0)
    return has_mask, has_bg, has_edge


def _lpips_regions(ref, dist, mask, edge, device, net):
    """Return (full, roi, bg, edge) official LPIPS scalars from one map.

    ``mask`` and ``edge`` are raw (unfeathered) ``(1, 1, H, W)`` tensors.
    Region scalars are NaN when the region is empty.
    """
    dmap = lpips_map(ref, dist, device=device, net=net)
    if dmap is None:
        return None, None, None, None
    full = float(dmap.mean())

    def masked(m):
        total = m.sum()
        if float(total) <= 0.0:
            return np.nan
        return float((dmap * m).sum() / total)

    roi = masked(mask)
    bg = masked(1.0 - mask)
    edge_value = masked(edge)
    return full, roi, bg, edge_value


def _group_boundaries(phase_schedule, num_steps):
    """Return the last 1-based step index of each phase group."""
    boundaries = []
    for phase in sorted(set(phase_schedule)):
        idx = max(i for i, p in enumerate(phase_schedule) if p == phase)
        boundaries.append(idx + 1)
    return boundaries


def _object_complete_step(phase_schedule, num_steps):
    """1-based step index at which the object phase (phase 1) completes.

    This is the last progressive step whose phase is 1, so an ROI-late schedule
    reports the correct boundary instead of a hardcoded step 4.
    """
    if isinstance(phase_schedule, str):
        schedule = [int(x) for x in phase_schedule.split(',') if x.strip() != '']
    else:
        schedule = list(phase_schedule)
    idxs = [i for i, p in enumerate(schedule) if p == 1]
    if idxs:
        return max(idxs) + 1
    return num_steps


def print_hierarchy_summary(result, phase_schedule, name='VTL'):
    """Print per-region per-step deltas and group-boundary cumulative gains.

    ``result`` is an ``EvalResult`` whose per-step arrays are already averaged
    across the full (or reduced) evaluation set.
    """
    psnr_full = result.psnr
    psnr_roi = result.psnr_roi
    psnr_bg = result.psnr_bg
    psnr_edge = result.psnr_edge
    num_steps = len(psnr_full)

    regions = [('full', psnr_full), ('roi', psnr_roi),
               ('bg', psnr_bg), ('edge', psnr_edge)]

    print('\n[%s] Per-step PSNR delta (step-over-step) per region:' % name)
    header = '%-5s' % 'Step' + ''.join('%-14s' % r for r, _ in regions)
    print(header)
    for step in range(num_steps):
        row = '%-5d' % (step + 1)
        for _, arr in regions:
            if arr is None:
                row += '%-14s' % 'n/a'
            elif step == 0:
                row += '%-14s' % '+init'
            else:
                row += '%+13.4f ' % (arr[step] - arr[step - 1])
        print(row)

    boundaries = _group_boundaries(phase_schedule, num_steps)
    print('\n[%s] Group-boundary cumulative gain per region:' % name)
    print('%-6s' % 'Boundary' + ''.join('%-14s' % r for r, _ in regions))
    prev = {r: 0.0 for r, _ in regions if _ is not None}
    for boundary in boundaries:
        row = 'L%-5d' % boundary
        for rname, arr in regions:
            if arr is None:
                row += '%-14s' % 'n/a'
                continue
            row += '%+13.4f ' % (arr[boundary - 1] - prev[rname])
            prev[rname] = arr[boundary - 1]
        print(row)

    if result.psnr_roi is not None and result.psnr_bg is not None:
        final_roi = result.psnr_roi[-1]
        final_bg = result.psnr_bg[-1]
        if boundaries:
            obj_step = _object_complete_step(phase_schedule, num_steps)
            roi_frac = (result.psnr_roi[obj_step - 1] / final_roi) if final_roi > 0 else float('nan')
            bg_frac = (result.psnr_bg[obj_step - 1] / final_bg) if final_bg > 0 else float('nan')
            print('[%s] Object-intact-at-L%d: ROI(L%d)/ROI(L%d)=%.4f '
                  'BG(L%d)/BG(L%d)=%.4f (object ~1.0, background <1.0 expected)'
                  % (name, obj_step, obj_step, num_steps, roi_frac,
                     obj_step, num_steps, bg_frac))

    raw_prefix = getattr(result, 'prefix_visual_bytes_raw', None)
    if raw_prefix is not None:
        print('[%s] Per-prefix raw packed visual bytes (before compression):' % name)
        print(' '.join('L%d=%.1f' % (i + 1, raw_prefix[i])
                       for i in range(num_steps)))
    if result.prefix_total_bytes is not None:
        print('[%s] Per-prefix compressed bytes (visual + mask/edge):' % name)
        print(' '.join('L%d=%.1f' % (i + 1, result.prefix_total_bytes[i])
                       for i in range(num_steps)))


def run_eval(model, eval_loader, args, output_suffix='',
             writer=None, global_step=0, tag_prefix='eval'):

  for sub_dir in ['codes']:
    cur_eval_dir = os.path.join(args.out_dir, output_suffix, sub_dir)
    if not os.path.exists(cur_eval_dir):
      print("Creating directory %s." % cur_eval_dir)
      os.makedirs(cur_eval_dir, exist_ok=True)

  all_losses, all_msssim, all_psnr = [], [], []
  all_psnr_ee1, all_psnr_ee2, all_psnr_ee3, all_psnr_ee4 = [], [], [], []
  all_msssim_ee1, all_msssim_ee2, all_msssim_ee3, all_msssim_ee4 = [], [], [], []
  all_psnr_roi, all_psnr_bg, all_psnr_edge, all_msssim_bbox = [], [], [], []
  all_lpips_roi, all_lpips_bg, all_lpips_edge, all_lpips_full = [], [], [], []
  all_mask_bytes, all_edge_bytes, all_visual_bytes = [], [], []
  all_prefix_raw, all_prefix_visual, all_prefix_total = [], [], []
  all_vmaf, all_lpips = [], []
  any_mask = False
  any_edge = False
  semantic_enabled = bool(
      getattr(args, 'eval_mask', None) or getattr(args, 'eval_edge', None))

  workers = _eval_workers(args)
  ee_msssim_mode = getattr(args, 'eval_ee_msssim', 'all')
  bbox_mode = getattr(args, 'eval_bbox_msssim', 'all')
  start_time = time.time()
  for i, (batch, ctx_frames, filenames) in enumerate(eval_loader):

      with torch.inference_mode():
          batch = batch.cuda(non_blocking=True)

          if getattr(args, 'eval_fp16', False):
              with torch.autocast(device_type='cuda', dtype=torch.float16):
                  original, out_imgs, out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4, losses, code_batch = eval_forward(
                          model, (batch, ctx_frames), args)
          else:
              original, out_imgs, out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4, losses, code_batch = eval_forward(
                  model, (batch, ctx_frames), args)

          if writer is not None and i == 0:
              writer.add_image(
                  '%s/images/original' % tag_prefix,
                  original[0], global_step, dataformats='CHW')
              writer.add_image(
                  '%s/images/recon_step_01' % tag_prefix,
                  out_imgs[0][0], global_step, dataformats='CHW')
              writer.add_image(
                  '%s/images/recon_final' % tag_prefix,
                  out_imgs[-1][0], global_step, dataformats='CHW')
              writer.add_image(
                  '%s/images/ee1_final' % tag_prefix,
                  out_imgs_ee1[-1][0], global_step, dataformats='CHW')
              writer.add_image(
                  '%s/images/ee2_final' % tag_prefix,
                  out_imgs_ee2[-1][0], global_step, dataformats='CHW')
              writer.add_image(
                  '%s/images/ee3_final' % tag_prefix,
                  out_imgs_ee3[-1][0], global_step, dataformats='CHW')
              writer.add_image(
                  '%s/images/ee4_final' % tag_prefix,
                  out_imgs_ee4[-1][0], global_step, dataformats='CHW')

          if i == 0:
              save_probe_frame(args, output_suffix, filenames, original,
                               out_imgs, out_imgs_ee1, out_imgs_ee2,
                               out_imgs_ee3, out_imgs_ee4)

          losses, msssim, psnr = finish_batch(
                  args, filenames, original, out_imgs,
                  losses, code_batch, output_suffix)

          ee_msssim, ee_psnr = [], []
          for ee in (out_imgs_ee1, out_imgs_ee2, out_imgs_ee3, out_imgs_ee4):
              m, p = get_msssim_psnr(original, ee, workers, ee_msssim_mode)
              ee_msssim.append(m)
              ee_psnr.append(p)

          all_losses += losses
          all_msssim += msssim
          all_psnr += psnr

          all_psnr_ee1 += [p for p in ee_psnr[0]]
          all_psnr_ee2 += [p for p in ee_psnr[1]]
          all_psnr_ee3 += [p for p in ee_psnr[2]]
          all_psnr_ee4 += [p for p in ee_psnr[3]]
          all_msssim_ee1 += [m for m in ee_msssim[0]]
          all_msssim_ee2 += [m for m in ee_msssim[1]]
          all_msssim_ee3 += [m for m in ee_msssim[2]]
          all_msssim_ee4 += [m for m in ee_msssim[3]]

          num_steps = len(out_imgs)
          batch_size = batch.shape[0]
          prefix_raw_by_ex = [None] * batch_size
          prefix_comp_by_ex = [None] * batch_size
          visual_by_ex = [None] * batch_size
          if code_batch.size > 0:
            for ex in range(batch_size):
              code_full = code_batch[-1, ex]
              visual_by_ex[ex] = float(code_bitrate_bytes(code_full))
              prefix_raw_by_ex[ex] = _slot_visual_bytes(
                  code_full, args.iterations, args.bits)
              prefix_comp_by_ex[ex] = _prefix_visual_bytes_compressed(
                  code_full, args.iterations, args.bits)

          mask_bytes_by_ex = [0.0] * batch_size
          edge_bytes_by_ex = [0.0] * batch_size

          if batch.shape[1] >= 15:
            S_np = batch[:, 13:14].cpu().numpy()
            E_np = batch[:, 14:15].cpu().numpy()
            use_lpips_roi = getattr(args, 'use_lpips', False)
            lpips_net = getattr(args, 'lpips_net', 'alex')
            for ex in range(batch_size):
              mask = S_np[ex, 0]
              edge = E_np[ex, 0]
              has_mask, has_bg, has_edge = _region_flags(mask, edge)

              mask_bytes_by_ex[ex] = float(bitrate_bytes(mask))
              edge_bytes_by_ex[ex] = float(bitrate_bytes(edge))
              any_mask = any_mask or has_mask
              any_edge = any_edge or has_edge

              if use_lpips_roi:
                mask_t = torch.as_tensor(mask, dtype=torch.float32,
                                         device=batch.device).view(1, 1, *mask.shape)
                edge_t = torch.as_tensor(edge, dtype=torch.float32,
                                         device=batch.device).view(1, 1, *edge.shape)
                lpips_full_steps, lpips_roi_steps = [], []
                lpips_bg_steps, lpips_edge_steps = [], []
                for step in range(num_steps):
                  full, roi, bg, edge_value = _lpips_regions(
                      original[ex], out_imgs[step][ex],
                      mask_t, edge_t, batch.device, lpips_net)
                  if full is None:
                    lpips_full_steps.append(np.nan)
                    lpips_roi_steps.append(np.nan)
                    lpips_bg_steps.append(np.nan)
                    lpips_edge_steps.append(np.nan)
                  else:
                    lpips_full_steps.append(full)
                    lpips_roi_steps.append(roi if has_mask else np.nan)
                    lpips_bg_steps.append(bg if has_bg else np.nan)
                    lpips_edge_steps.append(edge_value if has_edge else np.nan)
                all_lpips_full.append(lpips_full_steps)
                if has_mask:
                  all_lpips_roi.append(lpips_roi_steps)
                if has_bg:
                  all_lpips_bg.append(lpips_bg_steps)
                if has_edge:
                  all_lpips_edge.append(lpips_edge_steps)

              if not (has_mask or has_edge):
                continue

              orig_u8 = as_img_array(original[None, ex])[0]
              roi_steps, bg_steps, edge_steps, bbox_steps = [], [], [], []
              for step in range(num_steps):
                recon_u8 = as_img_array(out_imgs[step][None, ex])[0]
                if has_mask:
                  roi_steps.append(psnr_masked(orig_u8, recon_u8, mask))
                if has_bg:
                  bg_steps.append(psnr_masked(orig_u8, recon_u8, 1.0 - mask))
                if has_edge:
                  edge_steps.append(psnr_masked(orig_u8, recon_u8, edge))
                if has_mask:
                  if (bbox_mode == 'all'
                          or (bbox_mode == 'final' and step == num_steps - 1)):
                    bbox_steps.append(msssim_bbox(orig_u8, recon_u8, mask))
                  else:
                    bbox_steps.append(np.nan)
              if has_mask:
                all_psnr_roi.append(roi_steps)
                all_msssim_bbox.append(bbox_steps)
              if has_bg:
                all_psnr_bg.append(bg_steps)
              if has_edge:
                all_psnr_edge.append(edge_steps)

          for ex in range(batch_size):
            all_mask_bytes.append(mask_bytes_by_ex[ex])
            all_edge_bytes.append(edge_bytes_by_ex[ex])
            if visual_by_ex[ex] is None:
              continue
            all_visual_bytes.append(visual_by_ex[ex])
            all_prefix_raw.append(prefix_raw_by_ex[ex])
            all_prefix_visual.append(prefix_comp_by_ex[ex])
            extra_bytes = (mask_bytes_by_ex[ex] + edge_bytes_by_ex[ex]
                           if semantic_enabled else 0.0)
            all_prefix_total.append(prefix_comp_by_ex[ex] + extra_bytes)

          if getattr(args, 'use_vmaf', False):
            for ex in range(original.shape[0]):
              value = vmaf(original[ex], out_imgs[-1][ex])
              if value is not None:
                all_vmaf.append(value)

          if getattr(args, 'use_lpips', False):
            for ex in range(original.shape[0]):
              value = lpips(original[ex], out_imgs[-1][ex],
                            device=batch.device)
              if value is not None:
                all_lpips.append(value)

      if i % 10 == 0:
        print('\tevaluating iter %d (%f seconds)...' % (
          i, time.time() - start_time))

  if semantic_enabled:
    mask_bytes = float(np.mean(all_mask_bytes)) if all_mask_bytes else None
    edge_bytes = float(np.mean(all_edge_bytes)) if all_edge_bytes else None
  else:
    mask_bytes = None
    edge_bytes = None
  visual_bytes = float(np.mean(all_visual_bytes)) if all_visual_bytes else None
  vmaf_mean = float(np.mean(all_vmaf)) if all_vmaf else None
  lpips_mean = float(np.mean(all_lpips)) if all_lpips else None

  psnr_roi = np.array(all_psnr_roi).mean(axis=0) if all_psnr_roi else None
  psnr_bg = np.array(all_psnr_bg).mean(axis=0) if all_psnr_bg else None
  psnr_edge = np.array(all_psnr_edge).mean(axis=0) if all_psnr_edge else None
  msssim_bbox_mean = np.array(all_msssim_bbox).mean(axis=0) if all_msssim_bbox else None
  lpips_roi_mean = np.array(all_lpips_roi).mean(axis=0) if all_lpips_roi else None
  lpips_bg_mean = _nanmean_or_none(all_lpips_bg)
  lpips_edge_mean = _nanmean_or_none(all_lpips_edge)
  lpips_full_mean = _nanmean_or_none(all_lpips_full)

  mask_iou = 1.0 if any_mask else None
  edge_f1 = 1.0 if any_edge else None

  prefix_raw = None
  prefix_visual = None
  prefix_total = None
  if all_prefix_raw:
    prefix_raw = np.array(all_prefix_raw).mean(axis=0)
  if all_prefix_visual:
    prefix_visual = np.array(all_prefix_visual).mean(axis=0)
  if all_prefix_total:
    prefix_total = np.array(all_prefix_total).mean(axis=0)

  total_bytes = None
  if visual_bytes is not None:
    total_bytes = visual_bytes
    if mask_bytes is not None:
      total_bytes += mask_bytes
    if edge_bytes is not None:
      total_bytes += edge_bytes

  return EvalResult(
      np.array(all_losses).mean(axis=0),
      np.array(all_msssim).mean(axis=0),
      np.array(all_psnr).mean(axis=0),
      np.array(all_psnr_ee1).mean(axis=0),
      np.array(all_psnr_ee2).mean(axis=0),
      np.array(all_psnr_ee3).mean(axis=0),
      np.array(all_psnr_ee4).mean(axis=0),
      np.array(all_msssim_ee1).mean(axis=0) if all_msssim_ee1 else None,
      np.array(all_msssim_ee2).mean(axis=0) if all_msssim_ee2 else None,
      np.array(all_msssim_ee3).mean(axis=0) if all_msssim_ee3 else None,
      np.array(all_msssim_ee4).mean(axis=0) if all_msssim_ee4 else None,
      psnr_roi,
      psnr_bg,
      psnr_edge,
      msssim_bbox_mean,
      lpips_roi_mean,
      lpips_bg_mean,
      lpips_edge_mean,
      lpips_full_mean,
      mask_bytes,
      edge_bytes,
      mask_iou,
      edge_f1,
      vmaf_mean,
      lpips_mean,
      total_bytes,
      prefix_raw,
      prefix_visual,
      prefix_total,
      )
