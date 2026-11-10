import argparse
import os
import time
from collections import namedtuple

import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data as data

from util import eval_forward, evaluate, evaluate_psnr, get_models, set_eval, save_numpy_array_as_image, as_img_array
from torchvision import transforms
from dataset import get_loader
from mask_codec import bitrate_bytes
from semantic_metrics import vmaf, lpips
from semantic_metrics import psnr_masked, msssim_bbox, lpips_roi, vgg_reference_acts

EvalResult = namedtuple('EvalResult', [
    'loss', 'msssim', 'psnr', 'psnr_ee1', 'psnr_ee2', 'psnr_ee3', 'psnr_ee4',
    'psnr_roi', 'psnr_bg', 'msssim_bbox', 'lpips_roi',
    'mask_bytes', 'edge_bytes', 'mask_iou', 'edge_f1',
    'vmaf', 'lpips', 'total_bytes_per_frame',
])


def save_codes(name, codes):
  codes = (codes.astype(np.int8) + 1) // 2
  export = np.packbits(codes.reshape(-1))
  np.savez_compressed(
      name + '.codes',
      shape=codes.shape,
      codes=export)


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

  all_losses, all_msssim, all_psnr = [], [], []
  for ex_idx, filename in enumerate(filenames):
      if args.save_codes:
        rel = os.path.relpath(filename, args.eval).replace(os.sep, '_')
        save_codes(
          os.path.join(args.out_dir, output_suffix, 'codes', rel[:-4]),
          code_batch[:, ex_idx, :, :, :]
        )

      msssim, psnr = evaluate(
        original[None, ex_idx],
        [out_img[None, ex_idx] for out_img in out_imgs])

      all_losses.append(losses)
      all_msssim.append(msssim)
      all_psnr.append(psnr)

  return all_losses, all_msssim, all_psnr


def get_psnr(args, filenames, original, out_imgs):

  all_psnr = []
  for ex_idx, filename in enumerate(filenames):

      psnr = evaluate_psnr(
        original[None, ex_idx],
        [out_img[None, ex_idx] for out_img in out_imgs])

      all_psnr.append(psnr)

  return all_psnr


def run_eval(model, eval_loader, args, output_suffix='',
             writer=None, global_step=0, tag_prefix='eval'):

  for sub_dir in ['codes']:
    cur_eval_dir = os.path.join(args.out_dir, output_suffix, sub_dir)
    if not os.path.exists(cur_eval_dir):
      print("Creating directory %s." % cur_eval_dir)
      os.makedirs(cur_eval_dir, exist_ok=True)

  all_losses, all_msssim, all_psnr = [], [], []
  all_psnr_ee1, all_psnr_ee2, all_psnr_ee3, all_psnr_ee4 = [], [], [], []
  all_psnr_roi, all_psnr_bg, all_msssim_bbox, all_lpips_roi = [], [], [], []
  all_mask_bytes, all_edge_bytes, all_visual_bytes = [], [], []
  all_vmaf, all_lpips = [], []
  any_mask = False

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

          psnr_ee1 = get_psnr(args, filenames, original, out_imgs_ee1)
          psnr_ee2 = get_psnr(args, filenames, original, out_imgs_ee2)
          psnr_ee3 = get_psnr(args, filenames, original, out_imgs_ee3)
          psnr_ee4 = get_psnr(args, filenames, original, out_imgs_ee4)

          all_losses += losses
          all_msssim += msssim
          all_psnr += psnr

          all_psnr_ee1 += psnr_ee1
          all_psnr_ee2 += psnr_ee2
          all_psnr_ee3 += psnr_ee3
          all_psnr_ee4 += psnr_ee4

          num_steps = len(out_imgs)
          for ex in range(batch.shape[0]):
            if code_batch.size > 0:
              all_visual_bytes.append(bitrate_bytes(code_batch[-1, ex]))

          if batch.shape[1] >= 15:
            S_np = batch[:, 13:14].cpu().numpy()
            E_np = batch[:, 14:15].cpu().numpy()
            use_lpips_roi = getattr(args, 'use_lpips', False)
            for ex in range(batch.shape[0]):
              mask = S_np[ex, 0]
              edge = E_np[ex, 0]
              all_mask_bytes.append(bitrate_bytes(mask))
              all_edge_bytes.append(bitrate_bytes(edge))
              if mask.sum() <= 0:
                continue
              any_mask = True
              has_bg = (1.0 - mask).sum() > 0
              orig_u8 = as_img_array(original[None, ex])[0]
              roi_steps, bg_steps, bbox_steps, lpips_roi_steps = [], [], [], []
              ref_acts = None
              if use_lpips_roi:
                ref_acts = vgg_reference_acts(original[ex], device=batch.device)
              for step in range(num_steps):
                recon_u8 = as_img_array(out_imgs[step][None, ex])[0]
                roi_steps.append(psnr_masked(orig_u8, recon_u8, mask))
                if has_bg:
                  bg_steps.append(psnr_masked(orig_u8, recon_u8, 1.0 - mask))
                bbox_steps.append(msssim_bbox(orig_u8, recon_u8, mask))
                if use_lpips_roi:
                  lpips_roi_steps.append(lpips_roi(
                      original[ex], out_imgs[step][ex], mask,
                      device=batch.device, ref_acts=ref_acts))
              all_psnr_roi.append(roi_steps)
              if has_bg:
                all_psnr_bg.append(bg_steps)
              all_msssim_bbox.append(bbox_steps)
              if lpips_roi_steps:
                all_lpips_roi.append(lpips_roi_steps)

          if getattr(args, 'use_vmaf', False):
            for ex in range(original.shape[0]):
              all_vmaf.append(vmaf(original[ex], out_imgs[-1][ex]))

          if getattr(args, 'use_lpips', False):
            for ex in range(original.shape[0]):
              all_lpips.append(lpips(original[ex], out_imgs[-1][ex],
                                     device=batch.device))

      if i % 10 == 0:
        print('\tevaluating iter %d (%f seconds)...' % (
          i, time.time() - start_time))

  mask_bytes = float(np.mean(all_mask_bytes)) if all_mask_bytes else None
  edge_bytes = float(np.mean(all_edge_bytes)) if all_edge_bytes else None
  visual_bytes = float(np.mean(all_visual_bytes)) if all_visual_bytes else None
  vmaf_mean = float(np.nanmean(all_vmaf)) if all_vmaf else None
  lpips_mean = float(np.nanmean(all_lpips)) if all_lpips else None

  psnr_roi = np.array(all_psnr_roi).mean(axis=0) if all_psnr_roi else None
  psnr_bg = np.array(all_psnr_bg).mean(axis=0) if all_psnr_bg else None
  msssim_bbox_mean = np.array(all_msssim_bbox).mean(axis=0) if all_msssim_bbox else None
  lpips_roi_mean = np.array(all_lpips_roi).mean(axis=0) if all_lpips_roi else None

  mask_iou = 1.0 if any_mask else None
  edge_f1 = 1.0 if any_mask else None

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
      psnr_roi,
      psnr_bg,
      msssim_bbox_mean,
      lpips_roi_mean,
      mask_bytes,
      edge_bytes,
      mask_iou,
      edge_f1,
      vmaf_mean,
      lpips_mean,
      total_bytes,
      )
