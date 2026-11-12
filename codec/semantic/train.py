import math
import os
import random
import time
import copy
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as LS
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable

from dataset import get_loader
from evaluate import run_eval, print_hierarchy_summary
from train_options import parser, validate_semantic_args
from util import get_models, init_lstm, set_train, set_eval, init_d2
from util import prepare_inputs, forward_ctx, feathered_mask
from semantic_loss import build_weight_map, weighted_l1, weighted_lpips
from semantic_loss import region_l1, phase_of_step, phase_multipliers, gate_for_phase
from semantic_loss import code_block_mask

import network


def get_memory_gb():
    values = {}
    with open('/proc/meminfo') as meminfo:
        for line in meminfo:
            name, value = line.split(':', 1)
            values[name] = int(value.strip().split()[0]) / 1024 ** 2
    return values


def get_rss_gb():
    with open('/proc/self/status') as status:
        for line in status:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024 ** 2
    return 0.0


def _all_reduce_mean(value):
    """Average ``value`` across all DDP ranks.

    ``value`` may be a numpy array, a scalar, or None. Returns the same kind.
    """
    if value is None:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return value
    t = torch.tensor(np.asarray(value, dtype=np.float64), device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / dist.get_world_size()
    out = t.cpu().numpy()
    if out.ndim == 0:
        return float(out)
    return out


def seed_everything(seed):
    """Seed Python, NumPy, PyTorch, and all CUDA devices.

    DDP ranks must pass ``seed + rank`` so ranks do not share the same random
    stream (identical crops across GPUs would otherwise be possible).
    """
    seed = int(seed) % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _build_lr_scheduler(solver, args, base_lr):
    """Return an LR scheduler with optional linear warmup.

    Supports a step-decay schedule (``--lr-schedule multistep``, the default)
    and a cosine schedule (``--lr-schedule cosine``). ``--lr-warmup-iters`` adds
    a linear ramp from ~0 to ``base_lr`` at the start of either schedule. The
    scheduler is a single ``LambdaLR`` so ``scheduler.last_epoch`` resume and
    ``get_last_lr`` keep working unchanged.
    """
    warmup = max(0, int(getattr(args, 'lr_warmup_iters', 0)))

    if getattr(args, 'lr_schedule', 'multistep') == 'cosine':
        t_max = max(1, int(args.max_train_iters) - warmup)
        eta_min = float(getattr(args, 'lr_eta_min', 1e-6))

        def cosine_mult(step):
            if warmup > 0 and step < warmup:
                return (step + 1) / max(1, warmup)
            t = min(float(step - warmup), float(t_max))
            ratio = 0.5 * (1.0 + math.cos(math.pi * t / t_max))
            lr = eta_min + (base_lr - eta_min) * ratio
            return lr / base_lr

        return LS.LambdaLR(solver, lr_lambda=cosine_mult)

    milestones = sorted(int(s) for s in args.schedule.split(','))

    def step_mult(step):
        if warmup > 0 and step < warmup:
            return (step + 1) / max(1, warmup)
        mult = 1.0
        for m in milestones:
            if step >= m:
                mult *= args.gamma
        return mult

    return LS.LambdaLR(solver, lr_lambda=step_mult)


def main():
    args = parser.parse_args()
    args = validate_semantic_args(args)

    if args.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True

    if args.schedule is None:
        args.schedule = ','.join(
            str(int(args.max_train_iters * f)) for f in args.schedule_frac)

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    else:
        local_rank = 0
        rank = 0
        world_size = 1

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        dist.init_process_group(backend='nccl', init_method='env://',
                                timeout=timedelta(minutes=60))

    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank

    seed_everything(args.seed + rank)

    if rank == 0:
        print(args)
        print('Seed: %d (+ rank offset); deterministic: %s'
              % (args.seed, args.deterministic))

    writer = None
    if rank == 0 and args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=args.tensorboard_dir, flush_secs=5)
            print('TensorBoard logging to %s' % args.tensorboard_dir)
        except ImportError:
            print('Warning: TensorBoard is not installed. TensorBoard logging is off.')

    def log_scalar(tag, value, step):
        if writer is not None:
            writer.add_scalar(tag, value, step)

    train_loader = get_loader(
        is_train=True, root=args.train, mv_dir=args.train_mv, args=args)

    eval_loaders = {
        'VTL': get_loader(
            is_train=False, root=args.eval, mv_dir=args.eval_mv, args=args),
    }

    encoder, binarizer, decoder, unet = get_models(
        args=args, v_compress=args.v_compress,
        bits=args.bits,
        encoder_fuse_level=args.encoder_fuse_level,
        decoder_fuse_level=args.decoder_fuse_level)

    d2 = network.DecoderCell2(
        v_compress=args.v_compress, shrink=args.shrink,
        bits=args.bits, fuse_level=args.decoder_fuse_level,
        use_conditioning=args.use_conditioning,
        conditioning_mode=getattr(args, 'conditioning_mode', 'legacy'),
        conditioning_init=getattr(args, 'conditioning_init', 'random'),
        ee_head=getattr(args, 'ee_head', 'src')).cuda()

    nets = [encoder, binarizer, decoder, d2]
    if unet is not None:
        nets.append(unet)

    if world_size > 1:
        encoder = DDP(encoder, device_ids=[local_rank], find_unused_parameters=True)
        binarizer = DDP(binarizer, device_ids=[local_rank], find_unused_parameters=True)
        decoder = DDP(decoder, device_ids=[local_rank], find_unused_parameters=True)
        d2 = DDP(d2, device_ids=[local_rank], find_unused_parameters=True)
        if unet is not None:
            unet = DDP(unet, device_ids=[local_rank], find_unused_parameters=True)

        nets = [encoder, binarizer, decoder, d2]
        if unet is not None:
            nets.append(unet)

    if rank == 0:
        print(nets)

    params = [{'params': net.parameters()} for net in nets]

    base_lr = args.lr
    if args.warm_start_lr is not None and args.load_model_dir:
        base_lr = args.warm_start_lr

    solver = optim.Adam(params, lr=base_lr)
    scheduler = _build_lr_scheduler(solver, args, base_lr)

    if rank == 0 and not os.path.exists(args.model_dir):
        print("Creating directory %s." % args.model_dir)
        os.makedirs(args.model_dir, exist_ok=True)

    names = ['encoder', 'binarizer', 'decoder', 'd2', 'unet']

    def unwrap(net):
        if isinstance(net, (nn.DataParallel, DDP)):
            return net.module
        return net

    def resume(model_name, index, model_dir=None, strict=True):
        model_dir = model_dir or args.model_dir
        for net_idx, net in enumerate(nets):
            if net is None:
                continue
            name = names[net_idx]
            checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
                model_dir, model_name, name, index)
            if rank == 0:
                print('Loading %s from %s (strict=%s)...' % (name, checkpoint_path, strict))
            state = torch.load(checkpoint_path, map_location='cuda:%d' % local_rank)
            if strict:
                unwrap(net).load_state_dict(state)
            else:
                missing, unexpected = unwrap(net).load_state_dict(state, strict=False)
                if rank == 0:
                    print('  missing keys (kept at init): %d' % len(missing))
                    print('  unexpected keys (ignored): %d' % len(unexpected))

    def save_as(suffix):
        for net_idx, net in enumerate(nets):
            if net is None:
                continue
            torch.save(unwrap(net).state_dict(),
                       '{}/{}_{}_{}.pth'.format(
                           args.model_dir, args.save_model_name,
                           names[net_idx], suffix))

    def save(index):
        if rank == 0:
            save_as('%08d' % index)

    def save_best():
        if rank == 0:
            save_as('best')

    train_iter = 0
    just_resumed = False
    if args.load_model_dir:
        # Warm start from a different model directory (e.g. C from B). Load the
        # shared weights with strict=False so new conditioning keys keep their
        # zero/random init, and start a fresh training run from iteration 0.
        if rank == 0:
            print('Warm start from %s@iter %s (strict=%s)' % (
                args.load_model_dir, args.load_iter, args.load_strict))
        resume(args.load_model_name, args.load_iter, args.load_model_dir,
               strict=(args.load_strict == 'true'))
    elif args.load_model_name:
        if rank == 0:
            print('Loading %s@iter %d' % (args.load_model_name, args.load_iter))
        resume(args.load_model_name, args.load_iter, strict=True)
        train_iter = args.load_iter
        scheduler.last_epoch = train_iter
        just_resumed = True

    best_psnr = 0.0
    best_iter = 0
    patience = 0
    stop_training = False

    epoch = 0
    while True:
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        for batch, (crops, ctx_frames, _) in enumerate(train_loader):
            train_iter += 1
            if train_iter > args.max_train_iters:
                break

            batch_t0 = time.time()

            solver.zero_grad(set_to_none=True)

            (encoder_h_1, encoder_h_2, encoder_h_3,
             decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = init_lstm(
                batch_size=(crops[0].size(0) * args.num_crops),
                height=crops[0].size(2), width=crops[0].size(3), args=args)

            if args.v_compress:
                unet_output1, unet_output2 = forward_ctx(unet, ctx_frames)
            else:
                unet_output1 = Variable(torch.zeros(args.batch_size,)).cuda()
                unet_output2 = Variable(torch.zeros(args.batch_size,)).cuda()

            res, frame1, frame2, warped_unet_output1, warped_unet_output2, S, E = prepare_inputs(
                crops, args, unet_output1, unet_output2)
            in_img = res

            sem = None
            if args.use_conditioning and S is not None and E is not None:
                sem = torch.cat([S, E], dim=1)

            is_region = args.loss_mode == 'region'
            S_f, E_f = feathered_mask(S, E, args.gate_feather)

            W = None
            if S is not None or E is not None:
                if not is_region:
                    W = build_weight_map(
                        S, E, args.alpha_sem, args.alpha_edge)

            lpips_W = None
            lpips_steps = 'final'
            if args.use_wlpips:
                if is_region:
                    lpips_W = S_f if args.lpips_scope == 'roi' else None
                    lpips_steps = args.lpips_steps
                else:
                    lpips_W = W

            region_roi_sum = 0.0
            region_bg_sum = 0.0
            region_edge_sum = 0.0
            region_fallback = 0
            region_terms = 0
            d2_outputs = []

            def region_term(recon, target, step_idx):
                phase = phase_of_step(step_idx, args.phase_schedule)
                w_roi, w_edge, w_bg = phase_multipliers(
                    phase, args.w_roi, args.w_edge, args.w_bg)
                return region_l1(
                    recon, target, S_f, E_f, w_roi, w_edge, w_bg,
                    args.region_norm,
                    partition=getattr(args, 'region_partition', 'overlap'))

            losses = []
            rec2_losses = []
            ee1_losses, ee2_losses, ee3_losses, ee4_losses = [], [], [], []

            bp_t0 = time.time()
            batch_size, _, height, width = res.size()

            out_img = torch.zeros(1, 3, height, width).cuda() + 0.5

            b, d, h, w = batch_size, args.bits, height // 16, width // 16
            code_arr = [torch.zeros(b, d, h, w).cuda() for i in range(args.iterations)]

            for i in range(args.iterations):
                phase = phase_of_step(i, args.phase_schedule)
                gate_i = gate_for_phase(phase, S_f, E_f, args.gate_mode)
                gated_res = res if gate_i is None else res * gate_i

                if args.v_compress and args.stack:
                    encoder_input = torch.cat([frame1, gated_res, frame2], dim=1)
                else:
                    encoder_input = gated_res

                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
                    warped_unet_output1, warped_unet_output2, sem)

                codes = binarizer(encoded)
                if args.spatial_bit_alloc:
                    block_mask = code_block_mask(phase, S, E, h, w)
                    if block_mask is not None:
                        codes = codes * block_mask
                code_arr[i] = codes

                (output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
                    warped_unet_output1, warped_unet_output2, sem)

                if gate_i is None:
                    res = res - output
                else:
                    res = res - output * gate_i
                out_img = out_img + output.data

                if is_region:
                    scalar, c_roi, c_bg, c_edge, c_fb = region_term(res, 0.0, i)
                    losses.append(scalar)
                    region_roi_sum += c_roi
                    region_bg_sum += c_bg
                    region_edge_sum += c_edge
                    region_fallback += c_fb
                    region_terms += 1
                else:
                    losses.append(weighted_l1(res, 0.0, W))

                (d2_h_1, d2_h_2, d2_h_3, d2_h_4) = init_d2(
                    batch_size=(crops[0].size(0) * args.num_crops),
                    height=crops[0].size(2), width=crops[0].size(3), args=args)

                codes_d2 = torch.stack(code_arr, dim=1).reshape(b, -1, h, w)
                (output_d2, out_ee1, out_ee2, out_ee3, out_ee4,
                 d2_h_1, d2_h_2, d2_h_3, d2_h_4) = d2(
                     codes_d2, d2_h_1, d2_h_2, d2_h_3, d2_h_4,
                     warped_unet_output1, warped_unet_output2, sem)

                if is_region:
                    scalar, c_roi, c_bg, c_edge, c_fb = region_term(in_img, output_d2, i)
                    rec2_losses.append(scalar)
                    region_roi_sum += c_roi
                    region_bg_sum += c_bg
                    region_edge_sum += c_edge
                    region_fallback += c_fb
                    region_terms += 1

                    scalar, c_roi, c_bg, c_edge, c_fb = region_term(in_img, out_ee1, i)
                    ee1_losses.append(scalar)
                    region_roi_sum += c_roi
                    region_bg_sum += c_bg
                    region_edge_sum += c_edge
                    region_fallback += c_fb
                    region_terms += 1

                    scalar, c_roi, c_bg, c_edge, c_fb = region_term(in_img, out_ee2, i)
                    ee2_losses.append(scalar)
                    region_roi_sum += c_roi
                    region_bg_sum += c_bg
                    region_edge_sum += c_edge
                    region_fallback += c_fb
                    region_terms += 1

                    scalar, c_roi, c_bg, c_edge, c_fb = region_term(in_img, out_ee3, i)
                    ee3_losses.append(scalar)
                    region_roi_sum += c_roi
                    region_bg_sum += c_bg
                    region_edge_sum += c_edge
                    region_fallback += c_fb
                    region_terms += 1

                    scalar, c_roi, c_bg, c_edge, c_fb = region_term(in_img, out_ee4, i)
                    ee4_losses.append(scalar)
                    region_roi_sum += c_roi
                    region_bg_sum += c_bg
                    region_edge_sum += c_edge
                    region_fallback += c_fb
                    region_terms += 1
                else:
                    rec2_losses.append(weighted_l1(in_img, output_d2, W))
                    ee1_losses.append(weighted_l1(in_img, out_ee1, W))
                    ee2_losses.append(weighted_l1(in_img, out_ee2, W))
                    ee3_losses.append(weighted_l1(in_img, out_ee3, W))
                    ee4_losses.append(weighted_l1(in_img, out_ee4, W))

                if args.use_wlpips and lpips_steps != 'final':
                    d2_outputs.append(output_d2)

            bp_t1 = time.time()

            rec1_loss = sum(losses) / args.iterations
            rec2_loss = sum(rec2_losses) / args.iterations
            ee1_loss = sum(ee1_losses) / args.iterations
            ee2_loss = sum(ee2_losses) / args.iterations
            ee3_loss = sum(ee3_losses) / args.iterations
            ee4_loss = sum(ee4_losses) / args.iterations
            loss = (rec1_loss + rec2_loss) * 0.5 + (ee1_loss + ee2_loss + ee3_loss + ee4_loss) * args.ee_loss_weight
            lpips_value = 0.0
            if args.use_wlpips:
                if lpips_steps == 'final':
                    lpips_value = weighted_lpips(
                        in_img, output_d2, lpips_W, args.lpips_weight, args.lpips_net)
                else:
                    if lpips_steps == 'last3':
                        selected = d2_outputs[-3:]
                    else:
                        selected = d2_outputs
                    lpips_value = sum(
                        weighted_lpips(in_img, o, lpips_W, args.lpips_weight, args.lpips_net)
                        for o in selected) / len(selected)
                loss = loss + lpips_value
            loss.backward()

            for net in [encoder, binarizer, decoder, unet, d2]:
                if net is not None:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)

            solver.step()
            scheduler.step()

            batch_t1 = time.time()

            if rank == 0 and train_iter % 10 == 0:
                rec1_value = rec1_loss.item()
                rec2_value = rec2_loss.item()
                ee1_value = ee1_loss.item()
                ee2_value = ee2_loss.item()
                ee3_value = ee3_loss.item()
                ee4_value = ee4_loss.item()
                loss_value = loss.item()
                lr_value = scheduler.get_last_lr()[0]

                log_scalar('train/rec1_loss', rec1_value, train_iter)
                log_scalar('train/rec2_loss', rec2_value, train_iter)
                log_scalar('train/ee1_loss', ee1_value, train_iter)
                log_scalar('train/ee2_loss', ee2_value, train_iter)
                log_scalar('train/ee3_loss', ee3_value, train_iter)
                log_scalar('train/ee4_loss', ee4_value, train_iter)
                log_scalar('train/total_loss', loss_value, train_iter)
                log_scalar('train/lr', lr_value, train_iter)
                log_scalar('train/backprop_sec', bp_t1 - bp_t0, train_iter)
                log_scalar('train/batch_sec', batch_t1 - batch_t0, train_iter)

                if args.use_wlpips:
                    log_scalar('train/lpips', float(lpips_value), train_iter)

                if is_region:
                    term_count = max(region_terms, 1)
                    log_scalar('train/loss_roi', float(region_roi_sum / term_count), train_iter)
                    log_scalar('train/loss_bg', float(region_bg_sum / term_count), train_iter)
                    log_scalar('train/loss_edge', float(region_edge_sum / term_count), train_iter)
                    if S is not None:
                        log_scalar('train/roi_density', float(S.mean()), train_iter)
                    if E is not None:
                        log_scalar('train/edge_density', float(E.mean()), train_iter)
                    if S_f is not None:
                        w_roi, w_edge, w_bg = phase_multipliers(
                            1, args.w_roi, args.w_edge, args.w_bg)
                        effective = w_roi * S_f + w_bg * (1.0 - S_f)
                        if E_f is not None:
                            effective = effective + w_edge * E_f
                        log_scalar('train/effective_weight_mean', float(effective.mean()), train_iter)
                        log_scalar('train/effective_weight_max', float(effective.max()), train_iter)
                    fallback_count = int(region_fallback)
                    if fallback_count:
                        log_scalar('train/empty_mask_fallback', float(fallback_count), train_iter)

                if writer is not None:
                    memory = get_memory_gb()
                    log_scalar('system/host_mem_available_gb',
                               memory['MemAvailable'], train_iter)
                    log_scalar('system/host_mem_used_gb',
                               memory['MemTotal'] - memory['MemAvailable'], train_iter)
                    log_scalar('system/train_rss_gb', get_rss_gb(), train_iter)
                    writer.flush()

                print(
                    '[TRAIN] Iter[{}]; LR: {}; Losses [Rec1: {:.6f}; Rec2: {:.6f}; EE1: {:.6f}; EE2: {:.6f}; EE3: {:.6f}; EE4: {:.6f}]; Backprop: {:.4f} sec; Batch: {:.4f} sec'.
                    format(train_iter, lr_value,
                           rec1_value, rec2_value,
                           ee1_value, ee2_value,
                           ee3_value, ee4_value,
                           bp_t1 - bp_t0, batch_t1 - batch_t0))

            if rank == 0 and train_iter % 100 == 0:
                print('Loss at each step:')
                print(('{:.4f} ' * args.iterations +
                       '\n').format(*[l.item() for l in losses]))

            is_eval_iter = just_resumed or train_iter % args.eval_iters == 0

            if train_iter % args.checkpoint_iters == 0:
                save(train_iter)
            elif is_eval_iter:
                save(train_iter)

            if is_eval_iter:
                set_eval(nets)

                eval_args = copy.copy(args)
                eval_args.save_codes = False
                eval_args.save_out_img = False
                eval_args.eval_msssim = 'final'
                eval_args.eval_ee_msssim = 'final'
                eval_args.eval_bbox_msssim = 'final'

                for eval_name, eval_loader in eval_loaders.items():
                    eval_begin = time.time()
                    result = run_eval(
                        nets, eval_loader, eval_args,
                        output_suffix='iter%d' % train_iter,
                        writer=writer, global_step=train_iter,
                        tag_prefix='eval/%s' % eval_name)

                    if world_size > 1:
                        reduced = {}
                        for field in result._fields:
                            reduced[field] = _all_reduce_mean(getattr(result, field))
                        result = type(result)(**reduced)

                    eval_loss = result.loss
                    mssim = result.msssim
                    psnr = result.psnr
                    psnr_ee1 = result.psnr_ee1
                    psnr_ee2 = result.psnr_ee2
                    psnr_ee3 = result.psnr_ee3
                    psnr_ee4 = result.psnr_ee4
                    msssim_ee1 = result.msssim_ee1
                    msssim_ee2 = result.msssim_ee2
                    msssim_ee3 = result.msssim_ee3
                    msssim_ee4 = result.msssim_ee4
                    psnr_roi = result.psnr_roi
                    psnr_bg = result.psnr_bg
                    psnr_edge = result.psnr_edge
                    msssim_bbox = result.msssim_bbox
                    lpips_roi = result.lpips_roi
                    lpips_bg = result.lpips_bg
                    lpips_edge = result.lpips_edge
                    lpips_full = result.lpips_full
                    mask_bitrate = result.mask_bytes
                    edge_bitrate = result.edge_bytes
                    mask_iou = result.mask_iou
                    edge_f1 = result.edge_f1
                    total_bytes = result.total_bytes_per_frame
                    vmaf_mean = result.vmaf
                    lpips_mean = result.lpips
                    prefix_raw = result.prefix_visual_bytes_raw
                    prefix_visual = result.prefix_visual_bytes
                    prefix_total = result.prefix_total_bytes

                    if rank == 0:
                        print('Evaluation @iter %d done in %d secs' % (
                            train_iter, time.time() - eval_begin))
                        print('%s Loss    : ' % eval_name
                              + '\t'.join(['%.5f' % el for el in eval_loss.tolist()]))
                        print('%s MS-SSIM : ' % eval_name
                              + '\t'.join(['%.5f' % el for el in mssim.tolist()]))
                        print('%s PSNR    : ' % eval_name
                              + '\t'.join(['%.5f' % el for el in psnr.tolist()]))
                        print('%s EE1 PSNR: ' % eval_name
                              + '\t'.join(['%.5f' % el for el in psnr_ee1.tolist()]))
                        print('%s EE2 PSNR: ' % eval_name
                              + '\t'.join(['%.5f' % el for el in psnr_ee2.tolist()]))
                        print('%s EE3 PSNR: ' % eval_name
                              + '\t'.join(['%.5f' % el for el in psnr_ee3.tolist()]))
                        print('%s EE4 PSNR: ' % eval_name
                              + '\t'.join(['%.5f' % el for el in psnr_ee4.tolist()]))
                        if psnr_roi is not None:
                            print('%s ROI PSNR: ' % eval_name
                                  + '\t'.join(['%.5f' % el for el in psnr_roi.tolist()]))
                        if psnr_bg is not None:
                            print('%s BG PSNR : ' % eval_name
                                  + '\t'.join(['%.5f' % el for el in psnr_bg.tolist()]))
                        if psnr_edge is not None:
                            print('%s Edge PSNR: ' % eval_name
                                  + '\t'.join(['%.5f' % el for el in psnr_edge.tolist()]))
                        if lpips_full is not None:
                            print('%s LPIPS full: ' % eval_name
                                  + '\t'.join(['%.5f' % el
                                               for el in np.asarray(lpips_full).tolist()]))
                        if lpips_roi is not None:
                            print('%s LPIPS ROI : ' % eval_name
                                  + '\t'.join(['%.5f' % el
                                               for el in np.asarray(lpips_roi).tolist()]))
                        if mask_bitrate is not None:
                            print('%s Mask bytes/frame: %.2f' % (eval_name, mask_bitrate))
                        if edge_bitrate is not None:
                            print('%s Edge bytes/frame: %.2f' % (eval_name, edge_bitrate))
                        if total_bytes is not None:
                            print('%s Total bytes/frame: %.2f' % (eval_name, total_bytes))
                        if vmaf_mean is not None:
                            print('%s VMAF : %.4f' % (eval_name, vmaf_mean))
                        if lpips_mean is not None:
                            print('%s LPIPS: %.4f' % (eval_name, lpips_mean))

                        for step_idx in range(len(eval_loss)):
                            step = step_idx + 1
                            log_scalar('eval/%s/loss_step_%02d' % (eval_name, step),
                                       float(eval_loss[step_idx]), train_iter)
                            if np.isfinite(mssim[step_idx]):
                                log_scalar('eval/%s/msssim_step_%02d' % (eval_name, step),
                                float(mssim[step_idx]), train_iter)
                            log_scalar('eval/%s/psnr_step_%02d' % (eval_name, step),
                                       float(psnr[step_idx]), train_iter)
                            log_scalar('eval/%s/psnr_ee1_step_%02d' % (eval_name, step),
                                       float(psnr_ee1[step_idx]), train_iter)
                            log_scalar('eval/%s/psnr_ee2_step_%02d' % (eval_name, step),
                                       float(psnr_ee2[step_idx]), train_iter)
                            log_scalar('eval/%s/psnr_ee3_step_%02d' % (eval_name, step),
                                       float(psnr_ee3[step_idx]), train_iter)
                            log_scalar('eval/%s/psnr_ee4_step_%02d' % (eval_name, step),
                                       float(psnr_ee4[step_idx]), train_iter)
                            if psnr_roi is not None:
                                log_scalar('eval/%s/psnr_roi_step_%02d' % (eval_name, step),
                                           float(psnr_roi[step_idx]), train_iter)
                            if psnr_bg is not None:
                                log_scalar('eval/%s/psnr_bg_step_%02d' % (eval_name, step),
                                           float(psnr_bg[step_idx]), train_iter)
                            if msssim_bbox is not None and np.isfinite(msssim_bbox[step_idx]):
                                log_scalar('eval/%s/msssim_bbox_step_%02d' % (eval_name, step),
                                           float(msssim_bbox[step_idx]), train_iter)
                            if lpips_roi is not None:
                                log_scalar('eval/%s/lpips_roi_step_%02d' % (eval_name, step),
                                           float(lpips_roi[step_idx]), train_iter)
                            if psnr_edge is not None:
                                log_scalar('eval/%s/psnr_edge_step_%02d' % (eval_name, step),
                                           float(psnr_edge[step_idx]), train_iter)
                            for ee_idx, msssim_ee in enumerate(
                                    (msssim_ee1, msssim_ee2, msssim_ee3, msssim_ee4), 1):
                                if msssim_ee is not None and np.isfinite(msssim_ee[step_idx]):
                                    log_scalar('eval/%s/msssim_ee%d_step_%02d'
                                               % (eval_name, ee_idx, step),
                                               float(msssim_ee[step_idx]), train_iter)
                            if lpips_full is not None:
                                log_scalar('eval/%s/lpips_full_step_%02d' % (eval_name, step),
                                           float(lpips_full[step_idx]), train_iter)
                            if lpips_bg is not None and np.isfinite(lpips_bg[step_idx]):
                                log_scalar('eval/%s/lpips_bg_step_%02d' % (eval_name, step),
                                           float(lpips_bg[step_idx]), train_iter)
                            if lpips_edge is not None and np.isfinite(lpips_edge[step_idx]):
                                log_scalar('eval/%s/lpips_edge_step_%02d' % (eval_name, step),
                                           float(lpips_edge[step_idx]), train_iter)

                        log_scalar('eval/%s/loss_mean' % eval_name,
                                   float(eval_loss.mean()), train_iter)
                        mssim_mean = (float(np.nanmean(mssim))
                                     if np.isfinite(mssim).any() else float('nan'))
                        log_scalar('eval/%s/msssim_mean' % eval_name,
                            mssim_mean, train_iter)
                        log_scalar('eval/%s/psnr_mean' % eval_name,
                                   float(psnr.mean()), train_iter)
                        if mask_bitrate is not None:
                            log_scalar('eval/%s/mask_bytes_per_frame' % eval_name,
                                       mask_bitrate, train_iter)
                        if edge_bitrate is not None:
                            log_scalar('eval/%s/edge_bytes_per_frame' % eval_name,
                                       edge_bitrate, train_iter)
                        if mask_iou is not None:
                            log_scalar('eval/%s/mask_iou' % eval_name, mask_iou, train_iter)
                        if edge_f1 is not None:
                            log_scalar('eval/%s/edge_f1' % eval_name, edge_f1, train_iter)
                        if total_bytes is not None:
                            log_scalar('eval/%s/total_bytes_per_frame' % eval_name,
                                       total_bytes, train_iter)
                        if vmaf_mean is not None:
                            log_scalar('eval/%s/vmaf' % eval_name,
                                       vmaf_mean, train_iter)
                        if lpips_mean is not None:
                            log_scalar('eval/%s/lpips' % eval_name,
                                       lpips_mean, train_iter)
                        if prefix_raw is not None:
                            for pidx in range(len(prefix_raw)):
                                log_scalar('eval/%s/prefix_raw_bytes_l%02d'
                                           % (eval_name, pidx + 1),
                                           float(prefix_raw[pidx]), train_iter)
                        if prefix_visual is not None:
                            for pidx in range(len(prefix_visual)):
                                log_scalar('eval/%s/prefix_visual_bytes_l%02d'
                                           % (eval_name, pidx + 1),
                                           float(prefix_visual[pidx]), train_iter)
                        if prefix_total is not None:
                            for pidx in range(len(prefix_total)):
                                log_scalar('eval/%s/prefix_bytes_l%02d' % (eval_name, pidx + 1),
                                           float(prefix_total[pidx]), train_iter)

                        print_hierarchy_summary(result, args.phase_schedule, eval_name)

                    if args.best_metric == 'mean':
                        eval_psnr = float(psnr.mean())
                        best_step = -1
                    elif args.best_metric == 'max':
                        best_step = int(np.argmax(psnr))
                        eval_psnr = float(psnr[best_step])
                    elif args.best_metric == 'psnr+msssim':
                        best_step = -1
                        msssim_score = (float(np.nanmean(mssim))
                                        if np.isfinite(mssim).any() else 0.0)
                        eval_psnr = (float(psnr.mean()) / 50.0
                                     + args.best_metric_w * msssim_score)
                    else:
                        best_step = len(psnr) - 1
                        eval_psnr = float(psnr[best_step])

                    if eval_psnr > best_psnr:
                        best_psnr = eval_psnr
                        best_iter = train_iter
                        patience = 0
                        if rank == 0:
                            save_best()
                            print('New best score %.5f at iter %d (step %s)' % (
                                best_psnr, best_iter, best_step))
                    else:
                        patience += 1

                    if rank == 0:
                        log_scalar('eval/%s/best_psnr' % eval_name, best_psnr, train_iter)
                        log_scalar('eval/%s/best_step' % eval_name, best_step, train_iter)

                    if args.early_stop > 0 and patience >= args.early_stop:
                        if rank == 0:
                            print('Early stopping after %d evaluations without improvement.'
                                  % patience)
                        stop_training = True
                        break

                    if writer is not None:
                        writer.flush()

                set_train(nets)
                just_resumed = False
                if stop_training:
                    break

        epoch += 1

        if stop_training or train_iter > args.max_train_iters:
            if rank == 0:
                print('Training done.')
            if writer is not None:
                writer.close()
            break

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
