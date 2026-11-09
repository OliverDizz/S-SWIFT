import os
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
from evaluate import run_eval
from train_options import parser
from util import get_models, init_lstm, set_train, set_eval, init_d2
from util import prepare_inputs, forward_ctx
from semantic_loss import build_weight_map, weighted_l1, weighted_lpips

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


def main():
    torch.backends.cudnn.benchmark = True

    args = parser.parse_args()

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

    if rank == 0:
        print(args)

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
        use_conditioning=args.use_conditioning).cuda()

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

    solver = optim.Adam(params, lr=args.lr)

    milestones = [int(s) for s in args.schedule.split(',')]
    scheduler = LS.MultiStepLR(solver, milestones=milestones, gamma=args.gamma)

    if rank == 0 and not os.path.exists(args.model_dir):
        print("Creating directory %s." % args.model_dir)
        os.makedirs(args.model_dir, exist_ok=True)

    names = ['encoder', 'binarizer', 'decoder', 'd2', 'unet']

    def unwrap(net):
        if isinstance(net, (nn.DataParallel, DDP)):
            return net.module
        return net

    def resume(model_name, index):
        for net_idx, net in enumerate(nets):
            if net is None:
                continue
            name = names[net_idx]
            checkpoint_path = '{}/{}_{}_{:08d}.pth'.format(
                args.model_dir, model_name, name, index)
            if rank == 0:
                print('Loading %s from %s...' % (name, checkpoint_path))
            unwrap(net).load_state_dict(
                torch.load(checkpoint_path, map_location='cuda:%d' % local_rank))

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
    if args.load_model_name:
        if rank == 0:
            print('Loading %s@iter %d' % (args.load_model_name, args.load_iter))
        resume(args.load_model_name, args.load_iter)
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

            W = None
            if S is not None or E is not None:
                W = build_weight_map(
                    S, E, args.alpha_sem, args.alpha_edge)

            losses = []
            rec2_losses = []
            ee1_losses, ee2_losses, ee3_losses, ee4_losses = [], [], [], []

            bp_t0 = time.time()
            batch_size, _, height, width = res.size()

            out_img = torch.zeros(1, 3, height, width).cuda() + 0.5

            b, d, h, w = batch_size, args.bits, height // 16, width // 16
            code_arr = [torch.zeros(b, d, h, w).cuda() for i in range(args.iterations)]

            for i in range(args.iterations):
                if args.v_compress and args.stack:
                    encoder_input = torch.cat([frame1, res, frame2], dim=1)
                else:
                    encoder_input = res

                encoded, encoder_h_1, encoder_h_2, encoder_h_3 = encoder(
                    encoder_input, encoder_h_1, encoder_h_2, encoder_h_3,
                    warped_unet_output1, warped_unet_output2, sem)

                codes = binarizer(encoded)
                code_arr[i] = codes

                (output, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4) = decoder(
                    codes, decoder_h_1, decoder_h_2, decoder_h_3, decoder_h_4,
                    warped_unet_output1, warped_unet_output2, sem)

                res = res - output
                out_img = out_img + output.data
                losses.append(weighted_l1(res, 0.0, W))

                (d2_h_1, d2_h_2, d2_h_3, d2_h_4) = init_d2(
                    batch_size=(crops[0].size(0) * args.num_crops),
                    height=crops[0].size(2), width=crops[0].size(3), args=args)

                codes_d2 = torch.stack(code_arr, dim=1).reshape(b, -1, h, w)
                (output_d2, out_ee1, out_ee2, out_ee3, out_ee4,
                 d2_h_1, d2_h_2, d2_h_3, d2_h_4) = d2(
                     codes_d2, d2_h_1, d2_h_2, d2_h_3, d2_h_4,
                     warped_unet_output1, warped_unet_output2, sem)

                rec2_losses.append(weighted_l1(in_img, output_d2, W))
                ee1_losses.append(weighted_l1(in_img, out_ee1, W))
                ee2_losses.append(weighted_l1(in_img, out_ee2, W))
                ee3_losses.append(weighted_l1(in_img, out_ee3, W))
                ee4_losses.append(weighted_l1(in_img, out_ee4, W))

            bp_t1 = time.time()

            rec1_loss = sum(losses) / args.iterations
            rec2_loss = sum(rec2_losses) / args.iterations
            ee1_loss = sum(ee1_losses) / args.iterations
            ee2_loss = sum(ee2_losses) / args.iterations
            ee3_loss = sum(ee3_losses) / args.iterations
            ee4_loss = sum(ee4_losses) / args.iterations
            loss = (rec1_loss + rec2_loss) * 0.5 + (ee1_loss + ee2_loss + ee3_loss + ee4_loss) * 0.25
            if args.use_wlpips:
                loss = loss + weighted_lpips(
                    in_img + 0.5, output_d2 + 0.5, W, args.lpips_weight)
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

                eval_args = args if rank == 0 else copy.copy(args)
                eval_args.save_codes = False
                eval_args.save_out_img = False

                for eval_name, eval_loader in eval_loaders.items():
                    eval_begin = time.time()
                    (eval_loss, mssim, psnr,
                     psnr_ee1, psnr_ee2, psnr_ee3, psnr_ee4,
                     mask_bitrate, edge_bitrate, vmaf_mean, lpips_mean) = run_eval(
                        nets, eval_loader, eval_args,
                        output_suffix='iter%d' % train_iter,
                        writer=writer, global_step=train_iter,
                        tag_prefix='eval/%s' % eval_name)

                    if world_size > 1:
                        eval_loss = _all_reduce_mean(eval_loss)
                        mssim = _all_reduce_mean(mssim)
                        psnr = _all_reduce_mean(psnr)
                        psnr_ee1 = _all_reduce_mean(psnr_ee1)
                        psnr_ee2 = _all_reduce_mean(psnr_ee2)
                        psnr_ee3 = _all_reduce_mean(psnr_ee3)
                        psnr_ee4 = _all_reduce_mean(psnr_ee4)
                        mask_bitrate = _all_reduce_mean(mask_bitrate)
                        edge_bitrate = _all_reduce_mean(edge_bitrate)

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
                        if mask_bitrate is not None:
                            print('%s Mask bytes/frame: %.2f' % (eval_name, mask_bitrate))
                        if edge_bitrate is not None:
                            print('%s Edge bytes/frame: %.2f' % (eval_name, edge_bitrate))
                        if vmaf_mean is not None:
                            print('%s VMAF : %.4f' % (eval_name, vmaf_mean))
                        if lpips_mean is not None:
                            print('%s LPIPS: %.4f' % (eval_name, lpips_mean))

                        for step_idx in range(len(eval_loss)):
                            step = step_idx + 1
                            log_scalar('eval/%s/loss_step_%02d' % (eval_name, step),
                                       float(eval_loss[step_idx]), train_iter)
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

                        log_scalar('eval/%s/loss_mean' % eval_name,
                                   float(eval_loss.mean()), train_iter)
                        log_scalar('eval/%s/msssim_mean' % eval_name,
                                   float(mssim.mean()), train_iter)
                        log_scalar('eval/%s/psnr_mean' % eval_name,
                                   float(psnr.mean()), train_iter)
                        if mask_bitrate is not None:
                            log_scalar('eval/%s/mask_bytes_per_frame' % eval_name,
                                       mask_bitrate, train_iter)
                        if edge_bitrate is not None:
                            log_scalar('eval/%s/edge_bytes_per_frame' % eval_name,
                                       edge_bitrate, train_iter)
                        if vmaf_mean is not None:
                            log_scalar('eval/%s/vmaf' % eval_name,
                                       vmaf_mean, train_iter)
                        if lpips_mean is not None:
                            log_scalar('eval/%s/lpips' % eval_name,
                                       lpips_mean, train_iter)

                    eval_psnr = float(psnr[-1])
                    if eval_psnr > best_psnr:
                        best_psnr = eval_psnr
                        best_iter = train_iter
                        patience = 0
                        if rank == 0:
                            save_best()
                            print('New best PSNR %.5f at iter %d' % (best_psnr, best_iter))
                    else:
                        patience += 1

                    if rank == 0:
                        log_scalar('eval/%s/best_psnr' % eval_name, best_psnr, train_iter)

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
