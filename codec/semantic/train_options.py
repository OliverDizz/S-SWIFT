"""Training options."""

import argparse


def _int_list(value):
    return [int(x) for x in value.split(',') if x.strip() != '']


def _float_list(value):
    return [float(x) for x in value.split(',') if x.strip() != '']


def validate_semantic_args(args):
    """Validate cross-flag constraints.

    The phase schedule must have exactly ``--iterations`` entries so a changed
    ``--iterations`` cannot silently misalign the phases and gates. The per-phase
    weight lists must cover every phase.
    """
    schedule = args.phase_schedule
    if isinstance(schedule, str):
        schedule = _int_list(schedule)
        args.phase_schedule = schedule
    assert len(schedule) == args.iterations, (
        'len(--phase-schedule) = %d != --iterations = %d'
        % (len(schedule), args.iterations))
    num_phases = max(schedule) + 1
    for name in ('w_roi', 'w_edge', 'w_bg'):
        values = getattr(args, name)
        if isinstance(values, str):
            values = _float_list(values)
            setattr(args, name, values)
        assert len(values) >= num_phases, (
            'len(--%s) = %d < num_phases = %d' % (name, len(values), num_phases))
    return args


parser = argparse.ArgumentParser()

######## Data ########
parser.add_argument('--train', required=True, type=str,
                    help='Path to training data.')
parser.add_argument('--eval', required=True, type=str,
                    help='Path to eval data.')
# (distance1, distance2) should be (1, 2), (3, 3), or (6, 6).
# They correspond to the 3 levels of hierarchy described in paper.
parser.add_argument('--distance1', type=int,
                    help='Distance to left interpolation source.')
parser.add_argument('--distance2', type=int,
                    help='Distance to right interpolation source.')
parser.add_argument('--train-mv', type=str,
                    help='Path to motion vectors of training set.')
parser.add_argument('--eval-mv', type=str,
                    help='Path to motion vectors of evaluation set.')
parser.add_argument('--train-mask', type=str,
                    help='Path to binary SAM3 masks of the training set.')
parser.add_argument('--eval-mask', type=str,
                    help='Path to binary SAM3 masks of the evaluation set.')
parser.add_argument('--train-edge', type=str,
                    help='Path to binary Canny edges of the training set.')
parser.add_argument('--eval-edge', type=str,
                    help='Path to binary Canny edges of the evaluation set.')

######## Semantic weighting ########
parser.add_argument('--alpha-sem', type=float, default=1.0,
                    help='Weight of the semantic mask channel in W.')
parser.add_argument('--alpha-edge', type=float, default=0.5,
                    help='Weight of the edge channel in W.')
parser.add_argument('--use-wlpips', action='store_true',
                    help='Add a mask-weighted LPIPS term to the visual loss.')
parser.add_argument('--lpips-weight', type=float, default=0.1,
                    help='Weight of the (weighted) LPIPS term.')

######## Region loss (Design A) ########
parser.add_argument('--loss-mode', type=str, default='legacy',
                    choices=['legacy', 'region'],
                    help='legacy = current W path; region = explicit ROI/bg/edge loss.')
parser.add_argument('--roi-weight', type=float, default=2.0,
                    help='Weight of the ROI term in region mode.')
parser.add_argument('--bg-weight', type=float, default=0.5,
                    help='Weight of the background term in region mode.')
parser.add_argument('--edge-weight', type=float, default=0.5,
                    help='Weight of the edge term in region mode.')
parser.add_argument('--boundary-band', type=int, default=0,
                    help='Pixels; 0 = hard S, >0 = boost a dilated boundary band.')
parser.add_argument('--boundary-weight', type=float, default=0.5,
                    help='Extra weight for the boundary band.')
parser.add_argument('--roi-ramp', type=str, default='none',
                    choices=['none', 'linear'],
                    help='none or linear; ramp ROI emphasis across progressive steps.')
parser.add_argument('--roi-ramp-start', type=float, default=0.5,
                    help='ROI weight multiplier at step 1 when roi-ramp=linear.')

######## Semantic hierarchy / residual gating ########
parser.add_argument('--phase-schedule', type=_int_list,
                    default=[0, 1, 1, 1, 2, 2, 3, 3, 3, 3],
                    help='Comma-separated phase per progressive step '
                         '(0=coarse, 1=ROI, 2=edge, 3=background).')
parser.add_argument('--gate-mode', type=str, default='mask',
                    choices=['none', 'mask'],
                    help='mask = gate the encoder residual by the phase mask; '
                         'none = no gating.')
parser.add_argument('--gate-feather', type=int, default=2,
                    help='Dilate+blur pixels applied to S and E (0 = hard gate).')
parser.add_argument('--w-roi', type=_float_list,
                    default=[1.0, 4.0, 1.0, 0.5],
                    help='Per-phase ROI weight list.')
parser.add_argument('--w-edge', type=_float_list,
                    default=[1.0, 0.25, 2.0, 0.5],
                    help='Per-phase edge weight list.')
parser.add_argument('--w-bg', type=_float_list,
                    default=[1.0, 0.25, 0.5, 1.0],
                    help='Per-phase background weight list.')
parser.add_argument('--lpips-scope', type=str, default='full',
                    choices=['full', 'roi'],
                    help='full or roi; region LPIPS weighting scope.')
parser.add_argument('--lpips-steps', type=str, default='all',
                    choices=['final', 'last3', 'all'],
                    help='final, last3, or all; progressive steps for LPIPS.')
parser.add_argument('--lpips-net', type=str, default='alex',
                    choices=['alex', 'vgg'],
                    help='Official lpips backbone: alex or vgg.')
parser.add_argument('--region-norm', type=str, default='mean1',
                    choices=['sum', 'mean1'],
                    help='mean1 = mean-1 weighted mean; sum = legacy region sum.')
parser.add_argument('--region-partition', type=str, default='overlap',
                    choices=['overlap', 'disjoint'],
                    help='overlap = w_roi*S + w_edge*E + w_bg*(1-S); '
                         'disjoint = object/edge/background-other split.')
parser.add_argument('--spatial-bit-alloc', action='store_true',
                    help='Mask the binarized code map by the phase ROI/edge '
                         'block mask so masked positions transmit no bits.')
parser.add_argument('--ee-loss-weight', type=float, default=0.25,
                    help='Weight of the combined early-exit loss term '
                         '(reference codec uses 0.25).')
parser.add_argument('--ee-head', type=str, default='src',
                    choices=['src', 'refined'],
                    help='src = reference PixelShuffle+1x1 early-exit head; '
                         'refined = src plus a zero-initialized residual '
                         'refinement branch (B/C/D).')

######## Semantic conditioning ########
parser.add_argument('--use-conditioning', action='store_true',
                    help='Enable semantic conditioning (encoder MaskAdd + decoder FiLM).')
parser.add_argument('--conditioning-mode', type=str, default='legacy',
                    choices=['legacy', 'spatial'],
                    help='legacy = pooled FiLM; spatial = spatially varying FiLM.')
parser.add_argument('--conditioning-init', type=str, default='random',
                    choices=['random', 'zero'],
                    help='random or zero; zero initializes new conditioning heads.')
parser.add_argument('--load-strict', type=str, default='true',
                    choices=['true', 'false'],
                    help='false loads shared weights and ignores new keys (warm start).')
parser.add_argument('--load-model-dir', type=str, default=None,
                    help='Optional source model directory for warm starts.')

######## Semantic metrics ########
parser.add_argument('--use-vmaf', action='store_true',
                    help='Compute VMAF during evaluation when available.')
parser.add_argument('--use-lpips', action='store_true',
                    help='Compute LPIPS during evaluation when available.')
parser.add_argument('--use-vgg-l2', action='store_true',
                    help='Compute the diagnostic VGG-L2 distance during eval.')

######## Model ########
parser.add_argument('--v-compress', action='store_true',
                    help='True: video compression model. False: image compression.')
parser.add_argument('--iterations', type=int, default=10, 
                    help='# iterations of progressive encoding/decoding.')
parser.add_argument('--bits', default=16, type=int, 
                    help='Bottle neck size.')
parser.add_argument('--patch', default=64, type=int, 
                    help='Patch size.')
parser.add_argument('--shrink', type=int, default=2, 
                    help='Reducing # channels in U-net by this factor.')

# More model variants for ablation study. Please see paper for details.
parser.add_argument('--warp', action='store_true',
                    help='Whether to use motion information to warp U-net features.')
parser.add_argument('--fuse-encoder', action='store_true',
                    help='Whether to fuse context features into encoder.')
parser.add_argument('--encoder-fuse-level', type=int,
                    help='# encoder layers to fuse context information into.')
parser.add_argument('--decoder-fuse-level', type=int,
                    help='# decoder layers to fuse context information into.')
parser.add_argument('--stack', action='store_true', 
                    help='Whether to stack context frames as encoder input.')

######## Learning ########
parser.add_argument('--max-train-iters', type=int, default=100000,
                    help='Max training iterations.')
parser.add_argument('--lr', type=float, default=0.00025,
                    help='Learning rate.')
parser.add_argument('--clip', type=float, default=0.5,
                    help='Gradient clipping.')
# parser.add_argument('--schedule', default='15000,40000,100000,250000', type=str,
#                     help='Schedule milestones.')
parser.add_argument('--schedule', default=None, type=str,
                    help='Comma-separated LR milestone iterations. If unset, '
                         'derived from --schedule-frac x --max-train-iters.')
parser.add_argument('--schedule-frac', type=_float_list,
                    default=[0.5, 0.6, 0.7, 0.8, 0.9],
                    help='Fractions of --max-train-iters for the LR milestones '
                         'when --schedule is unset (matches the reference '
                         'codec 10000,12000,14000,16000,18000 at 20k).')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--lr-schedule', type=str, default='multistep',
                    choices=['multistep', 'cosine'],
                    help='multistep = step decay; cosine = cosine annealing.')
parser.add_argument('--lr-warmup-iters', type=int, default=0,
                    help='Linear LR warmup steps at the start of training (0 = off).')
parser.add_argument('--lr-eta-min', type=float, default=1e-6,
                    help='Final LR for the cosine schedule.')
parser.add_argument('--warm-start-lr', type=float, default=None,
                    help='Base LR override for --load-model-dir warm starts.')
parser.add_argument('--batch-size', type=int, default=16, 
                    help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1,
                    help='Batch size for evaluation.')
parser.add_argument('--eval-stride', type=int, default=1,
                    help='Evaluate every Nth test sample (1 = full set).')
parser.add_argument('--eval-fp16', action='store_true',
                    help='Use FP16 autocast during evaluation inference.')
parser.add_argument('--eval-workers', type=int, default=0,
                    help='Threads for the CPU MS-SSIM eval pass (0 = auto).')
parser.add_argument('--eval-msssim', type=str, default='all',
                    choices=['all', 'final', 'off'],
                    help='Main-output MS-SSIM steps during eval '
                         '(all = every step, final = last step, off = none).')
parser.add_argument('--eval-ee-msssim', type=str, default='all',
                    choices=['all', 'final', 'off'],
                    help='Early-exit MS-SSIM steps during eval '
                         '(all = every step, final = last step, off = none).')
parser.add_argument('--eval-bbox-msssim', type=str, default='all',
                    choices=['all', 'final', 'off'],
                    help='Bbox MS-SSIM steps during eval '
                         '(all = every step, final = last step, off = none).')

# To save computation, we compute objective for multiple
# crops for each forward pass.
parser.add_argument('--num-crops', type=int, default=2,
                    help='# training crops per example.')
parser.add_argument('--gpus', default='0', type=str,
                    help='GPU indices separated by comma, e.g. \"0,1\".')
parser.add_argument('--num-workers', type=int, default=2,
                    help='Number of DataLoader worker processes.')
parser.add_argument('--early-stop', type=int, default=0,
                    help='Stop after this many evaluations without PSNR improvement. 0 disables it.')
parser.add_argument('--best-metric', type=str, default='last',
                    choices=['last', 'mean', 'max', 'psnr+msssim'],
                    help='PSNR criterion for best checkpoint: last, mean, max '
                         'over steps, or psnr+msssim combined score.')
parser.add_argument('--best-metric-w', type=float, default=1.0,
                    help='Relative weight of msssim in the psnr+msssim score '
                         '(score = psnr_mean/50 + best-metric-w * msssim_mean).')
parser.add_argument('--mp-context', type=str, default='fork',
                    help='DataLoader multiprocessing context: fork or spawn.')

######## Reproducibility ########
parser.add_argument('--seed', type=int, default=0,
                    help='Base random seed. DDP ranks use seed + rank.')
parser.add_argument('--deterministic', action='store_true',
                    help='Disable cuDNN autotuning and enable deterministic '
                         'kernels. Exact GPU reproducibility is approximate '
                         'without this flag.')

######## Experiment ########
parser.add_argument('--out-dir', type=str, default='output',
                    help='Output directory (for compressed codes & output images).')
parser.add_argument('--model-dir', type=str, default='model',
                    help='Path to model folder.')
parser.add_argument('--load-model-name', type=str,
                    help='Checkpoint name to load. (Do nothing if not specified.)')
parser.add_argument('--load-iter', type=int,
                    help='Iteraction of checkpoint to load.')
parser.add_argument('--save-model-name', type=str, default='demo',
                    help='Checkpoint name to save.')
parser.add_argument('--save-codes', action='store_true',
                    help='If true, write compressed codes during eval.')
parser.add_argument('--save-out-img', action='store_true',
                    help='If true, save output images during eval.')
parser.add_argument('--checkpoint-iters', type=int, default=10000,
                    help='Model checkpoint period.')
parser.add_argument('--eval-iters', type=int, default=4500,
                    help='Evaluation period.')
parser.add_argument('--tensorboard-dir', type=str, default=None,
                    help='If set, write TensorBoard events to this directory.')
