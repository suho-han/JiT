import argparse
import ast
import copy
import datetime
import logging
import os
import random
import time
from pathlib import Path

import autorootcwd
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from natsort import natsorted
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import losses.losses as custom_losses
import util.misc as misc
from denoiser import Denoiser
from engine_jit import train_one_epoch, validation
from util.crop import center_crop_arr
from util.isicdataset import ISICSegmentationDataset, get_isic_transform
from util.monudataset import MoNuSegmentationDataset, get_monu_transform
from util.octadataset import OCTASegmentationDataset, get_octa_transform


def get_args_parser():
    parser = argparse.ArgumentParser('JiT', add_help=False)

    # architecture
    parser.add_argument('--model', default='JiT-B/16', type=str, metavar='MODEL',
                        help='Name of the model to train')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--attn_dropout', type=float, default=0.0,
                        help='Attention dropout rate')
    parser.add_argument('--proj_dropout', type=float, default=0.0,
                        help='Projection dropout rate')
    parser.add_argument('--img_channel', type=int, default=3,
                        help='Number of image channels')
    parser.add_argument('--mask_channel', type=int, default=1,
                        help='Number of mask channels')
    # "{'cond': 'fixed', 'low_cond': 'fixed', 'high_cond': 'fixed'}"
    # fixed(1.0), shared(all blocks share the same weight),
    # learnable(all blocks have their own weight initialized to 1), zero_init(all blocks have their own weight initialized to 0)
    parser.add_argument('--cond_weight', type=str, default=None,
                        help='Weight configs for cond, low_cond, high_cond')

    # training
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--blr', type=float, default=5e-5, metavar='LR',
                        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (default: 0.0)')
    parser.add_argument('--ema_decay1', type=float, default=0.9999,
                        help='The first ema to track. Use the first ema for sampling by default.')
    parser.add_argument('--ema_decay2', type=float, default=0.9996,
                        help='The second ema to track')
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=5e-2, type=float)
    parser.add_argument('--label_drop_prob', default=0.1, type=float)
    parser.add_argument('--add_loss', action='store_true', default=False)
    parser.add_argument('--add_loss_name', type=str, default='dice_bce')
    parser.add_argument('--add_loss_weight', type=float, default=1.0)

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # sampling
    parser.add_argument('--sampling_method', default='heun', type=str,
                        help='ODE samping method')
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg', default=1.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval_min', default=0.0, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval_max', default=1.0, type=float,
                        help='CFG interval max')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency (in epochs) for evaluation')
    parser.add_argument('--online_eval', action='store_true')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Threshold for binary segmentation metrics')

    # dataset
    parser.add_argument('--dataset', default='OCTA500_6M', type=str,
                        help='Dataset name')
    parser.add_argument('--data_path', default='data/OCTA500_6M', type=str,
                        help='Path to the dataset')

    # checkpointing
    parser.add_argument('--output_dir', default='outputs',
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--resume', default='',
                        help='Folder that contains checkpoint to resume from')
    parser.add_argument('--save_last_freq', type=int, default=10,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser


def main(args):
    if isinstance(args.cond_weight, str):
        try:
            args.cond_weight = ast.literal_eval(args.cond_weight)
        except Exception:
            pass

    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    cond_weight_str = ""
    if hasattr(args, 'cond_weight') and args.cond_weight is not None:
        try:
            c = args.cond_weight.get('cond', 'fixed')[0]
            l = args.cond_weight.get('low_cond', 'fixed')[0]
            h = args.cond_weight.get('high_cond', 'fixed')[0]
            cond_weight_str = f"-{c}{l}{h}"
        except Exception:
            pass

    # Set up output directory and logging (only on main process)
    args.output_dir = os.path.join(args.output_dir, f"{args.model.replace('/', '-')}{cond_weight_str}-{args.dataset}{f'-{args.add_loss_name}' if args.add_loss else ''}")
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    tensorboard_dir = os.path.join(args.output_dir, 'tensorboard')
    if misc.is_main_process():
        os.makedirs(checkpoint_dir, exist_ok=True)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=tensorboard_dir)
    else:
        log_writer = None

    logger = logging.getLogger("jit_train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if misc.is_main_process() and not logger.handlers:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"training_{timestamp}.log")
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
    elif not logger.handlers:
        logger.addHandler(logging.NullHandler())

    logger.info('Job directory: %s', os.path.dirname(os.path.realpath(__file__)))
    logger.info("Arguments:\n%s", str(args).replace(', ', ',\n'))

    # Data augmentation transforms

    # Load dataset - support both ImageFolder and OCTA segmentation formats
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')

    if 'OCTA' in args.dataset:
        # OCTA segmentation dataset format
        transform_train, transform_val, _ = get_octa_transform(image_size=args.img_size)
        dataset_train = OCTASegmentationDataset(
            train_path,
            img_size=args.img_size,
            transform=transform_train,
        )
        dataset_val = OCTASegmentationDataset(
            val_path,
            img_size=args.img_size,
            transform=transform_val,
        )
    elif 'MoNuSeg' == args.dataset:
        transform_train,  _ = get_monu_transform(image_size=args.img_size)
        dataset_train = MoNuSegmentationDataset(
            train_path,
            img_size=args.img_size,
            transform=transform_train,
        )
    elif 'ISIC2016' == args.dataset:
        transform_train, _,  _ = get_isic_transform(image_size=args.img_size)
        dataset_train = ISICSegmentationDataset(
            train_path,
            img_size=args.img_size,
            transform=transform_train,
        )
    elif 'ISIC2018' == args.dataset:
        transform_train, transform_val,  _ = get_isic_transform(image_size=args.img_size)
        dataset_train = ISICSegmentationDataset(
            train_path,
            img_size=args.img_size,
            transform=transform_train,
        )
        dataset_val = ISICSegmentationDataset(
            val_path,
            img_size=args.img_size,
            transform=transform_val,
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not supported.")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    logger.info("Sampler_train = %s", sampler_train)

    if args.online_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        logger.info("Sampler_val = %s", sampler_val)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False

    # Create denoiser
    model = Denoiser(args)

    logger.info("Model = %s", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Number of trainable parameters: %.6fM", n_params / 1e6)

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    if args.lr is None:  # only base_lr (blr) is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("Base lr: %.2e", args.lr * 256 / eff_batch_size)
    logger.info("Actual lr: %.2e", args.lr)
    logger.info("Effective batch size: %d", eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # Set up optimizer with weight decay adjustment for bias and norm layers
    param_groups = misc.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    logger.info("%s", optimizer)

    # Resume from checkpoint if provided
    checkpoint = None
    ckpt_to_load = None

    if args.resume == "scratch":
        logger.info("Training from scratch explicitly requested.")
    else:
        # Load specified resume file or fallback to last checkpoint
        ckpt_to_load = os.path.join(checkpoint_dir, args.resume) if args.resume else os.path.join(checkpoint_dir, "checkpoint-last.pth")

        if os.path.exists(ckpt_to_load):
            try:
                checkpoint = torch.load(ckpt_to_load, map_location='cpu', weights_only=False)
                logger.info("Successfully loaded checkpoint from %s", ckpt_to_load)
            except Exception as exc:
                logger.warning("Failed to load checkpoint from %s: %s", ckpt_to_load, exc)

        if checkpoint is None and not args.resume:
            logger.info("Auto-resume enabled but checkpoint-last.pth is unavailable or corrupted. Searching for available checkpoints...")
            import glob
            from natsort import natsorted
            all_ckpts = natsorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint-*.pth')))
            all_ckpts = [c for c in all_ckpts if 'last' not in c]
            for c in reversed(all_ckpts):
                try:
                    checkpoint = torch.load(c, map_location='cpu', weights_only=False)
                    logger.info("Successfully loaded fallback checkpoint from %s", c)
                    ckpt_to_load = c
                    break
                except Exception as e:
                    logger.warning("Failed to load fallback checkpoint from %s: %s", c, e)

        elif checkpoint is None and args.resume:
            logger.warning("Specified resume checkpoint '%s' not found or could not be loaded.", ckpt_to_load)

    if checkpoint is not None:
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)

        ema_state_dict1 = checkpoint['model_ema1']
        ema_state_dict2 = checkpoint['model_ema2']
        model_without_ddp.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        model_without_ddp.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model_without_ddp.named_parameters()]
        logger.info("Resumed checkpoint from %s", ckpt_to_load)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            logger.info("Loaded optimizer & scaler state!")
        del checkpoint
    else:
        model_without_ddp.ema_params1 = copy.deepcopy(list(model_without_ddp.parameters()))
        model_without_ddp.ema_params2 = copy.deepcopy(list(model_without_ddp.parameters()))
        logger.info("Training from scratch (no checkpoint loaded)")

    # Define additional loss if args.add_loss is specified
    additional_loss_fn = None
    if args.add_loss:
        if args.add_loss_name == 'bce':
            additional_loss_fn = torch.nn.BCEWithLogitsLoss()
        elif args.add_loss_name == 'dice':
            additional_loss_fn = custom_losses.DiceLoss()
        elif args.add_loss_name == 'soft_dice':
            additional_loss_fn = custom_losses.SoftDiceLoss()
        elif args.add_loss_name == 'dice_bce':
            additional_loss_fn = custom_losses.BCEDiceLoss()
        elif args.add_loss_name == 'weighted_dice_bce':
            # Example weight, can be parameterized further if needed
            additional_loss_fn = custom_losses.WeightedBCEDiceLoss(w_bce=1.0, w_dice=2.0)
        else:
            raise ValueError(
                f"Unknown additional loss function: {args.add_loss_name}")

    # Training loop
    logger.info("Start training for %d epochs", args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args, additional_loss_fn=additional_loss_fn)

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0:
            if epoch + 1 == args.epochs:
                epoch_name = args.epochs
            else:
                epoch_name = "last"
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name=epoch_name
            )

        if epoch % 1000 == 0 and epoch > 0:
            misc.save_model(
                args=args,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                epoch=epoch
            )

        # Perform validation at specified intervals
        if args.online_eval and (epoch % args.eval_freq == 0 or epoch + 1 == args.epochs) and epoch != 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                validation(model_without_ddp, data_loader_val, device, epoch, log_writer=log_writer, threshold=args.threshold,
                           dataset=args.dataset, add_loss=args.add_loss, additional_loss_fn=additional_loss_fn)
            torch.cuda.empty_cache()

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()
    misc.save_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        epoch=args.epochs,
        epoch_name=args.epochs
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time: %s", total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
