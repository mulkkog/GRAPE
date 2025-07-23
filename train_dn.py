import os
import yaml
import random
import argparse 
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

import torch.nn.functional as F
import datasets
import models
import utils
from test_dn import eval_psnr

# WANB 사용 시 import
try:
    import wandb
except ImportError:
    wandb = None


def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log(f'{tag} dataset: size={len(dataset)}')
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=(tag == 'train'),
        num_workers=4,
        pin_memory=True
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def create_lr_scheduler(optimizer, schedule_config, start_iter=0):
    name = schedule_config.get('name', 'MultiStepLR')
    args = schedule_config.get('args', {})

    if name == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, **args)

    elif name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **args)

    elif name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, **args)

    elif name == 'CosineAnnealingWarmRestarts':          # ★ 추가
        scheduler = CosineAnnealingWarmRestarts(optimizer, **args)

    else:
        raise ValueError(f"Unsupported lr scheduler: {name}")

    # epoch 기반 스케줄러라면 현재 진행 상황을 맞춰 줌
    if hasattr(scheduler, 'last_epoch'):
        scheduler.last_epoch = start_iter - 1          # 0-based → 이미 끝난 epoch
    if hasattr(scheduler, '_step_count'):              # WarmRestarts용 보정
        scheduler._step_count = start_iter

    return scheduler

    
def prepare_training():
    if config.get('pre_train') is not None:
        print('loading pre_train model... ', config['pre_train'])
        model = models.make(config['model']).cuda()
        model_dict = model.state_dict()

        sv_file = torch.load(config['pre_train'])
        pretrained_dict = sv_file['model']['sd']
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        start_iter = 0
        if config.get('lr_schedule') is None:
            lr_scheduler = None
        else:
            lr_scheduler = create_lr_scheduler(optimizer, config['lr_schedule'], start_iter=start_iter)
    elif config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(model.parameters(), sv_file['optimizer'], load_sd=True)
        start_iter = sv_file.get('iter', 0) + 1
        state = sv_file.get('state', None)
        if state is not None:
            torch.set_rng_state(state)
        print(f"Resuming from iter {start_iter}...")
        if config.get('lr_schedule') is None:
            lr_scheduler = None
        else:
            lr_scheduler = create_lr_scheduler(optimizer, config['lr_schedule'], start_iter=start_iter)
            lr_scheduler.last_epoch = start_iter - 1
    else:
        print('prepare_training from start')
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
        start_iter = 0
        if config.get('lr_schedule') is None:
            lr_scheduler = None
        else:
            lr_scheduler = create_lr_scheduler(optimizer, config['lr_schedule'], start_iter=start_iter)

    log(f"model: #params={utils.compute_num_params(model, text=True)}")
    log(f"model: #struct={model}")
 
    return model, optimizer, start_iter, lr_scheduler

def ellipse_regularizer(sigma_x, sigma_y, margin=0.04):
    # r = |σx-σy| / max(σx,σy)  (0 = 완전 원)
    r = torch.abs(sigma_x - sigma_y) / (torch.maximum(sigma_x, sigma_y) + 1e-9)
    return F.relu(margin - r)          # margin보다 작을 때만 페널티
 
def fft_loss(pred: torch.Tensor,
             gt:   torch.Tensor,
             eps: float = 1e-6) -> torch.Tensor:
    """
    pred, gt : (B, C, H, W)   값 범위 0–1
    반환     : 스칼라 텐서
    """
    # 1) 2-D FFT (H, W 방향) – 전체 복소 스펙트럼
    pred_f = torch.fft.fft2(pred, dim=(-2, -1), norm='ortho')
    gt_f   = torch.fft.fft2(gt,   dim=(-2, -1), norm='ortho')

    # 2) 실수·허수부를 (-1) 새 축으로 결합 → (B,C,H,W,2)
    pred_f = torch.stack((pred_f.real, pred_f.imag), dim=-1)
    gt_f   = torch.stack((gt_f.real,   gt_f.imag),   dim=-1)
 
    return F.l1_loss(pred_f, gt_f)

def main(config_, save_path):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path)

    # config 저장
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    # WANDB 초기화
    if config.get('use_wanb', False):
        if wandb is None:
            raise ImportError("wandb 모듈이 설치되어 있지 않습니다. pip install wandb 로 설치해주세요.")
        wandb.init(project="GaussianSR", name=config.get('run_name', None), config=config)

    train_loader, val_loader = make_data_loaders()

    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, start_epoch, lr_scheduler = prepare_training()
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    if n_gpus > 1:
        model = nn.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val', 5)
    epoch_save = config.get('epoch_save', 10)

    best_val = -float('inf')
    timer = utils.Timer()

    for epoch in range(start_epoch, epoch_max + 1):
        t_epoch_start = timer.t()
        model.train()
        epoch_loss = utils.Averager()
        l1_fn   = nn.L1Loss()

        for batch in tqdm(train_loader, leave=False, desc=f'Epoch {epoch}/{epoch_max}'):
            for k, v in batch.items():
                batch[k] = v.cuda()

            inp_norm = config['data_norm']['inp']
            gt_norm = config['data_norm']['gt']
            inp_sub = torch.FloatTensor(inp_norm['sub']).view(1, -1, 1, 1).cuda()
            inp_div = torch.FloatTensor(inp_norm['div']).view(1, -1, 1, 1).cuda()
            gt_sub = torch.FloatTensor(gt_norm['sub']).view(1, 1, -1).cuda()
            gt_div = torch.FloatTensor(gt_norm['div']).view(1, 1, -1).cuda()

            inp = (batch['inp'] - inp_sub) / inp_div
            gt = (batch['gt'] - gt_sub) / gt_div
 
            pred =  model(inp)

            loss  = l1_fn(pred, gt) 

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss.add(loss.item())

        if lr_scheduler is not None:
            lr_scheduler.step()

        log(f'Epoch {epoch}/{epoch_max} - Train Loss: {epoch_loss.item():.4f}')
        writer.add_scalar('loss/train', epoch_loss.item(), epoch)

        if config.get('use_wanb', False):
            wandb.log({'train_loss': epoch_loss.item(), 'epoch': epoch})

        # Validation
        if epoch_val and epoch % epoch_val == 0:
            with torch.no_grad():
                model_eval = model.module if n_gpus > 1 else model
                val_psnr, val_ssim, _, _ = eval_psnr(val_loader, model_eval, data_norm=config['data_norm'], eval_type=None, eval_bsize=config.get('eval_bsize'))
                log(f'Validation Epoch {epoch}- PSNR: {val_psnr:.4f} SSIM: {val_ssim:.4f}') 
                writer.add_scalar('psnr/val', val_psnr, epoch)
                if config.get('use_wanb', False):
                    wandb.log({'val_psnr': val_psnr})

                if val_psnr > best_val:
                    best_val = val_psnr
                    torch.save({
                        'model': {**config['model'], 'sd': model_eval.state_dict()},
                        'optimizer': {**config['optimizer'], 'sd': optimizer.state_dict()},
                        'epoch': epoch,
                        'state': torch.get_rng_state()
                    }, os.path.join(save_path, 'epoch-best.pth'))
                    log(f'New best model saved @ Epoch {epoch} with PSNR: {val_psnr:.4f}')

        # Save every N epochs
        if epoch_save and epoch % epoch_save == 0:
            model_save = model.module if n_gpus > 1 else model
            torch.save({
                'model': {**config['model'], 'sd': model_save.state_dict()},
                'optimizer': {**config['optimizer'], 'sd': optimizer.state_dict()},
                'epoch': epoch,
                'state': torch.get_rng_state()
            }, os.path.join(save_path, f'epoch_{epoch}.pth'))

        # Save last epoch
        torch.save({
            'model': {**config['model'], 'sd': model.module.state_dict() if n_gpus > 1 else model.state_dict()},
            'optimizer': {**config['optimizer'], 'sd': optimizer.state_dict()},
            'epoch': epoch
        }, os.path.join(save_path, 'epoch-last.pth'))

        t_epoch = timer.t() - t_epoch_start
        avg_epoch_time = timer.t() / (epoch - start_epoch + 1)
        remaining_epochs = epoch_max - epoch
        eta_seconds = remaining_epochs * avg_epoch_time
        eta_text = utils.time_text(eta_seconds)
        log(f"Epoch {epoch} complete in {utils.time_text(t_epoch)} - 남은 시간: {eta_text}")

    if config.get('use_wanb', False):
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--resume', default=None)
    # WANB 관련 인자 추가
    parser.add_argument('--use_wanb', action='store_false', help='Enable wandb logging')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    setup_seed(2025)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print("config loaded.")

    # WANB 사용 여부 및 run_name 설정
    config['use_wanb'] = args.use_wanb
    config['run_name'] = f"{config['model']['args']['encoder_spec']['name']}-{config['model']['name']}"

    # config 파일명에서 'train-' prefix와 '.yaml' 확장자를 제거하여 save 폴더명 생성
    save_name = args.config.split('/')[-1][len('train-'):-len('.yaml')]
    if args.tag is not None:
        save_name += args.tag
    save_path = os.path.join('./save', save_name)

    if args.resume is None:
        config['resume'] = None

    main(config, save_path)
