import argparse
import os

import torch
from thop import profile

import models  # 사용자가 정의한 models 모듈

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='save/edsr+gauss_learnhybrid/iter-best.pth')
    parser.add_argument('--gpu', default='3')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # 1. 모델 로드
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    # 2. 파라미터 정보 출력 (단위: M)
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    # 3. MACs 측정을 위한 입력 텐서 생성
    inp_shape = (3, 256, 256)
    x = torch.randn(1, *inp_shape).cuda()

    # 4. THOP으로 FLOPs 측정 (FLOPs ≈ 2 × MACs로 보기도 하나, 여기선 MACs를 FLOPs로 간주)
    flops, _ = profile(model, inputs=(x, ))

    # 결과 출력 (GMACs = GFLOPs)
    print(f"FLOPs (Multiply-Accumulate): {flops:.2f}")
    print(f"FLOPs in G: {flops / 1e9:.4f} G")
