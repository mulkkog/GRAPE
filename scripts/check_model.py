from operator import imod
import os
import torch
import argparse 
from models.my_sr import MyGaussianSimple,  MyGaussianEnsemble, MyGaussianSimpleFast, MyGaussianEnsembleFast 
from models.gaussian import ContinuousGaussian
from utils import make_coord

def build_model(model_name, encoder_name):
 
    encoder_spec = {
        'name': encoder_name,
        'args': {
            'no_upsampling': False,
            'scale': 4
        }
    }
 
    if model_name == 'MyGaussianSimple':
        return MyGaussianSimple(encoder_spec=encoder_spec).cuda()
    elif model_name == 'MyGaussianSimpleFast':
        return MyGaussianSimpleFast(encoder_spec=encoder_spec).cuda()  
    elif model_name == 'MyGaussianEnsemble':
        return MyGaussianEnsemble(encoder_spec=encoder_spec).cuda()  
    elif model_name == 'MyGaussianEnsembleFast':
        return MyGaussianEnsembleFast(encoder_spec=encoder_spec).cuda()   
    elif model_name == 'ContinuousGaussian':
        return ContinuousGaussian(encoder_spec=encoder_spec).cuda()  

    else:
        raise ValueError(f"Unsupported model: {model_name}")

def dummy_input(h=128, w=128):
    img = torch.randn(1, 3, h, w).cuda()

    return img 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ContinuousGaussian')
    parser.add_argument('--encoder', type=str, default='edsr-baseline', help='edsr-baseline, swinir 등')
    parser.add_argument('--gpu', type=str, default='4', help='GPU device id')
    parser.add_argument('--scale', type=str, default='4')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    model = build_model(args.model, args.encoder)
    model.eval()

    input_img = dummy_input()
    scale = torch.tensor([int(args.scale)]).cuda()

    # 메모리 측정 초기화
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        start_peak = torch.cuda.max_memory_allocated()

        _ = model(input_img, scale)
 
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

    print(f"\nInference GPU Memory Usage")
    print(f"- Initial Memory: {start_mem / 1024**2:.2f} MB")
    print(f"- After Inference: {end_mem / 1024**2:.2f} MB")
    print(f"- Peak Memory Used: {peak_mem / 1024**2:.2f} MB")

    # 속도 측정
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(10):  # warm-up
        _ = model(input_img, scale)
 
    starter.record()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_img, scale)
 
    ender.record()
    torch.cuda.synchronize()

    elapsed_time_ms = starter.elapsed_time(ender) / 100
    print(f"\nAverage Inference Time: {elapsed_time_ms:.2f} ms")
    print(f"FPS: {1000 / elapsed_time_ms:.2f}")

    from thop import profile
    inputs = (input_img, scale)  
    macs, params = profile(model, inputs=inputs)
    print(f"\nFLOPs: {macs / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")
