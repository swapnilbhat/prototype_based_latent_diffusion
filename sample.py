import os
import torch
import argparse
from math import ceil

from unet import Unet
from diffusion import GaussianDiffusion
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group, all_gather, get_world_size
from torchvision import transforms
from torchvision.utils import save_image
from diffusers import AutoencoderKL

def decode_img(vae,latents):
    # Batch of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images.detach()

@torch.no_grad()
def sample(params:argparse.Namespace):
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    # Initialize the process group for multiple GPUs
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl") 
    
    # set device
    device = torch.device(f"cuda:{local_rank}")
    
    # load models
    hyperparams_model = {}
    with open(os.path.join(params.path, 'params.txt'), 'r') as f:
        for line in f.readlines():
            key, val = line.split(':')
            hyperparams_model[key] = val[:-1]
    
    net = Unet(
                in_ch = int(hyperparams_model["inch"]),
                mod_ch = int(hyperparams_model["modch"]),
                out_ch = int(hyperparams_model["outch"]),
                ch_mul = [ int(i)  for i in hyperparams_model["chmul"][2:-1].split(",")],
                num_res_blocks = int(hyperparams_model["numres"]),
                cdim = int(hyperparams_model["cdim"]),
                use_conv= True if hyperparams_model["useconv"] == ' True' else False,
                droprate = float(hyperparams_model["droprate"]),
                dtype=torch.float32
            ).to(device)
    print("model is created")
    
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae=vae.to(device)
    
    checkpoint = torch.load(os.path.join(params.path, f'checkpoints/ckpt_{params.epoch}_checkpoint.pt'), map_location='cpu')
    
    net.load_state_dict(checkpoint['net'])
    cemblayer = ConditionalEmbedding(params.clsnum, int(hyperparams_model["cdim"]), int(hyperparams_model["cdim"])).to(device)
    cemblayer.load_state_dict(checkpoint['cemblayer'])
    
    # settings for diffusion model
    betas = get_named_beta_schedule(num_diffusion_timesteps = int(hyperparams_model["T"]))
    
    diffusion = GaussianDiffusion(
                    dtype = torch.float32,
                    model = net,
                    betas = betas,
                    w = float(hyperparams_model["w"]),
                    v =  float(hyperparams_model["v"]),
                    device = device
                )
    
    if torch.cuda.device_count() > 1:
        diffusion.model = DDP(diffusion.model, device_ids=[local_rank], output_device=local_rank)
        cemblayer = DDP(cemblayer, device_ids=[local_rank], output_device=local_rank)
    
    #eval
    diffusion.model.eval()
    cemblayer.eval()
    cnt = torch.cuda.device_count()
    
    if params.fid:
        numloop = ceil(params.genum  / params.genbatch)
    else:
        numloop = 1
    
    each_device_batch = params.genbatch // cnt
    # label settings
    if params.label == 'range':
        lab = torch.ones(params.clsnum, each_device_batch // params.clsnum).type(torch.long) \
            * torch.arange(start = 0, end = params.clsnum).reshape(-1,1)
        lab = lab.reshape(-1, 1).squeeze()
        lab = lab.to(device)
    else:
        lab = torch.randint(low = 0, high = params.clsnum, size = (each_device_batch,), device=device)
    
    cemb = cemblayer(lab)
    #genshape = (each_device_batch, 3, int(hyperparams_model["image_size"]), int(hyperparams_model["image_size"]))#change the image shape
    genshape=(each_device_batch, 4, 32, 32)
    all_samples = []
    if local_rank == 0:
        print(numloop)
    
    for _ in range(numloop):
        if params.ddim:
            generated = diffusion.ddim_sample(genshape, params.num_steps, float(hyperparams_model["eta"]), hyperparams_model["select"][1:], cemb = cemb)
        else:
            generated = diffusion.sample(genshape, cemb = cemb)
        img = generated
        img=decode_img(vae,img).to(device)
        img = img.reshape(params.clsnum, each_device_batch // params.clsnum, 3, 256, 256).contiguous()#reshape size change
        if torch.cuda.device_count() > 1:
            gathered_samples = [torch.zeros_like(img) for _ in range(get_world_size())]        
            all_gather(gathered_samples, img)
            all_samples.extend([img.cpu() for img in gathered_samples])
        else:
            gathered_samples = [torch.zeros_like(img) for _ in range(1)]        
            gathered_samples[0] = img
            all_samples.extend([img.cpu() for img in gathered_samples])
               
    samples = torch.concat(all_samples, dim = 1).reshape(params.genbatch * numloop, 3, 256, 256)
    
    invTrans = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=[1 / 0.5, 1 / 0.5, 1 / 0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.]),
    ])
    
    samples = invTrans(samples).clamp(0, 1)
    
    if local_rank == 0:
        save_path = os.path.join(params.path, f'samples_{params.epoch}/')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i in range(params.genbatch * numloop):
            save_image(samples[i], os.path.join(save_path, f'{i}.png'),normalize = True)
            
    if torch.cuda.device_count() > 1:
        destroy_process_group()

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--genbatch', type=int, default=16, help='batch size for sampling process')
    parser.add_argument('--path', type=str, default=None, help='path for loading models')
    parser.add_argument('--epoch', type=int, default=49, help='epochs for loading models')
    parser.add_argument('--label', type=str, default='range', help='labels of generated images')
    parser.add_argument('--clsnum', type=int, default=16, help='num of label classes')
    parser.add_argument('--fid', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=False, help='generate samples used for quantative evaluation')
    parser.add_argument('--genum', type=int, default=64, help='num of generated samples')
    parser.add_argument('--select', type=str, default='linear', help='selection stragies for DDIM')
    parser.add_argument('--ddim', type=lambda x:(str(x).lower() in ['true','1', 'yes']), default=False, help='whether to use ddim') 
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--num_steps', type=int, default=50, help='sampling steps for DDIM')
    args = parser.parse_args()
    sample(args)
    