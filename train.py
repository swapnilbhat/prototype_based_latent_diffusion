import os
import torch
import argparse
import itertools
import numpy as np
from unet import Unet
from tqdm import tqdm
import torch.optim as optim
from diffusion import GaussianDiffusion
from utils import get_named_beta_schedule
from embedding import ConditionalEmbedding
from scheduler import GradualWarmupScheduler
from dataloaders import load_dataset
import random
import time 
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import get_rank, init_process_group, destroy_process_group
from diffusers import AutoencoderKL

def encode_img(vae,input_imgs):
    with torch.no_grad():
        # Encode the batch of images and scale appropriately
        latent = vae.encode(input_imgs * 2 - 1)  # Scaling to [-1, 1]
    return 0.18215 * latent.latent_dist.sample()  # Apply latent scaling

def decode_img(vae,latents):
    # Batch of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    return images.detach()

def seed_everything(seed):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def train(params:argparse.Namespace):
    # Read the local rank from the environment
    local_rank = int(os.getenv('LOCAL_RANK', 0))  
    
    # Initialize the process group for multiple GPUs
    if torch.cuda.device_count() > 1:
        init_process_group(backend="nccl")  
    
    save_dir='results/'
    assert params.genbatch % (torch.cuda.device_count() * params.clsnum) == 0 , 'please re-set your genbatch!!!'
    
    # set device
    device = torch.device(f"cuda:{local_rank}")
    seed_everything(int(local_rank))
    
    # dataloader,sampler,centers=load_dataset(params.batchsize, params.numworkers)
    
    if torch.cuda.device_count() > 1:
        dataloader, sampler, centers = load_dataset(params.batchsize, params.numworkers, use_distributed_sampler=True)
    else:
        dataloader, _, centers = load_dataset(params.batchsize, params.numworkers, use_distributed_sampler=False)
    class_number=centers.shape[0]
    
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    vae=vae.to(device)
    
    if params.path_resume != None:
        print('path resume is given')
        hyperparams_model = {}
        with open(os.path.join(params.path_resume, 'params.txt'), 'r') as f:
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
        
        print("Model is created")
        if os.path.exists(params.path_resume):
            save_dir = params.path_resume
            lastpath = params.path_resume + "/checkpoints"
            lastepc = torch.load(lastpath + "/last_epoch.pt")['last_epoch']
            # load checkpoints
            checkpoint = torch.load(os.path.join(lastpath, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
            net.load_state_dict(checkpoint['net'])
            
            cemblayer = ConditionalEmbedding(class_number, int(hyperparams_model["cdim"]), int(hyperparams_model["cdim"]))
            cemblayer.load_state_dict(checkpoint['cemblayer'])
            cemblayer.condEmbedding[0].weight.requires_grad = False
            cemblayer = cemblayer.to(device)
            
            print(f'load checkpoints from {lastpath}')
        else: 
            raise "path not exist"
    else:
        #Unet model initialization
        net = Unet(
                    in_ch = params.inch,
                    mod_ch = params.modch,
                    out_ch = params.outch,
                    ch_mul = params.chmul,
                    num_res_blocks = params.numres,
                    cdim = params.cdim,
                    use_conv = params.useconv,
                    droprate = params.droprate,
                    dtype = params.dtype
                )
        net=net.to(device)
        
        cemblayer = ConditionalEmbedding(class_number, params.cdim, params.cdim)
        cemblayer.condEmbedding[0].weight = torch.nn.Parameter(centers)
        cemblayer.condEmbedding[0].weight.requires_grad = False
        cemblayer = cemblayer.to(device)
        print("Prototypes are loaded")
        
        lastepc = 0
        if local_rank == 0:
            os.makedirs(save_dir,exist_ok=True)
            os.makedirs(save_dir + '/samples',exist_ok=True)
            os.makedirs(save_dir + '/checkpoints',exist_ok=True)
            # write params to file
            with open(os.path.join(save_dir, 'params.txt'), 'w') as f:
                for k, v in vars(params).items():
                    f.write(f'{k}: {v}\n')
                    
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
               dtype = params.dtype,
                model = net,
                betas = betas,
                w = params.w,
                v = params.v,
                device = device
            )

    if torch.cuda.device_count() > 1:
        diffusion.model = DDP(diffusion.model, device_ids=[local_rank], output_device=local_rank)
        cemblayer = DDP(cemblayer, device_ids=[local_rank], output_device=local_rank)
    # optimizer settings
    optimizer = torch.optim.AdamW(
                    itertools.chain(
                        diffusion.model.parameters(),
                        cemblayer.parameters()
                    ),
                    lr = params.lr,
                    weight_decay = 1e-4
                )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer = optimizer,
                            T_max = params.epoch, 
                            eta_min = 0,
                            last_epoch = -1
                        )
    warmUpScheduler = GradualWarmupScheduler(
                            optimizer = optimizer,
                            multiplier =  params.multiplier,
                            warm_epoch =  params.epoch // 20, 
                            after_scheduler = cosineScheduler,
                            last_epoch = lastepc
                        )
    
    if lastepc != 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        warmUpScheduler.load_state_dict(checkpoint['scheduler'])
        
    # training
    cnt = torch.cuda.device_count()
    for epc in range(lastepc, params.epoch): 
        diffusion.model.train()
        cemblayer.train()
        if torch.cuda.device_count() > 1:
            sampler.set_epoch(epc)  
        
        with tqdm(dataloader, dynamic_ncols=True, disable=(local_rank % cnt != 0)) as tqdmDataLoader:
            for img,lab in tqdmDataLoader:
                f = open(os.path.join(save_dir, 'log.txt'), 'a')
                if local_rank == 0:
                    f.write(f'epoch: {epc + 1}, iteration: {tqdmDataLoader.n}\n')
                b = img.shape[0]
                optimizer.zero_grad()
                x_0 = img.to(device)
                x_0 = encode_img(vae,x_0)
                lab = lab.to(device)
                cemb = cemblayer(lab)
                
                loss = diffusion.trainloss(x_0, cemb, file=f)
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(
                    ordered_dict={
                        "epoch": epc + 1,
                        "loss: ": loss.item(),
                        "batch per device: ":x_0.shape[0],
                        "img shape: ": x_0.shape[1:],
                        "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                    }
                )
                f.close()
        warmUpScheduler.step()
        
        # save checkpoint
        if (epc + 1) %  params.interval == 0:
            print('saving checkpoint')
            checkpoint = {
                'net': diffusion.model.module.state_dict() if isinstance(diffusion.model, DDP) else diffusion.model.state_dict(),
                'cemblayer': cemblayer.module.state_dict() if isinstance(cemblayer, DDP) else cemblayer.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': warmUpScheduler.state_dict()
            }
            torch.save({'last_epoch':epc+1}, os.path.join(save_dir,'checkpoints/last_epoch.pt'))
            torch.save(checkpoint,  os.path.join(save_dir, f'checkpoints/ckpt_{epc+1}_checkpoint.pt'))
        torch.cuda.empty_cache()

    if torch.cuda.device_count() > 1:
        destroy_process_group()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='test for diffusion model')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size per device for training Unet model')
    parser.add_argument('--numworkers', type=int, default=4, help='num workers for training Unet model')
    parser.add_argument('--inch', type=int, default=4, help='input channels for Unet model')
    parser.add_argument('--modch', type=int, default=64, help='model channels for Unet model')
    parser.add_argument('--T', type=int, default=1000, help='timesteps for Unet model')
    parser.add_argument('--outch', type=int, default=4, help='output channels for Unet model')
    parser.add_argument('--chmul', type=list, default=[1,2,4,4], help='architecture parameters training Unet model')
    parser.add_argument('--numres', type=int, default=2, help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim', type=int, default=768, help='dimension of conditional embedding')
    parser.add_argument('--useconv', type=bool, default=True, help='whether use convlution in downsample')
    parser.add_argument('--dtype', default=torch.float32)
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--w', type=float, default=1.8, help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v', type=float, default=0.3, help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch', type=int, default=1000, help='epochs for training')
    parser.add_argument('--multiplier', type=float, default=2.5, help='multiplier for warmup')
    parser.add_argument('--path_resume', default=None, type=str, help='Path of the checkpoint to resume training')
    parser.add_argument('--genbatch', type=int, default=64, help='batch size for sampling process')
    parser.add_argument('--clsnum', type=int, default=16, help='num of label classes')
    parser.add_argument('--droprate', type=float, default=0, help='dropout rate for model')
    parser.add_argument('--interval', type=int, default=20, help='checkpoint saving interval')
    
    args = parser.parse_args()
    train(args)
    