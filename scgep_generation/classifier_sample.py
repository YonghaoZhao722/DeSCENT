"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

# Set environment variables for distributed training to avoid warnings
# Get GPU ID from environment variable, default to 0
gpu_id = os.environ.get('CUDA_DEVICE_ID', '0')
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    model_and_diffusion_defaults,
    classifier_and_diffusion_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import scanpy as sc
import torch
from VAE.VAE_model import VAE
from tqdm import tqdm

def load_VAE(ae_dir, num_gene):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(ae_dir))
    return autoencoder

def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def load_cell_type_mapping(cell_ratios_file):
    """Load cell type mapping from the cell ratios file"""
    import pandas as pd
    
    # Read the file to get column names (excluding the first column which is sample/cell_type)
    if cell_ratios_file.endswith('.tsv'):
        df = pd.read_csv(cell_ratios_file, sep='\t', nrows=0)  # Only read header
    else:
        df = pd.read_csv(cell_ratios_file, nrows=0)  # Only read header
    
    # Get cell type names from columns (excluding the first column)
    cell_types = df.columns[1:].tolist()
    
    # Create mapping from cell type name to index
    cell_type_to_index = {cell_type: idx for idx, cell_type in enumerate(cell_types)}
    
    return cell_type_to_index

def main(args):
    cell_type = args.cell_type
    num_samples = args.num_samples
    sample_id = args.sample_id
    multi = args.multi
    inter = args.inter
    weight = args.weight

    # Set batch_size equal to num_samples
    args.batch_size = num_samples

    # Ensure minimum batch size of 2 for VAE processing
    if args.batch_size < 2:
        print(f"Skipping cell type {cell_type} as it requires less than 2 cells")
        return

    dist_util.setup_dist()
    # Configure logger without directory message
    # logger.configure(format_strs=["stdout"])  # Only use stdout format

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    if multi:
        args.num_class = args.num_class1 # how many classes in this condition
        classifier1 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier1.load_state_dict(
            dist_util.load_state_dict(args.classifier_path1, map_location="cpu")
        )
        classifier1.to(dist_util.dev())
        classifier1.eval()

        args.num_class = args.num_class2 # how many classes in this condition
        classifier2 = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier2.load_state_dict(
            dist_util.load_state_dict(args.classifier_path2, map_location="cpu")
        )
        classifier2.to(dist_util.dev())
        classifier2.eval()

    else:
        classifier = create_classifier(**args_to_dict(args, (['num_class']+list(classifier_and_diffusion_defaults().keys()))[:3]))
        classifier.load_state_dict(
            dist_util.load_state_dict(args.classifier_path, map_location="cpu")
        )
        classifier.to(dist_util.dev())
        classifier.eval()

    '''
    control function for Gradient Interpolation Strategy
    '''
    def cond_fn_inter(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        # xt = diffusion.q_sample(th.tensor(init,device=dist_util.dev()),t*th.ones(init.shape[0],device=dist_util.dev(),dtype=torch.long),)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected1 = log_probs[range(len(logits)), y1.view(-1)]
            selected2 = log_probs[range(len(logits)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2

            # l2_loss = ((x_in-xt)**2).mean()
            # grad3 = th.autograd.grad(-l2_loss, x_in, retain_graph=True)[0] * 100

            return grad1+grad2#+grad3

    '''
    control function for multi-conditional generation
    Two conditional generation here
    '''
    def cond_fn_multi(x, t, y=None):
        assert y is not None
        y1 = y[:,0]
        y2 = y[:,1]
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits1 = classifier1(x_in, t)
            log_probs1 = F.log_softmax(logits1, dim=-1)
            selected1 = log_probs1[range(len(logits1)), y1.view(-1)]

            logits2 = classifier2(x_in, t)
            log_probs2 = F.log_softmax(logits2, dim=-1)
            selected2 = log_probs2[range(len(logits2)), y2.view(-1)]
            
            grad1 = th.autograd.grad(selected1.sum(), x_in, retain_graph=True)[0] * args.classifier_scale1
            grad2 = th.autograd.grad(selected2.sum(), x_in, retain_graph=True)[0] * args.classifier_scale2
            
            return grad1+grad2

    '''
    control function for one conditional generation
    '''
    def cond_fn_ori(x, t, y=None):
        # print('condition shape:',y.shape)
        assert y is not None
        # print('y:',y)
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            # print('logits.shape:',logits.shape)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            grad = th.autograd.grad(selected.sum(), x_in, retain_graph=True)[0] * args.classifier_scale
            return grad
        
    def model_fn(x, t, y=None, init=None, diffusion=None):
        assert y is not None
        if args.class_cond:
            return model(x, t, y if args.class_cond else None)
        else:
            return model(x, t)
        
    if inter:
        # input real cell expression data as initial noise
        ori_adata = sc.read_h5ad(args.init_cell_path)
        sc.pp.normalize_total(ori_adata, target_sum=1e4)
        sc.pp.log1p(ori_adata)

    all_cell = []
    sample_num = 0
    while sample_num < num_samples:
        model_kwargs = {}

        if not multi and not inter:
            classes = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)

        if multi:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            # classes3 = ... if more conditions
            classes = th.stack((classes1,classes2), dim=1)

        if inter:
            classes1 = (cell_type[0])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes2 = (cell_type[1])*th.ones((args.batch_size,), device=dist_util.dev(), dtype=th.long)
            classes = th.stack((classes1,classes2), dim=1)

        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        if inter:
            celltype = ori_adata.obs['period'].cat.categories.tolist()[cell_type[0]]
            adata = ori_adata[ori_adata.obs['period']==celltype].copy()

            start_x = adata.X
            autoencoder = load_VAE(args.ae_dir, args.num_gene)
            start_x = autoencoder(torch.tensor(start_x,device=dist_util.dev()),return_latent=True).detach().cpu().numpy()

            n, m = start_x.shape  
            if n >= args.batch_size:  
                start_x = start_x[:args.batch_size, :]  
            else:  
                repeat_times = args.batch_size // n  
                remainder = args.batch_size % n  
                start_x = np.concatenate([start_x] * repeat_times + [start_x[:remainder, :]], axis=0)  
            
            noise = diffusion.q_sample(th.tensor(start_x,device=dist_util.dev()),args.init_time*th.ones(start_x.shape[0],device=dist_util.dev(),dtype=torch.long),)
            model_kwargs["init"] = start_x
            model_kwargs["diffusion"] = diffusion

        if multi:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_multi,
                device=dist_util.dev(),
                noise = None,
                start_time=diffusion.betas.shape[0],
                start_guide_steps=args.start_guide_steps,
            )
        elif inter:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_inter,
                device=dist_util.dev(),
                noise = noise,
                start_time=diffusion.betas.shape[0],
                start_guide_steps=args.start_guide_steps,
            )
        else:
            sample, traj = sample_fn(
                model_fn,
                (args.batch_size, args.input_dim),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn_ori,
                device=dist_util.dev(),
                noise = None,
            )

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        if args.filter:
            for sample in gathered_samples:
                if multi:
                    logits1 = classifier1(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    logits2 = classifier2(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    prob1 = F.softmax(logits1, dim=-1)
                    prob2 = F.softmax(logits2, dim=-1)
                    type1 = torch.argmax(prob1, 1)
                    type2 = torch.argmax(prob2, 1)
                    select_index = ((type1 == cell_type[0]) & (type2 == cell_type[1]))
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
                elif inter:
                    logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    # print('logits',logits)
                    prob = F.softmax(logits, dim=-1)
                    left = (prob[:,cell_type[0]] > weight[0]/10-0.15) & (prob[:,cell_type[0]] < weight[0]/10+0.15)
                    right = (prob[:,cell_type[1]] > weight[1]/10-0.15) & (prob[:,cell_type[1]] < weight[1]/10+0.15)
                    select_index = left & right
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
                else:
                    logits = classifier(sample, torch.zeros((sample.shape[0]), device=sample.device))
                    # print('logits',logits)
                    prob = F.softmax(logits, dim=-1)
                    type = torch.argmax(prob, 1)
                    select_index = (type == cell_type[0])
                    all_cell.extend([sample[select_index].cpu().numpy()])
                    sample_num += select_index.sum().item()
            # logger.log(f"created {sample_num} samples")
        else:
            all_cell.extend([sample.cpu().numpy() for sample in gathered_samples])
            sample_num = len(all_cell) * args.batch_size
            # logger.log(f"created {sample_num} samples")

    arr = np.concatenate(all_cell, axis=0)
    os.makedirs(args.sample_dir+str(sample_id), exist_ok=True)
    save_data(arr, traj, args.sample_dir+str(sample_id)+'/'+str(cell_type[0]))
    dist.barrier()

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=9000,
        use_ddim=False,
        class_cond=False, 

        model_path="output/diffusion_checkpoint/muris_diffusion/model000000.pt", 

        # ***if common conditional generation & gradient interpolation, use this path***
        classifier_path="output/classifier_checkpoint/classifier_muris/model000100.pt",
        # ***if multi-conditional, use this path. replace this to your own classifiers***
        classifier_path1="output/classifier_checkpoint/classifier_muris_ood_type/model200000.pt",
        classifier_path2="output/classifier_checkpoint/classifier_muris_ood_organ/model200000.pt",
        num_class1 = 2,  # set this to the number of classes in your own dataset. this is the first condition (for example cell organ).
        num_class2 = 2,  # this is the second condition (for example cell type).

        # ***if common conditional generation, use this scale***
        classifier_scale=2,
        # ***in multi-conditional, use this scale. scale1 and scale2 are the weights of two classifiers***
        # ***in Gradient Interpolation, use this scale, too. scale1 and scale2 are the weights of two gradients***
        classifier_scale1=2,
        classifier_scale2=2,

        # ***if gradient interpolation, replace these base on your own situation***
        ae_dir='output/Autoencoder_checkpoint/WOT/model_seed=0_step=150000.pt', 
        num_gene=19423,
        init_time = 600,    # initial noised state if interpolation
        init_cell_path = 'data/WOT/filted_data.h5ad',   #input initial noised cell state

        sample_dir=f"output/simulated_samples/",
        start_guide_steps = 500,     # the time to use classifier guidance
        filter = False,   # filter the simulated cells that are classified into other condition, might take long time
        
        # Added parameters for cell types and ratios
        cell_type=[0],  # Default cell type index
        multi=False,    # Multi-conditional
        inter=False,    # Interpolation
        weight=[10,10], # Weights for interpolation
        sample_id='default', # Sample ID
        total_cells=10000, # Total cells to generate
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_and_diffusion_defaults())
    defaults['num_class']=12
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add cell ratios argument
    parser.add_argument('--cell_ratios', type=str, help='Comma separated list of cell type ratios in the format celltype:ratio,celltype:ratio')
    parser.add_argument('--cell_ratios_file', type=str, help='Path to the CSV/TSV file containing cell type proportions')
    
    return parser
    
if __name__ == "__main__":
    args = create_argparser().parse_args()
    
    # Load cell type mapping dynamically from file
    if args.cell_ratios_file:
        cell_type_to_index = load_cell_type_mapping(args.cell_ratios_file)
    else:
        # Fallback to hardcoded mapping if no file provided
        cell_type_to_index = {
            "ABS": 0, "ASC": 1, "B": 2, "CT": 3, "EE": 4, "END": 5,
            "FIB": 6, "GOB": 7, "MAS": 8, "MYE": 9, "PLA": 10, "SSC": 11,
            "STM": 12, "T": 13, "TAC": 14, "TUF": 15
        }
    
    try:
        # Check if cell ratios are provided as command line argument
        if args.cell_ratios:
            cell_ratios = {}
            for ratio_pair in args.cell_ratios.split(','):
                if ':' in ratio_pair:
                    cell_name, ratio = ratio_pair.split(':')
                    cell_ratios[cell_name] = float(ratio)
            
            # Generate cells based on the specified ratios
            for cell_name, ratio in tqdm(cell_ratios.items(),total=len(cell_ratios)):
                num_samples = int(ratio * args.total_cells)
                if num_samples >= 2:  # Only generate if we have at least 2 cells
                    cell_index = cell_type_to_index.get(cell_name)
                    if cell_index is not None:
                        
                        # Update args with current cell type and number of samples
                        args.cell_type = [cell_index]
                        args.num_samples = num_samples
                        
                        # Call the main function with the updated arguments
                        main(args)
                else:
                    print(f"Skipping {cell_name} as it requires less than 2 cells")
        else:
            # If no cell ratios are provided, use the default cell type and number of samples
            main(args)

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()