print("in python")
import os
import math
from itertools import product, combinations
from functools import partial
from collections import namedtuple
from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg, autocast
from torch.nn import Parameter, Dropout
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import _LRScheduler
from torch import distributed as dist

from sklearn.metrics import roc_auc_score

from family_split_validation_loader import SplitDNASequencesDataset

from mamba_ssm import Mamba
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from caduceus.caduceus.configuration_caduceus import CaduceusConfig
from caduceus.caduceus.modeling_caduceus import CaduceusEmbeddings, RCPSAddNormWrapper, create_block

# Magic numbers
model_width = 1536
num_layers = 8
peak_lr = 3 * 6e-4
min_lr = 2e-4
local_batch_size_train = 128
warmup_steps = 10_000
total_steps = 150_000
weight_decay = 5e-2


def setup(rank, world_size):
    print("Rank", rank, "of", world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

# Return the original iterable if not root node, or tqdm wrapped it  we are root node, so we only print one progress bar
def maybe_tqdm(iterable):
    if rank == 0:
        return tqdm(iterable,ncols=80)
    return iterable  
    

# I did not write this, it is from stock CurricularFace, but I'm not sure if there's a reason here for what would be typically considered bad practice w.r.t. norm
def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, m=0.0, s=30.0):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.register_buffer('t', torch.zeros(1))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, label, step=10000):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm).clamp(-1, 1)  # For numerical stability
        if label is None:
            return cos_theta * self.s, cos_theta * self.s

        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target + margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] = hard_example * (self.t + hard_example)
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s



class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""
    def __init__(
            self,
            d_model: int,
            bidirectional: bool = True,
            bidirectional_strategy: Optional[str] = "add",
            bidirectional_weight_tie: bool = True,
            **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!")
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(
            d_model=d_model,
            **mamba_kwargs
        )
        if bidirectional:
            self.mamba_rev = Mamba(
                d_model=d_model,
                **mamba_kwargs
            )
            if bidirectional_weight_tie:  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None,mask = None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        # print("here abcd")
        out = self.mamba_fwd(hidden_states, mask= mask, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(dims=(1,)),  # Flip along the sequence length dimension
                mask=mask.flip(dims=(1,)) if mask is not None else None,
                inference_params=inference_params
            ).flip(dims=(1,))  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(f"`{self.bidirectional_strategy}` for bi-directionality not implemented!")
        return out



class CaduceusMixerModel(nn.Module):
    def __init__(
            self,
            config: CaduceusConfig,
            device=None,
            dmodel=2048,
            dtype=None,
            dropout_rate=0.0,
            exclude_embeddings=False
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.fused_add_norm = config.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32
        if exclude_embeddings == False:
            self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)
        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        # print("debugme")
        self.layers = nn.ModuleList(
            [
                create_block(
                    dmodel,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )
        self.dropout = Dropout(dropout_rate)
        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = norm_f if (config.fused_add_norm or not config.rcps) else RCPSAddNormWrapper(norm_f)
    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False, seq_lengths=None):
        """Mixer forward."""
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)
        residual = None
            # Create a mask based on sequence lengths if provided
        if seq_lengths is not None:
            max_len = hidden_states.size(1)
            range_tensor = torch.arange(max_len).expand(len(seq_lengths), max_len).to(seq_lengths.device)
            mask = (range_tensor < seq_lengths.unsqueeze(1)).float()
            # Apply mask: set padded positions to zero
            mask_unsq=  mask.unsqueeze(-1)
            hidden_states *= mask_unsq
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
                # TODO: Add support for gradient checkpointing
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None,mask=mask
                )
                hidden_states *= mask_unsq
                hidden_states= self.dropout(hidden_states)
        else:
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states.append(hidden_states)
                # TODO: Add support for gradient checkpointing
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None
                )
                hidden_states= self.dropout(hidden_states)
        if not self.fused_add_norm:
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states = self.norm_f(hidden_states, residual=residual, prenorm=False)
            else:
                residual = (hidden_states + residual) if residual is not None else hidden_states
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., :hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., :hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2:].flip(dims=[-2, -1]),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat([hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1)
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            # hidden_states = self.dropout(hidden_states)
        return hidden_states, all_hidden_states



# the model 
class Medusa(nn.Module):
    def __init__(self, d_model,n_classes,embed_dim=1536):
        super(Medusa, self).__init__()
        self.backbone = CaduceusMixerModel(CaduceusConfig(d_model=embed_dim, n_layer=num_layers, vocab_size=1024, rcps=False, bidirectional=True, rms_norm=False, fused_add_norm=True,bidirectional_weight_tie=False),dropout_rate=0.0,dmodel=embed_dim)
        self.arcface = CurricularFace(embed_dim, n_classes,s=30.0,m=0.0)
        # self.weight = Parameter(torch.FloatTensor(n_classes, embed_dim))
        # nn.init.xavier_uniform_(self.weight) 
    def forward(self, x, lens=None,step=10000):
        x, y = x
        x = x.to(torch.long)

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            x, _ = self.backbone(x,seq_lengths=lens)
            # print(x.dtype)
        x = x.float()
        x = torch.sum(x, dim=1).squeeze()

        normed_embeddings = F.normalize(x)
        # out =F.linear(normed_embeddings,F.normalize(self.weight))
        out, out2 = self.arcface(x, y,step=step)
        # out *= 30.0
        # log softmax + nll loss is more stable than softmax + cross entropy, they're the same thing
        return F.log_softmax(out, dim=1,dtype=torch.float32), F.log_softmax(out, dim=1,dtype=torch.float32), normed_embeddings



# linear warmup then cosine decay
class WarmupCosineDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1, min_lr=0.0002):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [ base_lr * float(self.last_epoch) / float(max(1, self.warmup_steps)) for base_lr in self.base_lrs]
        else:
            cos_decay = 0.5 * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
            decayed = (1 - self.min_lr) * cos_decay + self.min_lr
            return [max(base_lr * decayed, self.min_lr) for base_lr in self.base_lrs]

# Function for averaging metrics which i abuse
def average_metrics(mega_batch_loss, num_train, mega_batch_correct, i, mega_batch_val_loss, device):
    metrics = torch.tensor([
        mega_batch_loss / num_train,
        mega_batch_correct / num_train,
        mega_batch_val_loss / i,
    ], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    world_size = dist.get_world_size()
    metrics /= world_size
    return metrics

def save_checkpoint(step, current_mega_batch_index, rank, model, optimizer, scaler, scheduler, loss_values, accuracy_values, filename):
    if rank == 0:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'annealing_dict': scheduler.state_dict(),
            'loss_values': loss_values,
            'accuracy_values': accuracy_values,
            'current_mega_batch_index': current_mega_batch_index,
            'step': step,
        }
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved at mega batch {current_mega_batch_index}")


rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

setup(rank, world_size)

# Dataset setup
datasets = SplitDNASequencesDataset(my_rank=rank, num_ranks=world_size)
data_loader, n_classes = datasets.train, datasets.train_nclasses

model = Medusa(model_width, n_classes).to(local_rank)
model = DDP(model, device_ids=[local_rank])

if rank == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Num training classes (genomes): {n_classes}")
    print(f'Total number of parameters: {total_params}')

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=weight_decay)
scheduler = WarmupCosineDecayScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps, min_lr=min_lr)

# Training state
step = 0
scaler = GradScaler()
loss_values = []
accuracy_values = []
current_mega_batch_index = 0

# Device setup
device = torch.device(f"cuda:{local_rank}")

# for debug variable i've tried clamp for numerical stability issues 
a_log_params = [param for name, param in model.named_parameters() if name.endswith("A_log")]


def main_subloop():
    global step
    batch_size = local_batch_size_train
    model.train()
    current_mega_batch_index = 0

    for _, mega_batch in zip(range(1), data_loader):
        current_mega_batch_index += 1

        if current_mega_batch_index % 20 == 19:
            save_checkpoint(step, current_mega_batch_index, rank, model, optimizer, scaler, scheduler, loss_values, accuracy_values, f'cross_validstep_{step}example.pt')

        if current_mega_batch_index % 8 == 7:
            save_checkpoint(step, current_mega_batch_index, rank, model, optimizer, scaler, scheduler, loss_values, accuracy_values, f'cross_valid_example.pt')

        (labels, padded_texts, attention_masks), _ = mega_batch
        dist.barrier()
        padded_texts = padded_texts.squeeze()[:, :512].contiguous()
        labels_indices = labels.squeeze().max(dim=1).indices
        num_mini_batches = len(padded_texts) // batch_size

        mega_batch_loss = 0.0

        num_train = 0
        dist.barrier()


        for i in maybe_tqdm(range(num_mini_batches)):
            step += 1

            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            mini_batch_padded_texts = padded_texts[start_idx:end_idx].to(device)
            mini_batch_labels_indices = labels_indices[start_idx:end_idx].to(device)
            dist.barrier()

            logits, _, _ = model((mini_batch_padded_texts, mini_batch_labels_indices), step=step)
            loss = F.nll_loss(logits, mini_batch_labels_indices)

            mega_batch_loss += loss.item() * logits.size(0)
            num_train += logits.size(0)

            scaler.scale(loss).backward()

            if (step + 1) % 2 == 0:
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                if i % 8 == 7 and rank == 0:
                    print(f"Gradient norm before clipping: {total_norm}")
                    print(optimizer.param_groups[0]['lr'])

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if i % 16 == 15 or i == num_mini_batches - 1:
                avg_metrics = average_metrics(mega_batch_loss, num_train, 0, i, mega_batch_loss, device)
                avg_mega_batch_loss, _, _ = avg_metrics

                if rank == 0:
                    print(f"\t step {step//2}/300000; {step/300000:.4f} Average Loss per Mega Batch: {avg_mega_batch_loss:.4f}")
                    with open('log_file.txt', 'a') as log_file:
                        log_file.write(f'{avg_mega_batch_loss:.4f}\t{step//2}\n')

        if step >= 150000:
            dist.destroy_process_group()
            exit()


def val_subloop():
    with torch.no_grad():
        model.eval()
        for _, mega_batch in zip(range(1), datasets.valid):
            dist.barrier()

            batch_size = 32
            (labels, padded_texts, attention_masks), _ = mega_batch
            padded_texts = padded_texts.squeeze()[:, :1536].contiguous()
            dist.barrier()

            labels_indices = labels.squeeze().max(dim=1).indices
            num_mini_batches = len(padded_texts) // (batch_size * 3)

            embeddings_list = []
            labels_list = []

            for i in maybe_tqdm(range(num_mini_batches)):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                mini_batch_padded_texts = padded_texts[start_idx:end_idx].to(device)
                mini_batch_labels_indices = labels_indices[start_idx:end_idx].to(device)
                dist.barrier()


                _, _, embeddings = model((mini_batch_padded_texts, None))
                embeddings_list.append(embeddings.detach().cpu())
                labels_list.append(mini_batch_labels_indices.detach().cpu())

            all_embeddings = torch.cat(embeddings_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            pairwise_distances = torch.cdist(all_embeddings, all_embeddings, p=2).flatten()

            pair_labels = torch.tensor([1 if all_labels[i] == all_labels[j] else 0 for i, j in combinations(range(len(all_labels)), 2)])
            upper_triangle_indices = torch.triu_indices(len(all_labels), len(all_labels), offset=1)
            pairwise_distances = pairwise_distances[upper_triangle_indices[0] * len(all_labels) + upper_triangle_indices[1]]

            auc = roc_auc_score(pair_labels.numpy(), -pairwise_distances.numpy())
            avg_metrics = average_metrics(auc, 1, 0, 1, auc, device)
            auc, _, _ = avg_metrics

            if rank == 0:
                print(f"AUC: {auc}")

                positive_distances = pairwise_distances[pair_labels == 1].numpy()
                negative_distances = pairwise_distances[pair_labels == 0].numpy()

                plt.figure(figsize=(10, 6))
                bins = np.linspace(0, max(pairwise_distances.max().item(), 2), 100)
                plt.hist(positive_distances, bins=bins, alpha=0.5, label='Positive Pairs (Same Species)', density=True)
                plt.hist(negative_distances, bins=bins, alpha=0.5, label='Negative Pairs (Different Species)', density=True)
                plt.xlabel('Pairwise Distance')
                plt.ylabel('Frequency (Density)')
                plt.title('Histogram of Pairwise Distances')

                plt.text(0.3, 0.05, f'AUC: {auc:.3f}', transform=plt.gca().transAxes,
                         verticalalignment='top', horizontalalignment='center',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                plt.legend()
                plt.tight_layout()
                plt.savefig('pairwise_distances_histogram.png')
                plt.close()

                print(f"Histogram saved as 'pairwise_distances_histogram.png'. Note: Only uses root node validation points for visualization")
                with open('AUC_log.tsv', 'a') as the_file:
                    the_file.write(f'{auc:.4f}\t{0:.4f}\t{step//2}\n')


while True:
    main_subloop()
    val_subloop()
