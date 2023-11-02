import math
import copy
from random import random
from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from torch_geometric.data import Data as geoData
from torch_geometric.data import DataLoader
from torch_geometric.nn import RGCNConv
from torch.nn.init import xavier_normal_

from einops import rearrange, reduce

from sa_building_block import Building_Block
import math


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

# deterministic feed forward neural network
class DeterministicFeedForwardNeuralNetwork(nn.Module):

    def __init__(self, dim_in, dim_out, hid_layers=[100,50],
                 use_batchnorm=False, negative_slope=0.01, dropout_rate=0):
        super(DeterministicFeedForwardNeuralNetwork, self).__init__()
        self.dim_in = dim_in  # dimension of nn input
        self.dim_out = dim_out  # dimension of nn output
        self.hid_layers = hid_layers  # nn hidden layer architecture
        self.nn_layers = [self.dim_in] + self.hid_layers  # nn hidden layer architecture, except output layer
        self.use_batchnorm = use_batchnorm  # whether apply batch norm
        self.negative_slope = negative_slope  # negative slope for LeakyReLU
        self.dropout_rate = dropout_rate
        layers = self.create_nn_layers()
        self.network = nn.Sequential(*layers)

    def create_nn_layers(self):
        layers = []
        for idx in range(len(self.nn_layers) - 1):
            layers.append(nn.Linear(self.nn_layers[idx], self.nn_layers[idx + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(self.nn_layers[idx + 1]))
            layers.append(nn.LeakyReLU(negative_slope=self.negative_slope))
            layers.append(nn.Dropout(p=self.dropout_rate))
        layers.append(nn.Linear(self.nn_layers[-1], self.dim_out))
        return layers

    def forward(self, x):
        return self.network(x)


# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z): # (bs*nreg*nhour)*2*dim
        w = self.project(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)
   

class KGFlowBlock(nn.Module):
    def __init__(self, dim, nr, nhour, cond_dim, kwargs, kgedim):
        super().__init__()
        self.dim = dim
        self.num_rgcns = kwargs['num_rgcns']
        self.num_sas = kwargs['num_sas']
        self.cond_dim = cond_dim
        self.kgedim = kgedim

        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(kwargs['time_dim'], dim * nhour)
        )

        self.kge_projection = nn.Conv1d(kgedim, dim, 1)
        self.conditioner_projection = nn.Conv1d(cond_dim, 2 * dim, 1)
        
        self.flowrgcns = nn.ModuleList()
        for _ in range(kwargs['num_flowrgcns']):
            self.flowrgcns.append(RGCNConv(dim, dim, nr))

        self.blocks = nn.ModuleList()
        for _ in range(kwargs['num_sas']):
            self.blocks.append(Building_Block(dim, kwargs['num_heads'], kwargs['dropout']))

        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=4)
        
        self.middle_projection = nn.Conv1d(dim, 2 * dim, 1)
        
        self.output_projection = nn.Conv1d(dim, 2 * dim, 1)

    def forward(self, x_in, flowkg, time_emb, cond, KGE): # x:bs*nreg*nhour*trans_dim
        bs, nreg, nhour, dim = x_in.shape
        assert self.dim == dim
        x = x_in

        time_emb = time_emb.reshape(bs, -1) # bs*48
        time_emb = self.mlp(time_emb) # bs*(nhour*dim)
        time_emb = time_emb.reshape(bs, nhour, dim)
        time_emb = time_emb[:, None, :, :] # bs,1,nhour,dim
        x = x + time_emb

        # RGCN
        xnew = torch.tensor([], device = x.device)
        for i in range(bs):
            tmpE = x[i] # nreg*nhour*(...)
            allhour = []
            for j in range(nhour):
                E_hour = tmpE[:, j, :] # nreg*d
                tmpdata = geoData(x=E_hour, edge_index=flowkg.edge_index, edge_type=flowkg.edge_type)
                allhour.append(tmpdata)
            tmploader = DataLoader(allhour, batch_size=len(allhour))
            for batch in tmploader:
                for k in range(len(self.flowrgcns)):
                    xbatch = self.flowrgcns[k](batch.x, edge_index = batch.edge_index, edge_type = batch.edge_type) # (nhour*nreg)*d
                    # xbatch = torch.tanh(xbatch)
                    xbatch = F.relu(xbatch)
            xbatch = xbatch.reshape(nhour, nreg, -1) # nhour*nreg*(...)
            xbatch = xbatch.permute(1,0,2) # nreg*nhour*(...)
            xbatch = xbatch[None, :, :, :] 
            xnew = torch.cat((xnew, xbatch), dim=0) # bs*nreg*nhour*d
        out_spatial = xnew.reshape(bs*nreg, nhour, -1)

        # temporal
        out_temporal = x.reshape(bs*nreg, nhour, -1) 

        # transformer
        for i in range(self.num_sas):
            out_temporal = self.blocks[i](out_temporal) # (bs*nreg)*nhour*dim

        # KGE: nreg*kgedim
        emb = KGE[None,:,:,None].repeat(bs,1,1,1).reshape(bs*nreg, self.kgedim, 1)
        emb = self.kge_projection(emb) # (bs*nreg)*dim*1
        emb = emb.repeat(1,1,nhour).permute(0,2,1) # (bs*nreg)*nhour*d
        
        # fusion
        out_spatial = out_spatial.reshape(-1, dim) # (bs*nreg*nhour)*dim
        out_temporal = out_temporal.reshape(-1, dim) # (bs*nreg*nhour)*dim
        emb = emb.reshape(-1, dim) # (bs*nreg*nhour)*dim
        out = torch.stack([out_spatial, out_temporal], dim=1) # (bs*nreg*nhour)*2*dim

        # Q=KGE, K=V=[spatial;temporal]
        key = out.permute(1,0,2) # 2*(bs*nreg*nhour)*dim
        value = out.permute(1,0,2) # 2*(bs*nreg*nhour)*dim
        query = emb[None,:,:] # 1*(bs*nreg*nhour)*dim

        out, weight = self.mha(query, key, value)
        out = out[0,:,:]

        # out = out_spatial + out_temporal + emb
        out = out.reshape(bs*nreg, nhour, dim)

        # cond:nreg*cond_dim
        cond = cond[None,:,:,None].repeat(bs,1,1,1).reshape(bs*nreg, self.cond_dim, 1)
        cond = self.conditioner_projection(cond) # (bs*nreg)*2dim*1

        out = out.permute(0,2,1) # (bs*nreg)*dim*nhour
        out = self.middle_projection(out) # (bs*nreg)*2dim*nhour
        out = out + cond  # (bs*nreg)*2dim*nhour

        gate, filter = torch.chunk(out, 2, dim=1) # (bs*nreg)*dim*nhour
        out = torch.sigmoid(gate) * torch.tanh(filter) # (bs*nreg)*dim*nhour

        out = self.output_projection(out) # (bs*nreg)*2d*nhour
        residual, skip = torch.chunk(out, 2, dim=1) # (bs*nreg)*d*nhour

        residual = residual.permute(0,2,1).reshape(bs, nreg, nhour, -1)
        skip = skip.permute(0,2,1).reshape(bs, nreg, nhour, -1)
        
        return (x_in + residual) / math.sqrt(2.0), skip


class KGFlow(nn.Module):
    def __init__(self, d, **kwargs):
        super(KGFlow, self).__init__()
        self.nreg = d.nreg
        ne = len(d.ent2id)
        nr = len(d.rel2id)
        self.num_rgcns = kwargs['num_rgcns']
        self.num_sas = kwargs['num_sas']
        self.flow_channels = len(d.train_data[0][0][0])

        fea_dim = d.features.shape[1]

        self.features = nn.Embedding.from_pretrained(torch.tensor(d.features, dtype = torch.float), freeze=True)

        # use pretrain KGE
        self.KGE=nn.Embedding.from_pretrained(torch.tensor(d.KGE_pretrain, dtype = torch.float), freeze=True)
        kgedim = d.KGE_pretrain.shape[1]
        
        scale_cat_dim = 16
        nhour = len(d.train_data[0][0])
        self.scale_mlp = nn.Linear(1, scale_cat_dim)

        # time embeddings
        dim = kwargs['dim']
        time_dim = len(d.train_data[0][0]) * self.flow_channels # 24*2
        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        kwargs['time_dim'] = time_dim
        
        
        kge_cat_dim = kwargs['kge_cat_dim']
        self.kge_mlp = nn.Linear(fea_dim, kge_cat_dim)

        # conv1d
        xt_cat_dim = kwargs['xt_cat_dim']
        self.input_projection = nn.Conv1d(self.flow_channels, xt_cat_dim, 1)

        cond_dim = scale_cat_dim + kge_cat_dim
        trans_dim = xt_cat_dim

        residual_layers = kwargs['n_layer']
        self.residual_layers = nn.ModuleList([
            KGFlowBlock(dim = trans_dim, nr = nr, nhour = nhour, 
                        cond_dim = cond_dim, kwargs = kwargs, kgedim = kgedim)
            for i in range(residual_layers)
        ])
        
        self.middle_projection = nn.Conv1d(trans_dim, trans_dim, 1)
        self.output_projection = nn.Conv1d(trans_dim, self.flow_channels, 1)
        

    def init(self):
        xavier_normal_(self.KGE.weight.data)

    def forward(self, x_in, regids, time, g, flowkg, predscale):
        t = self.time_mlp(time) # bs*time_dim
        bs, nreg, nhour, _ = x_in.shape
        
        # feature condition
        E = self.features.weight # ne*fea_dim
        E = E[regids] # nreg*fea_dim, only use region emb in KG
        E = self.kge_mlp(E) # nreg*kge_cat_dim
        
        # add scale condition
        cond = predscale # bs*nreg*nhour*2
        cond = cond[0,:,0,:1] # nreg*1
        cond = self.scale_mlp(cond) # nreg*scale_cat_dim
        E = torch.cat((E, cond), dim = 1) # nreg*(kge_cat_dim+scale_cat_dim)

        x = x_in.reshape(bs, nreg, -1) # bs*nreg*2nhour
        t = t[:,None,:] # bs*1*time_dim
        x = x.view(bs, nreg, -1, 2) # bs*nreg*nhour*2

        # input Conv1*1
        x = x.view(bs * nreg, -1, self.flow_channels) # (bs*nreg)*nhour*2
        x = x.permute(0, 2, 1) # (bs*nreg)*2*nhour
        x = self.input_projection(x) # (bs*nreg)*xt_cat_dim*nhour
        x = F.relu(x)
        x = x.permute(0, 2, 1) # (bs*nreg)*nhour*xt_cat_dim
        x = x.view(bs, nreg, nhour, -1) # bs*nreg*nhour*xt_cat_dim

        # KGE
        KGE = self.KGE.weight
        KGE = KGE[regids]

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, flowkg, t, E, KGE)
            skip = skip_connection if skip is None else skip_connection + skip

        out = skip / math.sqrt(len(self.residual_layers))

        out = x.reshape(bs*nreg, nhour, -1) # (bs*nreg)*nhour*dim
        out = out.permute(0, 2, 1) # (bs*nreg)*dim*nhour
        out = self.middle_projection(out) # (bs*nreg)*2*nhour
        out = F.relu(out)
        out = self.output_projection(out) # (bs*nreg)*2*nhour
        out = out.permute(0, 2, 1) # (bs*nreg)*nhour*2
        out = out.reshape(bs, nreg, nhour, -1)

        return out

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        cond_pred_model,
        d,
        data_shape, # nreg*nhour*2
        g,
        g_train,
        g_samp,
        image_size,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 1.
    ):
        super().__init__()
        
        # condition prediction model
        self.cond_pred_model = cond_pred_model

        self.model = model
        self.data_shape = data_shape
        self.g = g
        self.g_train = g_train
        self.g_samp = g_samp
        self.self_condition = False

        self.image_size = image_size #128

        self.objective = objective # pred_noise

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.) # [1.]+[alphas_cumprod[:-1]]

        timesteps, = betas.shape # 1000
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32)) # 不会被算作model parameters

        self.register_buffer('scale', torch.tensor(d.scale, dtype = torch.float)) # nreg*1
        self.register_buffer('scale_pred_X', torch.tensor(d.scale_pred_X, dtype = torch.float)) # nreg*37

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        coef3 = 1. + (torch.sqrt(alphas_cumprod) - 1.) * (torch.sqrt(alphas) + torch.sqrt(alphas_cumprod_prev)) / (1. - alphas_cumprod)
        register_buffer('posterior_mean_coef3', coef3)

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise, f_phi):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise -
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) - 1) * f_phi
        )

    def predict_noise_from_start(self, x_t, t, x0, f_phi):
        tmp = extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        return (
            (tmp * x_t - x0 - (tmp - 1) * f_phi) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t, sampids):
        f_phi = self.compute_guiding_prediction(x_start, sampids)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef3, t, x_t.shape) * f_phi
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def compute_guiding_prediction(self, x_start, regids):
        bs, nreg, nh, c = x_start.shape

        X = self.scale_pred_X[regids]
        pred = self.cond_pred_model(X) # nregs*1
        x_T_mean = pred[None, :, :, None]
        x_T_mean = x_T_mean.repeat(bs, 1, nh, c)

        return x_T_mean

    def model_predictions(self, x, sampids, t, x_self_cond = None, clip_x_start = False, mode = 'train'):
        
        f_phi = self.compute_guiding_prediction(x, sampids)

        assert mode in ['train','sample']
        if mode == 'train':
            model_output = self.model(x, sampids, t, self.g, self.g_train, f_phi)
        elif mode == 'sample':
            model_output = self.model(x, sampids, t, self.g, self.g_samp, f_phi)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise, f_phi)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start, f_phi)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, sampids, t, x_self_cond = None, clip_denoised = True):
        preds = self.model_predictions(x, sampids, t, x_self_cond, mode = 'sample')
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t, sampids = sampids)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, sampids, t: int, x_self_cond = None, clip_denoised = True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long) # [t]*bs
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, sampids = sampids, t = batched_times, x_self_cond = x_self_cond, clip_denoised = clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise # x_{t-1}
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, sampids, shape): # shape=(bs, nreg, nhour, 2)
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        # noise sample with predicted mean
        x_T_mean = self.compute_guiding_prediction(img, sampids) # nreg*1
        assert x_T_mean.shape == shape
        img = img + x_T_mean

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, sampids, t, self_cond)

        return img

    @torch.no_grad()
    def sample(self, sampids, batch_size = 16):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(sampids, (batch_size, len(sampids), self.data_shape[1], self.data_shape[2]))


    def q_sample(self, x_start, x_T_mean, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            (1 - extract(self.sqrt_alphas_cumprod, t, x_start.shape)) * x_T_mean +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    
    def p_losses(self, x_start, regids, t, noise = None):
        bs, nreg, nh, c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x_T_mean = self.compute_guiding_prediction(x_start, regids) # bs*nreg*nhour*2
        assert x_T_mean.shape == x_start.shape

        x = self.q_sample(x_start = x_start, x_T_mean = x_T_mean, t = t, noise = noise) # shape=x_start.shape

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step

        model_out = self.model(x, regids, t, self.g, self.g_train, x_T_mean) # shape=x_start.shape

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none') # x_start.shape
        loss = reduce(loss, 'b ... -> b (...)', 'mean') # bs*-1

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, regids, *args, **kwargs):
        b, nreg, nh, c, device, img_size, = *img.shape, img.device, self.image_size
        assert len(regids) == nreg
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, regids, t, *args, **kwargs)

