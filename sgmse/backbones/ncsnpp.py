# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file

from .ncsnpp_utils import layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np


from .shared import BackboneRegistry

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
FiLMResnetBlockDDPM = layerspp.JointResnetBlockDDPMpp
FiLMResnetBlockBigGAN = layerspp.JointResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

import torch
import gc

# 1. Giải phóng RAM & VRAM
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

@BackboneRegistry.register("ncsnpp")
class NCSNpp(nn.Module):
    """NCSN++ model, adapted from https://github.com/yang-song/score_sde repository"""

    @staticmethod
    def add_argparse_args(parser):
        # TODO: add additional arguments of constructor, if you wish to modify them.
        return parser

    def __init__(self,
        scale_by_sigma = True,
        nonlinearity = 'swish',
        nf = 128,
        ch_mult = (1, 1, 2, 2, 2, 2, 2),
        num_res_blocks = 2,
        attn_resolutions = (16,),
        resamp_with_conv = True,
        conditional = True,
        fir = True,
        fir_kernel = [1, 3, 3, 1],
        skip_rescale = True,
        resblock_type = 'biggan',
        progressive = 'output_skip',
        progressive_input = 'input_skip',
        progressive_combine = 'sum',
        init_scale = 0.,
        fourier_scale = 16,
        image_size = 256,
        embedding_type = 'fourier',
        dropout = .0,
        centered = True,
        spectogram_learning=False, 
        conditioning_dim = 512,
        conditioning_fusion = "additive",
        **unused_kwargs
    ):

        super().__init__()

        
        # self.vfeat_processing_order = unused_kwargs["vfeat_processing_order"] # "default" #
        # self.audio_only = unused_kwargs["audio_only"] #True #

        #in this model we consider just one configuration
        # assert (self.audio_only and self.vfeat_processing_order == "default" and not joint_noise_clean_speech_training)


        self.act = act = get_act(nonlinearity)

        self.nf = nf = nf
        ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions = attn_resolutions
        dropout = dropout
        resamp_with_conv = resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [image_size // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = conditional  # noise-conditional
        self.centered = centered
        self.scale_by_sigma = scale_by_sigma

        fir = fir
        fir_kernel = fir_kernel
        self.skip_rescale = skip_rescale = skip_rescale
        self.resblock_type = resblock_type = resblock_type.lower()
        self.progressive = progressive = progressive.lower()
        self.progressive_input = progressive_input = progressive_input.lower()
        self.embedding_type = embedding_type = embedding_type.lower()
        self.conditioning_fusion = conditioning_fusion = conditioning_fusion.lower()


        init_scale = init_scale
        assert progressive in ['none', 'output_skip', 'residual']
        assert progressive_input in ['none', 'input_skip', 'residual']
        assert embedding_type in ['fourier', 'positional']
        assert conditioning_fusion in ['additive', 'film']
        combine_method = progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        self.spectogram_learning = spectogram_learning
        if self.spectogram_learning:
            num_channels=1
        else:            
            num_channels = 2  # x.real, x.imag
        
        self.output_layer = nn.Conv2d(num_channels, 2, 1)

        if conditioning_fusion == "film":
            self.text_embedding = nn.Sequential(
                nn.LayerNorm(conditioning_dim),
                nn.Linear(conditioning_dim, 4 * nf),
                nn.SiLU(),
                nn.Linear(4 * nf, 4 * nf),
            )
        else:
            self.text_embedding = nn.Linear(conditioning_dim, 4 * nf)

        modules = []
        # timestep/noise_level embedding
        if embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            modules.append(layerspp.GaussianFourierProjection(
                embedding_size=nf, scale=fourier_scale
            ))
            embed_dim = 2 * nf
        elif embedding_type == 'positional':
            embed_dim = nf
        else:
            raise ValueError(f'embedding type {embedding_type} unknown.')

        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp,
            init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample,
            with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive == 'output_skip':
            self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive == 'residual':
            pyramid_upsample = functools.partial(layerspp.Upsample, fir=fir,
                fir_kernel=fir_kernel, with_conv=True)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if progressive_input == 'input_skip':
            self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
        elif progressive_input == 'residual':
            pyramid_downsample = functools.partial(layerspp.Downsample,
                fir=fir, fir_kernel=fir_kernel, with_conv=True)

        if conditioning_fusion == "film":
            resnet_block_ddpm = FiLMResnetBlockDDPM
            resnet_block_biggan = FiLMResnetBlockBigGAN
            resnet_block_kwargs = dict(cemb_dim=nf * 4)
        else:
            resnet_block_ddpm = ResnetBlockDDPM
            resnet_block_biggan = ResnetBlockBigGAN
            resnet_block_kwargs = {}

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(resnet_block_ddpm, act=act,
                dropout=dropout, init_scale=init_scale,
                skip_rescale=skip_rescale, temb_dim=nf * 4,
                **resnet_block_kwargs)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(resnet_block_biggan, act=act,
                dropout=dropout, fir=fir, fir_kernel=fir_kernel,
                init_scale=init_scale, skip_rescale=skip_rescale, temb_dim=nf * 4,
                **resnet_block_kwargs)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block

        channels = num_channels
        if progressive_input != 'none':
            input_pyramid_ch = channels

        modules.append(conv3x3(channels, nf))


        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                if progressive_input == 'input_skip':
                    modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
                    if combine_method == 'cat':
                        in_ch *= 2

                elif progressive_input == 'residual':
                    modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
                    input_pyramid_ch = in_ch

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        pyramid_ch = 0
        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):  # +1 blocks in upsampling because of skip connection from combiner (after downsampling)
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if progressive != 'none':
                if i_level == num_resolutions - 1:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, in_ch, bias=True))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name.')
                else:
                    if progressive == 'output_skip':
                        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                            num_channels=in_ch, eps=1e-6))
                        modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
                        pyramid_ch = channels
                    elif progressive == 'residual':
                        modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
                        pyramid_ch = in_ch
                    else:
                        raise ValueError(f'{progressive} is not a valid name')

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        if progressive != 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                                                    num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--no-centered", dest="centered", action="store_false", help="The data is not centered [-1, 1]")
        parser.add_argument("--centered", dest="centered", action="store_true", help="The data is centered [-1, 1]")
        parser.add_argument("--embedding_type", type=str, default="fourier", choices=['fourier', 'positional','none'], help="Choose the type of embedding for t")                            
        parser.add_argument("--conditioning_dim", type=int, default=512, help="Dimension of the text embedding")
        parser.add_argument("--conditioning_fusion", type=str, default="additive", choices=['additive', 'film'], help="How to fuse text conditioning into the score network")
        parser.set_defaults(centered=True)
        return parser

    def _apply_resnet(self, module, x, temb, cemb):
        if self.conditioning_fusion == "film":
            return module(x, temb, cemb)
        return module(x, temb)

    def forward(self, x, time_cond, text_embed):
        
        # timestep/noise_level embedding; only for continuous training
        modules = self.all_modules
        m_idx = 0

        # Convert real and imaginary parts of x into two

        if self.spectogram_learning:
            x = x[:,[0],:,:]
        else: ## common case             
            x = torch.cat((x[:,[0],:,:].real, x[:,[0],:,:].imag), dim=1)

        # print(f"[Step 1: After Concatenation (Real+Imag)] Shape: {x.shape} | Mean: {x.mean().item():.4f} | Std: {x.std().item():.4f}")



        if self.embedding_type == 'fourier':
            # Gaussian Fourier features embeddings.
            used_sigmas = time_cond
            temb = modules[m_idx](torch.log(used_sigmas + 1e-5))
            m_idx += 1

        elif self.embedding_type == 'positional':
            # Sinusoidal positional embeddings.
            timesteps = time_cond
            used_sigmas = self.sigmas[time_cond.long()]
            temb = layers.get_timestep_embedding(timesteps, self.nf)

        else:
            raise ValueError(f'embedding type {self.embedding_type} unknown.')

        cemb = None
        if self.conditional:
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
            if self.conditioning_fusion == "film":
                cemb = self.text_embedding(text_embed)
            else:
                temb = temb + self.text_embedding(text_embed)
        else:
            temb = None

        if not self.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.
            # print(f"[Step 2: After Centering (2*x - 1)] Shape: {x.shape} | Mean: {x.mean().item():.4f} | Std: {x.std().item():.4f}")

        # Downsampling block
        input_pyramid = None
        if self.progressive_input != 'none':
            input_pyramid = x

        # Input layer: Conv2d: 4ch -> 128ch
        input_conv = modules[m_idx]
        x_first_conv = input_conv(x)
        m_idx += 1

        # print(f"[Step 3: After First Conv3x3 (Input Layer)] Shape: {x_first_conv.shape} | Mean: {x_first_conv.mean().item():.4f} | Std: {x_first_conv.std().item():.4f}")

        hs = [x_first_conv]
        # print(f"[Input Layer] Shape: {hs[-1].shape} | Mean: {hs[-1].mean().item():.4f} | Std: {hs[-1].std().item():.4f}")

        # Down path in U-Net
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = self._apply_resnet(modules[m_idx], hs[-1], temb, cemb)
                m_idx += 1
                # Attention layer (optional)
                if h.shape[-2] in self.attn_resolutions: # edit: check H dim (-2) not W dim (-1)
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)

                # print(f"[Down Path - Level {i_level}, Block {i_block}] Std: {h.std().item():.4f}")

            # Downsampling
            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = self._apply_resnet(modules[m_idx], hs[-1], temb, cemb)
                    m_idx += 1

                if self.progressive_input == 'input_skip':   # Combine h with x
                    input_pyramid = self.pyramid_downsample(input_pyramid)
                    h = modules[m_idx](input_pyramid, h)
                    m_idx += 1

                elif self.progressive_input == 'residual':
                    input_pyramid = modules[m_idx](input_pyramid)
                    m_idx += 1
                    if self.skip_rescale:
                        input_pyramid = (input_pyramid + h) / np.sqrt(2.)
                    else:
                        input_pyramid = input_pyramid + h
                    h = input_pyramid
                hs.append(h)

                # print(f"[Down Path - Level {i_level} Downsampled] Std: {h.std().item():.4f}")

        h = hs[-1] # actualy equal to: h = h
        h = self._apply_resnet(modules[m_idx], h, temb, cemb)  # ResNet block
        m_idx += 1
        h = modules[m_idx](h)  # Attention block
        m_idx += 1
        h = self._apply_resnet(modules[m_idx], h, temb, cemb)  # ResNet block
        m_idx += 1

        # print(f"[Middle Block] Shape: {h.shape} | Mean: {h.mean().item():.4f} | Std: {h.std().item():.4f}")

        pyramid = None

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self._apply_resnet(modules[m_idx], torch.cat([h, hs.pop()], dim=1), temb, cemb)
                m_idx += 1

                # print(f"[Up Path - Level {i_level}, Block {i_block}] Shape: {h.shape} | Mean: {h.mean().item():.4f} | Std: {h.std().item():.4f}")

            # edit: from -1 to -2
            if h.shape[-2] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if self.progressive != 'none':
                if i_level == self.num_resolutions - 1:
                    if self.progressive == 'output_skip':
                        pyramid = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)  # Conv2D: 256 -> 4
                        m_idx += 1
                    elif self.progressive == 'residual':
                        pyramid = self.act(modules[m_idx](h))
                        m_idx += 1
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name.')
                else:
                    if self.progressive == 'output_skip':
                        pyramid = self.pyramid_upsample(pyramid)  # Upsample
                        pyramid_h = self.act(modules[m_idx](h))  # GroupNorm
                        m_idx += 1
                        pyramid_h = modules[m_idx](pyramid_h)
                        m_idx += 1
                        pyramid = pyramid + pyramid_h
                    elif self.progressive == 'residual':
                        pyramid = modules[m_idx](pyramid)
                        m_idx += 1
                        if self.skip_rescale:
                            pyramid = (pyramid + h) / np.sqrt(2.)
                        else:
                            pyramid = pyramid + h
                        h = pyramid
                    else:
                        raise ValueError(f'{self.progressive} is not a valid name')

            # Upsampling Layer
            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = self._apply_resnet(modules[m_idx], h, temb, cemb)  # Upspampling
                    m_idx += 1

                # print(f"[Up Path - Level {i_level} Upsampled] Shape: {h.shape} | Mean: {h.mean().item():.4f} | Std: {h.std().item():.4f}")

        assert not hs

        if self.progressive == 'output_skip':
            h = pyramid
        else:
            h = self.act(modules[m_idx](h))
            m_idx += 1
            h = modules[m_idx](h)
            m_idx += 1

        assert m_idx == len(modules), "Implementation error"
        # if self.scale_by_sigma:
        #     used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
        #     h = h / used_sigmas

        # Convert back to complex number
        h = self.output_layer(h)

        # print(f"[Final Output - Before Complex] result: {h.shape} Mean: {h.mean().item():.4f} | Std: {h.std().item():.4f}")

        h = torch.permute(h, (0, 2, 3, 1)).contiguous()
        h = torch.view_as_complex(h)[:,None, :, :]
        return h



@BackboneRegistry.register("ncsnpp28M")
class NCSNpp28M(NCSNpp):
    """Tiny-scale NCSN++ model. ~28M parameters"""

    def __init__(self, **kwargs):
        super().__init__( 
        nf = 128,
        ch_mult = (1, 2, 2, 2),
        num_res_blocks = 1,
        attn_resolutions = (0,),
        **kwargs)

    # @staticmethod
    # def add_argparse_args(parser):
    #     # parser.add_argument("--centered", action="store_true", help="The data is already centered [-1, 1]")
    #     return parser
    
@BackboneRegistry.register("ncsnpp12M")
class NCSNpp12M(NCSNpp):
    """Small-scale NCSN++ model. ~12M parameters"""

    def __init__(self, **kwargs):
        super().__init__( 
        nf = 96,
        ch_mult = (1, 2, 2, 1),
        num_res_blocks = 1,
        attn_resolutions = (0,),
        **kwargs)

    # @staticmethod
    # def add_argparse_args(parser):
    #     # parser.add_argument("--centered", action="store_true", help="The data is already centered [-1, 1]")
    #     return parser



@BackboneRegistry.register("ncsnpp6M")
class NCSNpp6M(NCSNpp):
    """Tiny-scale NCSN++ model. ~6M parameters"""

    def __init__(self, **kwargs):
        super().__init__( 
        nf = 96,
        ch_mult = (1, 1, 1, 1),
        num_res_blocks = 1,
        attn_resolutions = (0,),
        **kwargs)


# if __name__ == '__main__':
#     # from thop import profile,clever_format
#     model = NCSNpp6M().cuda()        
#     # print(f"######### model.modules[0].W.device   {model.all_modules[0].W.device}")
#     b = torch.randn(4,1,256,256).cuda()  
#     t = torch.rand(4).cuda()
     
#     # c = model(b,t)

if __name__ == '__main__':
    model = NCSNpp6M(conditioning_fusion='film', skip_rescale=True, init_scale=1.0).cuda()
    model.eval() 
    
    # print(f"model.modules[0].W.device: {model.all_modules[0].W.device}")

    b = torch.randn(4, 1, 256, 256, dtype=torch.complex64).cuda()  
    
    t = torch.rand(4).cuda()
    
    text_embed = torch.randn(4, 512).cuda()

    # raw_text_embed = [-0.08040023595094681, 0.057447466999292374, -0.02831258624792099, -0.0004184916615486145, 0.05933468043804169, 0.04018446058034897, -0.04081360995769501, 0.03869902342557907, 0.04315011203289032, -0.032399192452430725, -0.08030378818511963, 0.0030512120574712753, -0.00757546816021204, 0.03422931581735611, 0.013614707626402378, 0.014134063385426998, 0.023477952927350998, 0.023654738441109657, 0.042137064039707184, -0.01940065436065197, -0.016667161136865616, 0.03500300645828247, 0.04395637661218643, -0.05236024409532547, 0.01019001193344593, 0.045322369784116745, -0.12801310420036316, 0.06711561977863312, 0.060322199016809464, 0.04552333056926727, -0.009186509996652603, -0.07008134573698044, 0.03322758898139, 0.09936493635177612, 0.025500480085611343, -0.004522486589848995, -0.1079634577035904, 0.03551434352993965, -0.015817582607269287, 0.004582958295941353, -0.08436524122953415, -0.03319847583770752, -0.05630750209093094, 0.12342856079339981, 0.018285222351551056, 0.04364124685525894, -0.08143605291843414, 0.07281117141246796, 0.037486813962459564, 0.06766241788864136, 0.0026498492807149887, -0.05784644931554794, -0.07458075881004333, 0.06772830337285995, 0.058848340064287186, -0.011950576677918434, 0.061942532658576965, 0.04150397703051567, -0.024278337135910988, -0.014799218624830246, 0.08410920947790146, 0.011794928461313248, -0.003437761217355728, -0.046625178307294846, -0.043660543859004974, 0.01568920910358429, 0.005149818956851959, -0.021507710218429565, 0.08989930152893066, 0.003484092652797699, 0.037512362003326416, 0.005375237204134464, -0.12400412559509277, -0.06893706321716309, 0.07120361924171448, -0.020061643794178963, -0.06585679203271866, -0.038615882396698, 0.020014218986034393, -0.06913834810256958, 0.020082255825400352, -0.056689172983169556, 0.10891193151473999, 0.051566530019044876, 0.025240983814001083, -0.02472490817308426, -0.04412355273962021, 0.07583778351545334, -0.02456383965909481, -0.06251947581768036, 0.07734320312738419, -0.07603839039802551, -0.06565555930137634, 0.02662743628025055, -0.0016214391216635704, 0.04915982484817505, -0.012136604636907578, -0.07165034115314484, -0.03888161852955818, -0.01573660410940647, -0.046045929193496704, -0.10203629732131958, -0.01816273108124733, -0.02243841625750065, 0.034321531653404236, -0.06192024052143097, -0.04367813840508461, 0.034533143043518066, -0.06014558672904968, 0.0026136599481105804, -0.017046334221959114, -0.015937045216560364, 0.03596048802137375, 0.027230005711317062, -0.07860496640205383, -0.039082981646060944, -0.013462942093610764, 0.040560346096754074, -0.023102859035134315, 0.07865281403064728, -0.009838948026299477, 0.0021646805107593536, -0.009368153288960457, -0.03680065646767616, -0.027264364063739777, 0.07475690543651581, 0.0581020824611187, 0.07024842500686646, 0.00437040813267231, 0.059266701340675354, -0.01564241759479046, -0.04686468839645386, 0.06225717067718506, -0.055988576263189316, 0.1224108338356018, 0.01298770122230053, 0.09460333734750748, 0.02576109953224659, 0.012953178957104683, 0.006441233679652214, -0.08377361297607422, -0.09087586402893066, 0.06638992577791214, -0.02213081158697605, -0.006368085741996765, -0.08169065415859222, -0.022630441933870316, 0.014853404834866524, 0.008056111633777618, 0.026661116629838943, 0.021689731627702713, 0.013760914094746113, -0.05892235413193703, 0.03909672796726227, 0.04078909009695053, 0.11089926213026047, 0.023070337250828743, -0.0028326809406280518, 0.019174017012119293, 0.03555763140320778, -0.07349595427513123, 0.02272816188633442, -0.04228546470403671, -0.06951162219047546, -0.05984384939074516, 0.036557648330926895, -0.1582576036453247, -0.038435064256191254, 0.09304988384246826, 0.008086496964097023, -0.0447244867682457, -0.008169617503881454, -0.013823218643665314, 0.017331158742308617, -0.04090636596083641, 0.03704225644469261, 0.0008397388737648726, 0.048670198768377304, 0.041056569665670395, 0.03020505979657173, 0.08406941592693329, -0.011844082735478878, 0.0028035533614456654, -0.1407316029071808, -0.07009772956371307, -0.11777928471565247, 0.016629653051495552, -0.034431494772434235, -0.01866999827325344, 0.17417417466640472, -0.021785739809274673, 0.041236747056245804, 0.015246778726577759, -0.015999864786863327, 0.017279701307415962, -0.04665732383728027, -0.008301853202283382, -0.054207995533943176, 0.008483413606882095, -0.026112716645002365, -0.06133532524108887, 0.03979272395372391, 0.013633758760988712, -0.0732453390955925, -0.033069636672735214, -0.08880683779716492, 0.006260262802243233, 0.04941561818122864, 0.12754833698272705, -0.025526195764541626, 0.03583584725856781, 0.0921851396560669, 0.014477819204330444, 0.05116929113864899, -0.049598224461078644, 0.03964180499315262, 0.027532286942005157, -0.02607187256217003, 0.02194373495876789, -0.019554676488041878, -0.07919462025165558, 0.034959301352500916, 0.0486539863049984, 0.057800907641649246, -0.01641562581062317, 0.018607962876558304, -0.03365154191851616, -0.058284029364585876, 0.03284958750009537, -0.013203401118516922, 0.009576828218996525, -0.12377415597438812, -0.050867944955825806, -0.07289408147335052, 0.018972255289554596, -0.027935262769460678, -0.112583227455616, 0.08002167195081711, -0.014226148836314678, -0.012694735080003738, -0.026826204732060432, -0.10337638854980469, 0.0877271369099617, -0.0837363451719284, 0.04545295611023903, -0.0019044578075408936, 0.026006603613495827, -0.0024472810328006744, 0.06499405950307846, -0.010442158207297325, -0.015074405819177628, 0.0213470421731472, -0.03461514413356781, 0.0809236392378807, 0.07175732403993607, -0.011258352547883987, 0.021053355187177658, -0.0015717744827270508, -0.003798488527536392, -0.019620224833488464, 0.03654513135552406, -0.07943888008594513, -0.026993239298462868, 0.055344607681035995, -0.04014935344457626, 0.07679085433483124, -0.01071515865623951, -0.02225477620959282, 0.02732730656862259, -0.019105833023786545, 0.058359697461128235, -0.019629424437880516, -0.009291741997003555, 0.03210853785276413, -0.053624629974365234, -0.04644926264882088, 0.06394273042678833, 0.00680818036198616, -0.03927464783191681, -0.010596159845590591, -0.005518745630979538, -0.06337137520313263, 0.026982873678207397, -0.0008781757205724716, -0.13791708648204803, 0.11009564250707626, 0.10301262140274048, 0.01673370599746704, -0.030678048729896545, -0.004075301811099052, 0.033481769263744354, -0.09224799275398254, 0.005178999621421099, -0.013182548806071281, -0.030458923429250717, -0.05910363048315048, -0.01649325340986252, -0.056717220693826675, 0.03595630079507828, -0.005051039159297943, -0.0033236220479011536, 0.011343812569975853, 0.05840667709708214, 0.025247802957892418, 0.07403503358364105, 0.02177252247929573, -0.06657366454601288, 0.06217075139284134, 0.08986590802669525, 0.003978086169809103, 0.011581409722566605, 0.013452076353132725, -0.1281176209449768, -0.07563986629247665, 0.050775978714227676, -0.0003319568932056427, 0.08581577241420746, -0.03336973115801811, 0.026924192905426025, -0.07088612765073776, 0.08593274652957916, 0.001984592527151108, 0.025595396757125854, -0.03535536676645279, 0.08559393137693405, -0.04905465990304947, 0.04605157673358917, 0.06870897114276886, -0.01736561208963394, 0.028819730505347252, 0.012758538126945496, 0.10164546221494675, -0.05997690185904503, 0.09768824279308319, 0.11550391465425491, -0.016634823754429817, 0.012861302122473717, 0.059433430433273315, 0.05194786563515663, 0.08408695459365845, -0.03634374588727951, 0.011101868003606796, -0.020486824214458466, 0.014411797747015953, 0.12033706903457642, 0.0014858152717351913, -0.10802627354860306, -0.01769145578145981, -0.036451272666454315, -0.08847382664680481, -0.0732002854347229, 0.11916851997375488, -0.03564605116844177, 0.03470064699649811, 0.03674528747797012, 0.03253554552793503, -0.05186858028173447, -0.02685665339231491, -0.11453315615653992, -0.02484450303018093, 0.01837458461523056, -0.009377137757837772, 0.07023480534553528, -0.0010255780071020126, 0.020583752542734146, 0.039505843073129654, -0.02421267144382, 0.0348285511136055, 0.02493078075349331, 0.008568849414587021, 0.11005626618862152, -0.03589325770735741, -0.047411855310201645, 0.09756411612033844, -0.05246521160006523, 0.03237361088395119, -0.006843212991952896, -0.10367883741855621, -0.07290822267532349, -0.015243293717503548, -0.04250999540090561, 0.06902378797531128, 0.023505693301558495, -0.008191775530576706, 0.06985503435134888, -0.04199519380927086, 0.04624468833208084, 0.0059172287583351135, 0.06820018589496613, 0.04695998877286911, 0.027814356610178947, 0.006189529784023762, 0.039359092712402344, -0.049078844487667084, -0.060510389506816864, -0.008005443960428238, -0.07584211975336075, 0.08491929620504379, 0.06798920780420303, 0.08191393315792084, 0.026168908923864365, -0.007315638475120068, -0.04464716464281082, -0.039102934300899506, 0.05645095556974411, -0.012999734841287136, -0.08778639137744904, -0.09081786870956421, 0.09607236087322235, -0.05665287747979164, -0.06941647827625275, -0.003082405775785446, 0.027182094752788544, -0.07058923691511154, 0.05884034186601639, -0.05497913807630539, 0.01000755000859499, 0.01843908801674843, -0.0016331039369106293, -0.021884366869926453, 0.010984692722558975, -0.030425727367401123, 0.008091753348708153, 0.05358750745654106, -0.03072376549243927, -0.010219857096672058, -0.03868510574102402, -0.09254543483257294, -0.041574470698833466, 0.10375307500362396, -0.031001422554254532, 0.0320618599653244, 0.06076908856630325, 0.06793560087680817, -0.03894628956913948, 0.054513558745384216, 0.06392155587673187, 0.021159548312425613, -0.047299738973379135, -0.07542800903320312, -0.08797785639762878, 0.12709194421768188, 0.019454380497336388, -0.03772842139005661, -0.05303943529725075, 0.0654684454202652, -0.041415680199861526, -0.020257573574781418, -0.07394418120384216, -0.013298338279128075, 0.09427912533283234, 0.04619956761598587, 0.045637279748916626, -0.031026851385831833, 0.10053002834320068, 0.014989227056503296, 0.016286034137010574, 0.042927585542201996, -0.06494252383708954, -0.010789708234369755, 0.08177873492240906, -0.002978488802909851, -0.061214201152324677, -0.07144784927368164, -0.015645133331418037, -0.053129494190216064, 0.052059728652238846, 0.03544703871011734, 0.02945661172270775, -0.1277265101671219, 0.04244266450405121, 0.09526848793029785, 0.0613941065967083, -0.06371650099754333, 0.05287543684244156, -0.048876844346523285, 0.014730416238307953, 0.07316315174102783, -0.04349437355995178, 0.046292513608932495, 0.003330104984343052, -0.05688760057091713, -0.03014238551259041, 0.06579506397247314, 0.03012028895318508, 0.08286101371049881, -0.0771823450922966, 0.0688408687710762, -0.06134970486164093, 0.04846442490816116, 0.049321725964546204, 0.07440948486328125, -0.005552939139306545, -0.06014801561832428, -0.065953329205513, 0.024856679141521454, 0.1189383789896965, 0.10179199278354645, -0.02422863245010376, -0.04670443385839462, -0.03999148681759834, 0.027438011020421982, 0.03851622715592384, 0.07584507763385773, -0.020768003538250923, 0.16634877026081085, 0.06702715903520584, 0.007957454770803452, -0.03574085980653763, 0.028430428355932236, 0.07323269546031952, 0.04955822974443436]
    # text_embed_tensor = torch.tensor(raw_text_embed, dtype=torch.float32)
    # text_embed = text_embed_tensor.unsqueeze(0).repeat(4, 1).cuda()
    # print(len(raw_text_embed))

    with torch.no_grad():
        c = model(b, t, text_embed)
