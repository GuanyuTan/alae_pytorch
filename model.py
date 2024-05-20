import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from alae import *
import losses


class DLatent(nn.Module):
    # decay tracking moving average of W space during training
    def __init__(self, dlatent_size, layer_count):
        super(DLatent, self).__init__()
        buffer = torch.zeros(layer_count, dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class Model(nn.Module):
    def __init__(self,
                 startf=32,  #
                 maxf=256,  #
                 layer_count=3,  #
                 latent_size=128,  #
                 mapping_layers=5,  #
                 dlatent_avg_beta=None,
                 trucation_psi=None,
                 trucation_cutoff=None,
                 style_mixing_prob=None,
                 channels=3,  # number of color channels
                 generator="",  # type of generator, get from config
                 encoder="",  # type of encoder, get from config
                 # regression of z space instead of w space.
                 z_regression=False
                 ) -> None:
        """_summary_

        Args:
            startf (int, optional): Starting feature maps. Defaults to 32.
            maxf (int, optional): Maximum feature maps. Defaults to 256.
            layer_count (int, optional):Number of layers for generator and encoder, note that number of layers must be equivalent to the maximum level of detail. Defaults to 3.
            latent_size (int, optional): Latent size. Defaults to 128.
            mapping_layers (int, optional): Mapping layers for mapping_f and mapping_d. Defaults to 5.
            trucation_psi (_type_, optional): _description_. Defaults to None.
            trucation_cutoff (_type_, optional): _description_. Defaults to None.
            style_mixing_prob (_type_, optional): _description_. Defaults to None.
            channels (int, optional): Number of color channels. Defaults to 3.
            generator (str, optional): Generator type. Defaults to "".
            encoder (str, optional): Encoder type. Defaults to "".
            z_regression (bool, optional): Regression of z space and the encoder latent space. Defaults to False.
        """
        super(Model, self).__init__()
        self.layer_count = layer_count
        self.latent_size = latent_size
        self.startf = startf
        self.maxf = maxf
        self.channels = channels
        self.z_regression = z_regression

        self.mapping_f = MAPPINGS["MappingF"](

            # num_layers = 2* self.layer_count,
            # this is weird because it returns output of [batch_size, num_layers, latent_size]
            num_layers=self.layer_count*2,
            latent_size=self.latent_size,
            dlatent_size=self.latent_size,
            mapping_fmaps=self.latent_size,
            mapping_layers=mapping_layers,
        )
        self.dlatent_avg = DLatent(latent_size, self.mapping_f.num_layers)
        self.dlatent_avg_beta = dlatent_avg_beta
        self.trucation_cutoff = trucation_cutoff
        self.truncation_psi = trucation_psi
        self.style_mixing_prob = style_mixing_prob

        # TODO Check if num_layers needs to be removed or not.

        self.decoder = GENERATORS[generator](
            layer_count=self.layer_count,
            startf=self.startf,
            maxf=self.maxf,
            latent_size=self.latent_size,
            channels=self.channels,
        )
        self.encoder = ENCODERS[encoder](
            layer_count=self.layer_count,
            startf=self.startf,
            maxf=self.maxf,
            latent_size=self.latent_size,
            channels=self.channels,
        )
        self.mapping_d = MAPPINGS["MappingD"](
            latent_size=self.latent_size,
            dlatent_size=self.latent_size,
            mapping_fmaps=self.latent_size,
            mapping_layers=mapping_layers,
        )

    def generate(self,
                 lod,
                 blend_factor,
                 z=None,
                 count=32,  
                 mixing=False, 
                 noise=True, 
                 return_styles=False,
                 no_truncation=False):
        """_summary_

        Args:
            lod (int): Level of Detail
            blend_factor (float): Blend factor, ranges from [0,1], 1 is when the upper layer is fully incorporated, 0 is when upper layer is not yet trained.
            z (torch.Tensor, optional): Gaussian samples from z space. Defaults to None.
            count (int, optional): Batch size. Defaults to 32.
            mixing (bool, optional): Style mixing. Defaults to False.
            noise (bool, optional): Include noise in generator. Defaults to True.
            return_styles (bool, optional): Defaults to False.
            no_truncation (bool, optional): Don't do gaussian truncation for inference. Defaults to False.

        Returns:
            torch.Tensor: Returns reconstructed images with shape=[count, channels, 4*2^lod, 4*2^lod], styles if return_styles is True.
        """
        if z is None:
            z = torch.randn(count, self.latent_size)

        styles = self.mapping_f(z)[:, 0]
        s = styles.view(styles.shape[0], 1, styles.shape[1])

        styles = s.repeat(1, self.mapping_f.num_layers, 1)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = styles.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(
                    batch_avg.data, 1.0 - self.dlatent_avg_beta)

        if mixing and self.style_mixing_prob is not None:
            if random.random() < self.style_mixing_prob:
                z2 = torch.randn(count, self.latent_size)
                styles2 = self.mapping_f(z2)[:, 0]
                styles2 = styles2.view(styles2.shape[0], 1, styles2.shape[1]).repeat(
                    1, self.mapping_f.num_layers, 1)

                layer_idx = torch.arange(self.mapping_f.num_layers)[
                    np.newaxis, :, np.newaxis]
                cur_layers = (lod + 1) * 2
                mixing_cutoff = random.randint(1, cur_layers)
                styles = torch.where(
                    layer_idx < mixing_cutoff, styles, styles2)

        if (self.truncation_psi is not None) and not no_truncation:
            # Do truncation inference
            layer_idx = torch.arange(self.mapping_f.num_layers)[
                np.newaxis, :, np.newaxis]
            ones = torch.ones(layer_idx.shape, dtype=torch.float32)
            coefs = torch.where(
                layer_idx < self.truncation_cutoff, self.truncation_psi * ones, ones)
            styles = torch.lerp(self.dlatent_avg.buff.data, styles, coefs)

        rec = self.decoder(styles, lod, blend_factor, noise)
        if return_styles:
            return s, rec
        else:
            return rec

    def encode(self, x, lod, blend_factor):
        Z = self.encoder(x, lod, blend_factor)
        discriminator_prediction = self.mapping_d(Z)
        return Z[:, :1], discriminator_prediction

    def forward(self, x, lod, blend_factor, d_train, ae):
        batch_size = x.shape[0]

        if ae:
            self.encoder.requires_grad_(True)
            z = torch.randn(batch_size, self.latent_size)
            s, rec = self.generate(
                lod=lod,
                blend_factor=blend_factor,
                z=z,
                mixing=False,
                noise=True,
                return_styles=True
            )

            Z, d_result_fake = self.encode(rec, lod, blend_factor)
            assert Z.shape == s.shape

            if self.z_regression:
                Lae = torch.mean((Z[:0] - z)**2)
            else:
                Lae = torch.mean(((Z - s.detach())**2))
            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(
                    lod=lod,
                    blend_factor=blend_factor,
                    count=batch_size,
                    noise=True
                )
            self.encoder.requires_grad_(True)

            _, d_result_real = self.encode(
                x=x, lod=lod, blend_factor=blend_factor)
            _, d_result_fake = self.encode(
                x=Xp, lod=lod, blend_factor=blend_factor)

            loss_d = losses.discriminator_logistic_simple_gp(
                d_result_fake=d_result_fake,
                d_result_real=d_result_real,
                reals=x
            )
            return loss_d
        else:
            with torch.no_grad():
                z = torch.randn(batch_size, self.latent_size)

            self.encoder.requires_grad_(False)
            rec = self.generate(
                lod=lod,
                blend_factor=blend_factor,
                count=batch_size,
                # This should be uneccessary
                z=z.detach(),
                noise=True)

            _, d_result_fake = self.encode(
                x=rec,
                lod=lod,
                blend_factor=blend_factor)

            loss_g = losses.generator_logistic_non_saturating(
                d_result_fake=d_result_fake)

            return loss_g

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.mapping_d.parameters()) + list(self.mapping_f.parameters()) + list(
                self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_d.parameters()) + list(other.mapping_f.parameters()) + list(
                other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
