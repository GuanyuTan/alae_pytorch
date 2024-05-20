import torch
import os
from PIL import Image
from torchvision.utils import save_image
from model import Model
from custom_adam import LREQAdam
from scheduler import ComboMultiStepLR
from tracker import LossTracker
from checkpointer import Checkpointer
import lod_driver
from tqdm import tqdm
import numpy as np
from dataloader import *
import torch.nn.functional as F
from defaults import get_cfg_defaults
from launcher import run
import utils
import logging
from torch.utils.data import DataLoader

cfg = get_cfg_defaults()
logger = logging.getLogger()


def save_sample(lod2batch, tracker, sample, samplez, x, logger, model, cmodel, cfg, encoder_optimizer, decoder_optimizer):
    os.makedirs('results', exist_ok=True)
# Save Sample LOL
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info('\n[%d/%d] - ptime: %.2f, %s, blend: %.3f, lr: %.12f,  %.12f, max mem: %f",' % (
        (lod2batch.current_epoch +
         1), cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
        lod2batch.get_blend_factor(),
        encoder_optimizer.param_groups[0]['lr'], decoder_optimizer.param_groups[0]['lr'],
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    with torch.no_grad():
        model.eval()
        cmodel.eval()
        sample = sample[:lod2batch.get_per_GPU_batch_size()]
        samplez = samplez[:lod2batch.get_per_GPU_batch_size()]

        needed_resolution = model.decoder.layer_to_resolution[lod2batch.lod]
        sample_in = sample
        while sample_in.shape[2] > needed_resolution:
            sample_in = F.avg_pool2d(sample_in, 2, 2)
        assert sample_in.shape[2] == needed_resolution

        blend_factor = lod2batch.get_blend_factor()
        if lod2batch.in_transition:
            needed_resolution_prev = model.decoder.layer_to_resolution[lod2batch.lod - 1]
            sample_in_prev = F.avg_pool2d(sample_in, 2, 2)
            sample_in_prev_2x = F.interpolate(
                sample_in_prev, needed_resolution)
            sample_in = sample_in * blend_factor + \
                sample_in_prev_2x * (1.0 - blend_factor)

        sample_in = sample_in.to(device)
        Z, _ = model.encode(sample_in, lod2batch.lod, blend_factor)

        if cfg.MODEL.Z_REGRESSION:
            Z = model.mapping_f(Z[:, 0])
        else:
            Z = Z.repeat(1, model.mapping_f.num_layers, 1)

        rec1 = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)
        rec2 = cmodel.decoder(Z, lod2batch.lod, blend_factor, noise=True)

        Z = model.mapping_f(samplez)
        g_rec = model.decoder(Z, lod2batch.lod, blend_factor, noise=True)

        Z = cmodel.mapping_f(samplez)
        cg_rec = cmodel.decoder(Z, lod2batch.lod, blend_factor, noise=True)

        resultsample = torch.cat([sample_in, rec1, rec2, g_rec, cg_rec], dim=0)

        @utils.async_func
        def save_pic(x_rec):
            tracker.register_means(
                lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
            tracker.plot()

            result_sample = x_rec * 0.5 + 0.5
            result_sample = result_sample.cpu()
            f = os.path.join(cfg.OUTPUT_DIR,
                             'sample_%d_%d.jpg' % (
                                 lod2batch.current_epoch + 1,
                                 lod2batch.iteration // 1000))
            print("Saved to %s" % f)
            save_image(result_sample, f, nrow=min(
                32, lod2batch.get_per_GPU_batch_size()))
        save_pic(resultsample)


def train(cfg, logger):
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        z_regression=cfg.MODEL.Z_REGRESSION
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device=device)
    print(next(model.parameters()).is_cuda)
    decoder = model.decoder
    encoder = model.encoder
    mapping_d = model.mapping_d
    mapping_f = model.mapping_f
    dlatent_avg = model.dlatent_avg

    arguments = dict()
    arguments["iteration"] = 0

    decoder_optim = LREQAdam(
        [
            {'params': decoder.parameters()},
            {'params': mapping_f.parameters()}
        ],
        lr=cfg.TRAIN.BASE_LEARNING_RATE,
        betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
        weight_decay=0
    )
    encoder_optim = LREQAdam(
        [
            {'params': encoder.parameters()},
            {'params': mapping_d.parameters()}
        ],
        lr=cfg.TRAIN.BASE_LEARNING_RATE,
        betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1),
        weight_decay=0
    )

    scheduler = ComboMultiStepLR(
        optimizers={
            'encoder_optimizer': encoder_optim,
            'decoder_optimizer': decoder_optim
        },
        milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
        gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
        reference_batch_size=32,
        base_lr=cfg.TRAIN.LEARNING_RATES
    )

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_d,
        'mapping_fl': mapping_f,
        'dlatent_avg': dlatent_avg
    }

    tracker = LossTracker(
        cfg.OUTPUT_DIR
    )

    checkpointer = Checkpointer(cfg,
                                model_dict,
                                {
                                    'encoder_optimizer': encoder_optim,
                                    'decoder_optimizer': decoder_optim,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=True)

    extra_checkpoint_data = checkpointer.load()
    logger.info(f'Starting from epoch: {scheduler.start_epoch()}')

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = decoder.layer_to_resolution

    dataset = CovidTfRecordDataset(cfg, logger)

    rnd = np.random.RandomState(0)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.Tensor(latents).float().cuda()

    lod2batch = lod_driver.LODDriver(
        cfg,
        logger,
        world_size=1,
        dataset_size=len(dataset)
    )

    if cfg.DATASET.SAMPLES_PATH != 'no_path':
        path = cfg.DATASET.SAMPLES_PATH
        src = []
        with torch.no_grad():
            for filename in list(os.listdir(path))[:32]:
                img = np.asarray(Image.open(os.path.join(path, filename)))
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                im = img.transpose((2, 0, 1))
                x = torch.Tensor(np.asarray(
                    im, dtype=np.float32), requires_grad=True).cuda() / 127.5 - 1.
                if x.shape[0] == 4:
                    x = x[:3]
                src.append(x)
                sample = torch.stack(src)
    else:
        dataset.reset(cfg.DATASET.MAX_RESOLUTION_LEVEL)
        data_batch = next(iter(DataLoader(
            dataset=dataset,
            batch_size=lod2batch.get_per_GPU_batch_size()
        )))
        img_size = 2**cfg.DATASET.MAX_RESOLUTION_LEVEL
        sample = data_batch['data']
        sample = torch.cat([torch.frombuffer(i, dtype=torch.uint8).reshape(
            1, 1, img_size, img_size) for i in sample], dim=0)
        sample = (sample / 127.5 - 1.)
    lod2batch.set_epoch(scheduler.start_epoch(), [
                        encoder_optim, decoder_optim])

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [encoder_optim, decoder_optim])

        logger.info("Batch size: %d, Batch size per GPU: %d, LOD: %d - %dx%d, blend: %.3f, dataset size: %d" % (
            lod2batch.get_batch_size(),
            lod2batch.get_per_GPU_batch_size(),
            lod2batch.lod,
            2 ** lod2batch.get_lod_power2(),
            2 ** lod2batch.get_lod_power2(),
            lod2batch.get_blend_factor(),
            len(dataset)))
        img_size = 2 ** lod2batch.get_lod_power2()
        dataset.reset(lod2batch.get_lod_power2())
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=lod2batch.get_per_GPU_batch_size()
        )

        scheduler.set_batch_size(lod2batch.get_batch_size(), lod2batch.lod)

        model.train()

        need_permute = False
        epoch_start_time = time.time()

        i = 0
        for data in tqdm(dataloader):
            x_orig = torch.from_numpy(
                np.concatenate([
                    np.frombuffer(
                        i,
                        dtype=np.uint8).reshape(1, cfg.MODEL.CHANNELS, img_size, img_size)
                    for i in data['data']],
                    axis=0))
            x_orig = x_orig.to(device=device)
            i += 1
            with torch.no_grad():
                if x_orig.shape[0] != lod2batch.get_per_GPU_batch_size():
                    continue
                if need_permute:
                    x_orig = x_orig.permute(0, 3, 1, 2)
                x_orig = (x_orig / 127.5 - 1.)

                blend_factor = lod2batch.get_blend_factor()

                needed_resolution = layer_to_resolution[lod2batch.lod]
                x = x_orig

                if lod2batch.in_transition:
                    needed_resolution_prev = layer_to_resolution[lod2batch.lod - 1]
                    x_prev = F.avg_pool2d(x_orig, 2, 2)
                    x_prev_2x = F.interpolate(x_prev, needed_resolution)
                    x = x * blend_factor + x_prev_2x * (1.0 - blend_factor)

            x.requires_grad = True

            encoder_optim.zero_grad()
            loss_d = model(x, lod2batch.lod, blend_factor,
                           d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d))
            loss_d.backward()
            encoder_optim.step()

            decoder_optim.zero_grad()
            loss_g = model(x, lod2batch.lod, blend_factor,
                           d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g))
            loss_g.backward()
            decoder_optim.step()

            encoder_optim.zero_grad()
            decoder_optim.zero_grad()
            # this part is buggy, if ae=True and d_train=True, only ae logic will run.
            lae = model(x, lod2batch.lod, blend_factor, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            lae.backward()
            encoder_optim.step()
            decoder_optim.step()

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            lod_for_saving_model = lod2batch.lod
            lod2batch.step()
            if lod2batch.is_time_to_save():
                checkpointer.save(
                    "model_tmp_intermediate_lod%d" % lod_for_saving_model)
            if lod2batch.is_time_to_report():
                save_sample(lod2batch, tracker, sample, samplez, x, logger, model,
                            model.module if hasattr(
                                model, "module") else model, cfg, encoder_optim,
                            decoder_optim)

        scheduler.step()

        # if local_rank == 0:
        checkpointer.save("model_tmp_lod%d" % lod_for_saving_model)
        save_sample(lod2batch, tracker, sample, samplez, x, logger, model,
                    model.module if hasattr(model, "module") else model, cfg, encoder_optim, decoder_optim)

    logger.info("Training finish!... save training results")
    # if local_rank == 0:
    checkpointer.save("model_final").wait()


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='StyleGAN', default_config='configs/covid.yaml',
        world_size=gpu_count)
    