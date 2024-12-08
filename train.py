import warnings

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.transforms import MelSpectrogram, MelSpectrogramConfig
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    # logger.info(model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_mpd_params = filter(lambda p: p.requires_grad, model.mpd.parameters())
    mpd_optimizer = instantiate(config.mpd_optimizer, params=trainable_mpd_params)

    trainable_msd_params = filter(lambda p: p.requires_grad, model.msd.parameters())
    msd_optimizer = instantiate(config.msd_optimizer, params=trainable_msd_params)

    trainable_gen_params = filter(lambda p: p.requires_grad, model.gen.parameters())
    gen_optimizer = instantiate(config.gen_optimizer, params=trainable_gen_params)

    mpd_lr_scheduler = instantiate(config.mpd_lr_scheduler, optimizer=mpd_optimizer)
    msd_lr_scheduler = instantiate(config.msd_lr_scheduler, optimizer=msd_optimizer)
    gen_lr_scheduler = instantiate(config.gen_lr_scheduler, optimizer=gen_optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    spec_transform = instantiate(config.spec_transform).to(device)

    trainer = Trainer(
        model=model,
        spec_transform=spec_transform,
        criterion=loss_function,
        metrics=metrics,
        mpd_optimizer=mpd_optimizer,
        msd_optimizer=msd_optimizer,
        gen_optimizer=gen_optimizer,
        mpd_lr_scheduler=mpd_lr_scheduler,
        msd_lr_scheduler=msd_lr_scheduler,
        gen_lr_scheduler=gen_lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        logger=logger,
        writer=writer,
        epoch_len=epoch_len,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
