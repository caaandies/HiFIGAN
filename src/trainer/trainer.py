from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        gen_wavs = self.model.gen(batch["spectrogram"])
        real_wavs = batch["wav"]
        real_specs = batch["spectrogram"]

        # MPD
        self.mpd_optimizer.zero_grad()
        mpd_real_outputs, _ = self.model.mpd(real_wavs)
        mpd_gen_outputs, _ = self.model.mpd(gen_wavs.detach())
        mpd_losses = self.criterion.mpd_loss(mpd_real_outputs, mpd_gen_outputs)
        batch["mpd_loss"] = mpd_losses["loss"]
        batch["mpd_adv_loss"] = mpd_losses["adv_loss"]
        if self.is_train:
            batch["mpd_loss"].backward()
            self._clip_grad_norm(self.model.mpd)
            self.mpd_optimizer.step()
            if self.mpd_lr_scheduler is not None:
                self.mpd_lr_scheduler.step()

        # MSD
        self.msd_optimizer.zero_grad()
        msd_real_outputs, _ = self.model.msd(real_wavs)
        msd_gen_outputs, _ = self.model.msd(gen_wavs.detach())
        msd_losses = self.criterion.msd_loss(msd_real_outputs, msd_gen_outputs)
        batch["msd_loss"] = msd_losses["loss"]
        batch["msd_adv_loss"] = msd_losses["adv_loss"]
        if self.is_train:
            batch["msd_loss"].backward()
            self._clip_grad_norm(self.model.msd)
            self.msd_optimizer.step()
            if self.msd_lr_scheduler is not None:
                self.msd_lr_scheduler.step()

        # GEN
        self.gen_optimizer.zero_grad()

        _, mpd_real_features = self.model.mpd(real_wavs)
        mpd_gen_outputs, mpd_gen_features = self.model.mpd(gen_wavs)
        gen_mpd_losses = self.criterion.gen_loss(
            mpd_real_features, real_specs, mpd_gen_outputs, mpd_gen_features, gen_wavs
        )

        _, msd_real_features = self.model.msd(real_wavs)
        msd_gen_outputs, msd_gen_features = self.model.msd(gen_wavs)
        gen_msd_losses = self.criterion.gen_loss(
            msd_real_features, real_specs, msd_gen_outputs, msd_gen_features, gen_wavs
        )

        batch["gen_mpd_loss"] = gen_mpd_losses["loss"]
        batch["gen_mpd_adv_loss"] = gen_mpd_losses["adv_loss"]
        batch["gen_mpd_fm_loss"] = gen_mpd_losses["fm_loss"]
        batch["gen_mpd_mel_loss"] = gen_mpd_losses["mel_loss"]

        batch["gen_msd_loss"] = gen_msd_losses["loss"]
        batch["gen_msd_adv_loss"] = gen_msd_losses["adv_loss"]
        batch["gen_msd_fm_loss"] = gen_msd_losses["fm_loss"]
        batch["gen_msd_mel_loss"] = gen_msd_losses["mel_loss"]

        batch["gen_loss"] = gen_mpd_losses["loss"] + gen_msd_losses["loss"]

        if self.is_train:
            batch["gen_loss"].backward()
            self._clip_grad_norm(self.model.gen)
            self.gen_optimizer.step()
            if self.gen_lr_scheduler is not None:
                self.gen_lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
