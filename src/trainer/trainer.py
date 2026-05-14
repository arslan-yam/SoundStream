from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

import torch
from torch.nn.utils import clip_grad_norm_


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    
    def __init__(self, *args, optimizer_d=None, lr_scheduler_d=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer_d = optimizer_d
        self.lr_scheduler_d = lr_scheduler_d
        self.sample_rate = self.config.get("sample_rate", 16000)


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
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        if self.is_train:
            self.optimizer_d.zero_grad()
            with torch.no_grad():
                gen_out = self.model(audio=batch["audio"])
                
            audio_pred_d = gen_out["audio_pred"].detach()
            real_outs, _ = self.model.discriminator(batch["audio"])
            fake_outs, _ = self.model.discriminator(audio_pred_d)
            loss_d = self.criterion.discriminator_loss(real_outs, fake_outs)
            loss_d.backward()
            if self.config["trainer"].get("max_grad_norm", None) is not None:
                clip_grad_norm_(self.model.discriminator.parameters(), self.config["trainer"]["max_grad_norm"])
            
            self.optimizer_d.step()
            self.optimizer.zero_grad()
            gen_out = self.model(audio=batch["audio"])
            batch.update(gen_out)
            audio_pred = batch["audio_pred"]
            real_outs, real_feats = self.model.discriminator(batch["audio"])
            fake_outs, fake_feats = self.model.discriminator(audio_pred)

            loss_adv = self.criterion.adversarial_loss(fake_outs)
            loss_fm = self.criterion.feature_matching_loss(real_feats, fake_feats)
            loss_rec = self.criterion.rec_loss(batch["audio"], audio_pred)
            loss_commit = batch["commit_loss"]
            loss_g = (self.criterion.lambda_adv * loss_adv + self.criterion.lambda_feat * loss_fm + self.criterion.lambda_rec * loss_rec + self.criterion.lambda_commit * loss_commit)
            loss_g.backward()
            
            if self.config["trainer"].get("max_grad_norm", None) is not None:
                clip_grad_norm_(self.model.generator.parameters(), self.config["trainer"]["max_grad_norm"])
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if self.lr_scheduler_d is not None:
                self.lr_scheduler_d.step()

            batch["loss"] = loss_g
            batch["loss_g"] = loss_g
            batch["loss_d"] = loss_d
            batch["loss_adv"] = loss_adv
            batch["loss_fm"] = loss_fm
            batch["loss_rec"] = loss_rec
            batch["loss_commit"] = loss_commit
        else:
            gen_out = self.model(audio=batch["audio"])
            batch.update(gen_out)
            audio_pred = batch["audio_pred"]
            loss_rec = self.criterion.rec_loss(batch["audio"], audio_pred)
            loss_commit = batch["commit_loss"]
            loss_g = (
                self.criterion.lambda_rec * loss_rec
                + self.criterion.lambda_commit * loss_commit
            )
            batch["loss"] = loss_g
            batch["loss_g"] = loss_g
            batch["loss_d"] = torch.tensor(0.0, device=loss_g.device)
            batch["loss_adv"] = torch.tensor(0.0, device=loss_g.device)
            batch["loss_fm"] = torch.tensor(0.0, device=loss_g.device)
            batch["loss_rec"] = loss_rec
            batch["loss_commit"] = loss_commit

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch:
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
        for i in range(min(2, batch["audio"].shape[0])):
            T = batch["audio"].shape[-1]
            if "audio_lengths" in batch:
                T = batch["audio_lengths"][i].item()
            real = batch["audio"][i, :, :T]
            pred = batch["audio_pred"][i, :, :T].clamp(-1, 1)
            self.writer.add_audio(f"real_{i}", real, sample_rate=self.sample_rate)
            self.writer.add_audio(f"pred_{i}", pred, sample_rate=self.sample_rate)

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        params = self.model.generator.parameters()
        params = [p for p in params if p.grad is not None]
        if len(params) == 0:
            return 0.0
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]), norm_type)
        return total_norm.item()
