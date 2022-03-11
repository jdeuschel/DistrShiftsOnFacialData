"""
Base Class for PyTorch Lightning
"""
import os
from abc import abstractmethod
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import model.metric as module_metric
import model.model as module_arch
import model.optim as optim
import model.loss as module_loss
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import shutil
from pytorch_lightning.utilities import rank_zero_only
from model.augmentation.mixup import mixup

class BasePL(pl.LightningModule):
    """
    Base Class for Training with PyTorch Lightning

    You can overwrite every component if you want, e.g. the criterion will be initialized here, but if you pass it to your 
    child trainer than you can overwrite it. 

    """

    def __init__(self, config, model, criterion, model_idx):
        super().__init__()
        self.config = config
        self.automatic_optimization = self.config["trainer"]["automatic_optimization"]
        self.model = model
        self.criterion = criterion
        # self.optimizer = None
        # self.lr_scheduler = None
        self.cwd = os.getcwd()
        self.automatic_optimization=self.config["trainer"]["automatic_optimization"]

        # Automatic inits from config
        
        if model_idx is None:
            model_idx = 0
        # pdb.set_trace()
        self.save_dir = os.path.join(self.cwd, config.save_dir)
        self.checkpoint_dir = os.path.join(self.save_dir, "models"+str(model_idx))



    def prepare_trainer(self, hyperparams_additional=None):
        # Threading
        if self.config["trainer"]["threads"] > 0:
            torch.set_num_threads(self.config["trainer"]["threads"])
            print("Number of Threads set: ", torch.get_num_threads())

        # Do not save checkpoints on local workstation to avoid clutter.
        if self.config["trainer"]["monitor"] == "off" or "HOSTNAME" not in os.environ:
            checkpoint_callback = None
            self.checkpoint_active = False
        else:
            self.checkpoint_active = True
            print("Checkpoint is used...")
            mnt_mode, mnt_metric = self.config["trainer"]["monitor"].split()
            print("We track: ", mnt_metric)
            assert mnt_mode in ["min", "max"]
            checkpoint_callback = ModelCheckpoint(
                filepath=self.checkpoint_dir,
                save_top_k=1,
                verbose=True,
                monitor=mnt_metric,
                mode=mnt_mode,
                prefix="",
                period=self.config["trainer"]["save_period"],
            )

        distributed_backend = (
            None
            if (self.config["trainer"]["n_gpu"] == 1 and self.config["trainer"]["n_nodes"] == 1)
            else self.config["trainer"]["ddp_backend"]
        )
        max_steps = None if self.config["trainer"]["max_steps"] == 0 else self.config["trainer"]["max_steps"]

        if self.config["trainer"]["resume_from_checkpoint"]:
            resume_from_checkpoint = f"/output/{self.config['tags']['model_info']}_best_tmp.ckpt"
        else:
            resume_from_checkpoint = None

        if self.config["trainer"]["ddp_sharded"]:
            plugins = "ddp_sharded"
        else:
            plugins = None

        logger = []
        #add logger here
        if logger == []:
            logger = None

        # Callbacks
        callbacks = []
        if logger:
            print("--- Learning Rate Monitor is used")
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        return pl.Trainer(
            max_epochs=self.config["trainer"]["epochs"],
            gpus=self.config["trainer"]["n_gpu"],
            num_nodes=self.config["trainer"]["n_nodes"],
            accelerator=distributed_backend,
            logger=logger,
            precision=self.config["trainer"]["precision"],
            limit_val_batches=self.config["trainer"]["limit_val_batches"],
            limit_train_batches=self.config["trainer"]["limit_train_batches"],
            checkpoint_callback=checkpoint_callback,
            profiler=self.config["trainer"]["profiler"],
            callbacks=callbacks,
            deterministic=self.config["trainer"]["deterministic"],
            benchmark=self.config["trainer"]["benchmark"],
            log_gpu_memory=self.config["trainer"]["log_gpu"],
            fast_dev_run=self.config["trainer"]["fast_dev_run"],
            auto_lr_find=self.config["trainer"]["auto_lr_find"],
            max_steps=max_steps,
            resume_from_checkpoint=resume_from_checkpoint,
            log_every_n_steps=self.config["trainer"]["log_every_n_steps"],
            flush_logs_every_n_steps=self.config["trainer"]["flush_logs_every_n_steps"],
            auto_select_gpus=self.config["trainer"]["auto_select_gpus"],
            gradient_clip_val=self.config["trainer"]["gradient_clip_val"],
            terminate_on_nan=self.config["trainer"]["terminate_on_nan"],
            track_grad_norm=self.config["trainer"]["track_grad_norm"],
            plugins=plugins,
            # automatic_optimization=self.config["trainer"]["automatic_optimization"],
        )

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        """
        Return Optimizer and LR scheduler. Can be overwritten for special use cases. 
        """

        optimizer = self.config.init_obj("optimizer", optim, self.parameters())
        lr_scheduler = self.config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)
        return [optimizer], [lr_scheduler]

    @rank_zero_only
    def move_output(self,):
        try:
            if "HOSTNAME" in os.environ and self.checkpoint_active:
                print(os.listdir(self.checkpoint_dir))
                file_name = os.listdir(self.checkpoint_dir)[0]
                shutil.copy(
                    os.path.join(self.checkpoint_dir, file_name),
                    f"/output/{self.config['tags']['model_info']}_best_tmp.ckpt",
                )
                print(
                    "Moved checkpoint to output folder with name: ",
                    f"{self.config['tags']['model_info']}_best_tmp.ckpt",
                )
        except Exception as e:
            print(e)
            print("Could not move output")

    def on_epoch_end(self,):
        """ 
        Copy checkpoint to output folder if config activated. 
        """
        # try:
        self.move_output()

        # except:
        #    print("Could not move weights to output folder.")

    def prepare_mixup(self, data, target):
        data, target = mixup(data, target, self.config["augmentation"]["mixup_alpha"]) #, n_classes=self.num_classes)
        t1, t2, lam = target
        target = (t1.to(self.device), t2.to(self.device), lam)
        return data, target
