import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
# from logger import setup_logging
from utils import read_yml, write_yml
import collections
import json

class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "True", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "False", "f", "n", "0"):
        return False


class ConfigParser(metaclass=Singleton):
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
        and logging module.
        :param config: Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        :param run_id: Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = save_dir / "models" / exper_name / run_id
        self._log_dir = save_dir / "log" / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_yml(self.config, self.save_dir / "config.yml")

        # configure logging module
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        

        os.environ["MODE"] = "TRAINING"
        os.environ["MODEL_LOCATION"] = "saved/resnet_pretrained"  # todo: change location where to store imagenet pretrained model

    
        if self.config["augmentation"]["mixup"]:
            self.config["loss"] = "mixup"

        # Prepare Environment variables for cluster if TF Jobs is used
        if "TF_CONFIG" in os.environ and "HOSTNAME" in os.environ:
            print("TF_CONFIG is used")
            tf_config = os.environ["TF_CONFIG"]
            tf_config_parsed = json.loads(tf_config)
            try:
                os.environ["MASTER_ADDR"] = tf_config_parsed["cluster"]["ps"][0].split(".")[0]
            except:
                print("MASTER_ADDR not found")
            os.environ["MASTER_PORT"] = "2225"
            # os.environ["LOCAL_RANK"] = "0"
            if "ps" in tf_config_parsed["cluster"] and tf_config_parsed["task"]["type"] == "worker":
                os.environ["NODE_RANK"] = str(tf_config_parsed["task"]["index"] + 1)
                # os.environ["LOCAL_RANK"] = "0"  # str(tf_config_parsed["task"]["index"] + 1)
                # os.environ["RANK"] = str(tf_config_parsed["task"]["index"] + 1)
            else:
                os.environ["NODE_RANK"] = "0"
                # os.environ["LOCAL_RANK"] = "0"
                # os.environ["RANK"] = "0"
        try:
            if os.environ["RANK"]:
                os.environ["NODE_RANK"] = os.environ["RANK"]
                del os.environ["RANK"]
                del os.environ["WORLD_SIZE"]
        except:
            print("nothing found")
        # os.environ["MASTER_PORT"] = "2225"

    @classmethod
    def from_args(cls, args, options=""):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
        options_standard = [
            CustomArgs(["-lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
            CustomArgs(["-bs", "--batch_size"], type=int, target="data_loader;args;batch_size"),
            CustomArgs(["-e", "--epochs"], type=int, target="trainer;epochs"),
            CustomArgs(["-loss", "--loss"], type=str, target="loss"),
            CustomArgs(["-i", "--model_info"], type=str, target="tags;model_info"),
            CustomArgs(["-type", "--run_type"], type=str, target="tags;run_type"),
            CustomArgs(["-m", "--model"], type=str, target="arch;type"),
            CustomArgs(["-opt", "--optimizer"], type=str, target="optimizer;type"),
            CustomArgs(["-data", "--data"], type=str, target="data_loader;type"),
            CustomArgs(["-saving", "--saving"], type=int, target="trainer;save_period"),
            CustomArgs(["-size", "--size"], type=int, target="input_size"),
            CustomArgs(["-n_gpu", "--n_gpu"], type=int, target="trainer;n_gpu"),
            CustomArgs(["-n_nodes", "--n_nodes"], type=int, target="trainer;n_nodes"),
            CustomArgs(["-auto_lr_find", "--auto_lr_find"], type=str2bool, target="trainer;auto_lr_find"),
            CustomArgs(["-fast_dev_run", "--fast_dev_run"], type=str2bool, target="trainer;fast_dev_run"),
            CustomArgs(["-threads", "--threads"], type=int, target="trainer;threads"),
            CustomArgs(["-ddp_backend", "--ddp_backend"], type=str, target="trainer;ddp_backend"),
            CustomArgs(["-profiler", "--profiler"], type=str2bool, target="trainer;profiler"),
            CustomArgs(["-deterministic", "--deterministic"], type=str2bool, target="trainer;deterministic"),
            CustomArgs(["-benchmark", "--benchmark"], type=str2bool, target="trainer;benchmark"),
            CustomArgs(["-precision", "--precision"], type=int, target="trainer;precision"),
            CustomArgs(["-log_gpu", "--log_gpu"], type=str2bool, target="trainer;log_gpu"),
            CustomArgs(["-resume", "--resume_from_checkpoint"], type=str2bool, target="trainer;resume_from_checkpoint",),
            CustomArgs(["-limit_train_batches", "--limit_train_batches"], type=float, target="trainer;limit_train_batches"),
            CustomArgs(["-limit_val_batches", "--limit_val_batches"], type=float, target="trainer;limit_val_batches"),
            CustomArgs(["-max_steps", "--max_steps"], type=int, target="trainer;max_steps"),
            CustomArgs(["-gradient_clip_val", "--gradient_clip_val"], type=float, target="trainer;gradient_clip_val"),
            CustomArgs(["-ddp_sharded", "--ddp_sharded"], type=str2bool, target="trainer;ddp_sharded"),
            CustomArgs(["-group", "--group"], type=str, target="group"),
            CustomArgs(["-sequential", "--sequential"], type=str2bool, target="method;sequential"),
            CustomArgs(["-experiment", "--experiment"], type=str, target="experiment"),
            CustomArgs(["-mixup", "--mixup"], type=str2bool, target="augmentation;mixup"),
            CustomArgs(["-mixupalpha", "--alpha"], type=float, target="augmentation;mixup_alpha"),
            CustomArgs(["-weight_decay", "--weight_decay"], type=float, target="optimizer;args;weight_decay"),
        ]

        options = options_standard + options

        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / "config.yml"
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.yml', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)

        config = read_yml(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_yml(args.config))

        # parse custom cli options into dictionary
        modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        if name == "lr_scheduler" and module_name is None:
            return None
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all([k not in module_args for k in kwargs]), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        logger.handlers = []
        formatter = logging.Formatter("%(message)s")
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(formatter)

        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
