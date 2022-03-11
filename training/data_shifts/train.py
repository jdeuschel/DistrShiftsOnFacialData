import os
import argparse
import collections
import torch

# Project Modules
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.model as module_arch
import model.optim as optim
from parse_config import ConfigParser
from trainer import TrainModel


def main(config):
    print("Devices", torch.cuda.device_count())

    print(config["lr_scheduler"]["args"]["milestones"])


    # Data Loader
    experiment = config['group']
    print("starting experiment ",experiment)
    data_loader_factory = config.init_obj("data_loader", module_data, experiment=experiment)
    data_loader = data_loader_factory.get_training_loader()
    valid_data_loader = data_loader_factory.get_validation_loader()
    num_classes = data_loader_factory.num_classes


    # Init Modules
    criterion = getattr(module_loss, config["loss"])

    # Saving tests
    cluster_used = True if "HOSTNAME" in os.environ else False
    if cluster_used:
        path = "/output/data-shifts/" + config["data_loader"]["args"]["target"]
    else :
        path = "No output path set"
    
    print(cluster_used)
    print(path)
    print(os.path.exists("/output/data-shifts/"))
    

    #################################################################
    #                         ENSEMBLE                              #
    #################################################################

    # Init Ensemble
    num_ensemble = config["method"]["num_ensemble"]
    models = []

    for idx in range(0, num_ensemble):
        model = config.init_obj("arch", module_arch)
        models.append(model)
    
    print("Devices", torch.cuda.device_count())

    optimizers = []
    lr_schedulers = []

    for idx, model in enumerate(models):
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = config.init_obj("optimizer", optim, trainable_params)

        lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)
    


    # # # # # # # # # # # # # # # # # # # # # # #
    #                  Training                 #
    # # # # # # # # # # # # # # # # # # # # # # #
    sequential_training = config['method']['sequential']
    print("SEQUENTIAL: ", sequential_training)
    new_models = []

    for idx, _ in enumerate(models):
        train_model = TrainModel(config, [models[idx]], [optimizers[idx]], criterion, num_classes, [lr_schedulers[idx]], model_idx=idx)
        trainer = train_model.prepare_trainer()

        trainer.fit(train_model, data_loader, valid_data_loader)
        new_models.append(train_model.model[0])



    # # # # # # # # # # # # # # # # # # # # # # #
    #                  Testing                  #
    # # # # # # # # # # # # # # # # # # # # # # #
    print(sequential_training)
    print(config["trainer"]["csv"])
    print("Starting Testing")

    path = "/output/data-shifts/" + config["data_loader"]["args"]["target"]
    print(os.path.exists(path))

    if sequential_training:
        if config["trainer"]["csv"] == True: 
            path = "/output/data-shifts/" + config["data_loader"]["args"]["target"]
            print(os.path.exists(path))
            print(os.path.join(path, config.method + config.comment + config.temp + ".csv"))

        test_model = TrainModel(config, new_models, optimizers, criterion, num_classes, lr_schedulers)
        trainer_test = test_model.prepare_trainer()
        test_data_loader = data_loader_factory.get_test_loader()
        trainer_test.test( model=test_model,  test_dataloaders=test_data_loader, ckpt_path=None)

    else:
        test_data_loader = data_loader_factory.get_test_loader()
        trainer.test(test_dataloaders=test_data_loader, ckpt_path=None)

     

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Experiments")
    args.add_argument("-c", "--config", default="configs/config.yml", type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["-pretrained", "--pretrained"], type=bool, target="method;pretrained"),
        CustomArgs(["-finetuning", "--finetuning"], type=str, target="method;finetuning"),
        CustomArgs(["-target", "--target"], type=str, target="data_loader;args;target"),#
        CustomArgs(["-temp_scale", "--temperature"], type=str, target="calibration;temp_scaling"),
        CustomArgs(["-num_ensemble", "--num_ensemble"], type=int, target="method;num_ensemble"),
        CustomArgs(["-n_bins", "--n_bins"], type=int, target="trainer;n_bins"),
        CustomArgs(["-lr_milestones", "--lr_milestones"], type=list, target="lr_scheduler;args;milestones"),
        CustomArgs(["-comment", "--comment"], type=str, target="comment"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

    
