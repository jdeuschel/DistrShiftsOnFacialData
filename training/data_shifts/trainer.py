import numpy as np
import torch
import pytorch_lightning as pl
from base.base_pl import BasePL
import torch.nn.functional as F
import copy
from model.metrics.metric_uncertainty import ece as ece_score
from model.metrics.metric_uncertainty import get_entropy as get_entropy
from model.helper.temp_scaling import optimal_temp_scale
from pytorch_lightning.metrics import ConfusionMatrix
import pandas as pd
from pathlib import Path
import os
from training.data_shifts.utils.metrics import (
    ECELoss,
    AdaptiveECELoss,
    ClasswiseECELoss,
)



class TrainModel(BasePL):
    def __init__(self, config, model, optimizer, criterion, num_classes, lr_scheduler=None, model_idx=None,new_models=None):
        super().__init__(config, model, criterion, model_idx)
        self.optimizers_pre = optimizer
        self.lr_scheduler_pre = lr_scheduler
        self.num_classes = num_classes
        self.model_idx = model_idx
        self.temperature = 1.0
        self.new_models =new_models
        # Metrics
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()
        self.mixup = config["augmentation"]["mixup"]
        self.n_bins = config["trainer"]["n_bins"]

        # Track & save best model in-memory for testing
        self.curr_best_val = float("inf")
        self.curr_best_model = [i for i in self.model]

        # Class weights approach
        self.num_models = len(self.model)
        self.confmat = ConfusionMatrix(num_classes=self.num_classes)
        self.target_class = config["data_loader"]["args"]["target"]
        self.attributes = {
            "age":{
                "0-2": 0,
                "3-9": 1,
                "10-19": 2,
                "20-29": 3,
                "30-39": 4,
                "40-49": 5,
                "50-59": 6,
                "60-69": 7,
                "more than 70": 8,
            },
            "gender":{
                "Female": 0, 
                "Male": 1,
            },
            "race":{
                "Black": 0,
                "East Asian": 1,
                "Indian": 2,
                "Latino_Hispanic": 3,
                "Middle Eastern": 4,
                "Southeast Asian": 5,
                "White": 6,
            },
        }

        if self.target_class == "gender":
            self.class_names = list(self.attributes['gender'].keys()) 
        if self.target_class == "race":
            self.class_names = list(self.attributes['race'].keys())
        else: #age
            self.class_names = list(self.attributes['age'].keys())

        self.ece_loss = ECELoss(n_bins=self.n_bins)
        self.ada_ece_loss = AdaptiveECELoss(n_bins=self.n_bins)
        self.classwise_ece_loss = ClasswiseECELoss(n_bins=self.n_bins)



    def forward(self, x):
        return self.model(x)

    #############################################
    #                Training                   #
    #############################################

    # To Device
    def on_pretrain_routine_start(self,):
        for idx, model_tmp in enumerate(self.model):
            self.model[idx] = model_tmp.to(self.device)

    # To .train()
    def on_train_epoch_start(self,):
        for model in self.model:
            model.train()

    def training_step(self, batch, batch_idx, optimizer_idx=1):

        data, age, age_mse, gender, race = batch

        if self.target_class == "gender":
            target = gender
        elif self.target_class == "race":
            target = race
        else:  # "age"
            target = age

        loss_all = 0.0
        output_all = torch.zeros(data.shape[0], self.num_classes).to(self.device)
        # batch_size = data.size(0)
        outputs = []

        lr_schedulers = self.lr_scheduler_pre
        optimizers = self.optimizers_pre

        if not isinstance(lr_schedulers, list):
            lr_schedulers = [lr_schedulers]

        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        # Gather outputs of models
        for idx in range(0, len(self.model)):
            lr_scheduler = lr_schedulers[idx]
            optimizer = optimizers[idx]

            if self.mixup:
                data, target = self.prepare_mixup(data, target)

            model = self.model[idx]
            output_in = model(data)
            output_in_softmax = F.softmax(output_in, dim=1)
            
            if self.mixup:
                loss = self.criterion(output_in, target)

            else:
                loss = self.criterion(output_in, target)
                cre = F.cross_entropy(output_in, target)
                mse = F.mse_loss(torch.argmax(output_in_softmax,1).float(), target)
                self.log(
                    "loss_mse", mse, on_epoch=True,
                )
                self.log(
                    "loss_cre", cre, on_epoch=True,
                )

            # losses.append(loss)
            outputs.append(output_in_softmax)
            
            optimizer.zero_grad()
            self.manual_backward(loss, optimizer)
            optimizer.step()
            loss_all = loss_all + loss.item()
            lr_scheduler.step()

            if not self.mixup:
                self.log("accuracy_model_"+str(idx), self.accuracy(output_in_softmax, target))
            self.log("loss_model_"+str(idx), loss, on_epoch=True)


        # Average all predictions
        output_all = torch.sum(torch.stack(outputs, 0), dim=0)
        output_all = output_all / len(self.model)

        loss_all = loss_all / len(self.model)
        if not self.mixup:
            self.log("accuracy", self.accuracy(output_all, target), on_epoch=True)
        self.log("loss", loss_all, on_epoch=True)


    #############################################
    #                Validation                 #
    #############################################

    def on_validation_epoch_start(self,):
        for model in self.model:
            model.eval()
        print("Validation epoch start")

        # self.num_models = len(self.model)
        len_val_dataloader = len(self.trainer.val_dataloaders[0].dataset) #len_val_dataloader = 10954 = len(valid_data_loader.dataset)
        # len_val_dataloader = len(valid_data_loader.dataset)  # use this for debugging
        self.val_prediction_all = torch.zeros((self.num_models, len_val_dataloader, self.num_classes) ) 
        self.val_target_all = torch.zeros(len_val_dataloader)
        self.val_age_all = torch.zeros(len_val_dataloader)
        self.val_gender_all = torch.zeros(len_val_dataloader)
        self.val_race_all = torch.zeros(len_val_dataloader)
        self.curr_pos_val = 0 


    def validation_step(self, batch, batch_idx):

        data, age, age_mse, gender, race = batch

        if self.target_class == "gender":
            target = gender
        elif self.target_class == "race":
            target = race
        else:  # "age"
            target = age

        loss_all = 0
        output_all = torch.zeros(data.shape[0], self.num_classes).to(self.device)

        # losses = []
        outputs = []
        batch_size = data.shape[0]

        # Gather outputs of models
        for idx in range(0, len(self.model)):
            model = self.model[idx]
            output_in = model(data)
            output_in_softmax = F.softmax(output_in, dim=1)

            self.val_prediction_all[idx, self.curr_pos_val : (self.curr_pos_val + batch_size), :] = output_in_softmax  

            if self.mixup:
                loss = self.criterion(output_in, target, valid=True)
                
            else:
                loss = self.criterion(output_in, target)
                val_cre = F.cross_entropy(output_in, target)
                val_mse = F.mse_loss(torch.argmax(output_in_softmax,1).float(), target)
                self.log(
                    "val_loss_mse", val_mse, on_epoch=True,
                )
                self.log(
                    "val_loss_cre", val_cre, on_epoch=True,
                )


            outputs.append(output_in_softmax)
            loss_all = +loss.item()
            model_index = idx
            if self.model_idx != None:
                model_index = self.model_idx
            self.log("val_accuracy_model_"+str(model_index), self.val_accuracy(output_in_softmax, target))
            self.log("val_loss_model_"+str(model_index), loss, on_epoch=True)


        # Average all predictions
        output_all = torch.sum(torch.stack(outputs, 0), dim=0)
        output_all = output_all / len(self.model)
        

        loss_all = loss_all / len(self.model)
        self.log("val_accuracy", self.val_accuracy(output_all, target))
        self.log("val_loss", loss_all, on_epoch=True)

        target = target.int()
        

             
        self.val_target_all[self.curr_pos_val : (self.curr_pos_val + batch_size)] = target
        self.val_age_all[self.curr_pos_val : (self.curr_pos_val + batch_size)] = age.int()
        self.val_gender_all[self.curr_pos_val : (self.curr_pos_val + batch_size)] = gender.int()
        self.val_race_all[self.curr_pos_val : (self.curr_pos_val + batch_size)] = race.int()

        self.curr_pos_val = self.curr_pos_val + batch_size 

        self.target_dict_val_all = {
            "age":self.val_age_all,
            "gender":self.val_gender_all,
            "race":self.val_race_all,
        }
        
        return loss_all


    def on_train_epoch_end(self, outputs):
        val_loss = self.trainer.logged_metrics["val_loss"]
        if val_loss < self.curr_best_val:
            print("-- Model saved in-memory")
            for idx, model in enumerate(self.model):
                self.curr_best_model[idx] = copy.deepcopy(model)
            self.curr_best_val = val_loss
            self.log("curr_best_val", self.curr_best_val)



    def on_validation_epoch_end(self, ):
        
        self.val_prediction_mean = torch.mean(self.val_prediction_all, dim=0)  # ensemble average prediction, average over all models
        # calculation of temperature scaling parameter. (see https://github.com/gpleiss/temperature_scaling) 
        if self.config["calibration"]["temp_scaling"]:
            self.temperature = optimal_temp_scale(self.val_prediction_mean, self.val_target_all)  
        self.log("temperature", self.temperature)
        try:
            accuracy = self.val_accuracy(self.val_prediction_mean, self.val_target_all.int())
        except:
            # only for sanity check
            print("EXCEPT: second softmax necessary")
            accuracy = self.val_accuracy(F.softmax(self.val_prediction_mean,1), self.val_target_all.int())


        val_confidence_maxsoftmax, pred = torch.max(self.val_prediction_mean, 1)
        val_accuracies = pred.eq(self.val_target_all.view_as(pred))  # 1/0 whether pred correct for each input
        ece_old_all, bin_corrects, bins = ece_score(confidences=val_confidence_maxsoftmax, accuracies=val_accuracies, n_bins=self.n_bins)
        confidence_all = torch.mean(val_confidence_maxsoftmax)

        ece_all = self.ece_loss(
            softmax_in=self.val_prediction_mean, labels=self.val_target_all
        )
        ece_class_all = self.classwise_ece_loss(
            softmax_in=self.val_prediction_mean, labels=self.val_target_all
        )
        ece_ada_all = self.ada_ece_loss(
            softmax_in=self.val_prediction_mean, labels=self.val_target_all
        )

        conf_median = torch.median(val_confidence_maxsoftmax)
        conf_quantile1 = torch.quantile(val_confidence_maxsoftmax, q=0.25)
        conf_quantile2 = torch.quantile(val_confidence_maxsoftmax, q=0.75)

        

        # Log Metrics
        self.log("val_accuracy_all", accuracy)
        self.log("val_ece_old_all", ece_old_all)
        self.log("val_ece_all", ece_all)
        self.log("val_ece_class_all", ece_class_all)
        self.log("val_ece_ada_all", ece_ada_all)
        self.log("val_confidence_all", confidence_all)
        self.log("test_confidence_median", conf_median)
        self.log("test_confidence_quantile1", conf_quantile1)
        self.log("test_confidence_quantile2", conf_quantile2)



        # for each attribute accuracy is calculated
        for attribute in self.attributes.keys():
            for class_name, class_idx in zip(self.attributes[attribute].keys(), self.attributes[attribute].values()):
                output_attr = self.val_prediction_mean[self.target_dict_val_all[attribute]==class_idx]
                target_attr = self.val_target_all[self.target_dict_val_all[attribute]==class_idx]
                if output_attr.nelement() != 0:
                    try:
                        attr_acc = self.accuracy(output_attr, target_attr.int())
                    except:
                        # only for sanity check
                        print("EXCEPT: second softmax necessary")
                        attr_acc = self.accuracy(F.softmax(output_attr,1), target_attr.int()) 
                    self.log("val_attribute_accuracy_"+str(attribute)+str(class_name), attr_acc)#, on_epoch=True)
                    confidence_maxsoftmax_attr, pred = torch.max(output_attr, 1)
                    

                    ece = self.ece_loss(
                        softmax_in=output_attr, labels=target_attr
                    )
                    ece_ada = self.ada_ece_loss(
                        softmax_in=output_attr, labels=target_attr
                    )

                    self.log("val_attribute_ece_"+str(attribute)+str(class_name), ece)#, on_epoch=True)
                    self.log("val_attribute_ece_ada_"+str(attribute)+str(class_name), ece_ada)#, on_epoch=True)
                    self.log("val_confidence_"+str(attribute)+str(class_name), torch.mean(confidence_maxsoftmax_attr))#, on_epoch=True)


        del self.val_prediction_all



    #############################################
    #                Testing                    #
    #############################################
    def on_test_start(self,):
        for model in self.model:
            model.eval()

    def on_test_epoch_start(self,):
        len_test_dataloader = len(self.trainer.test_dataloaders[0].dataset)
        # len_val_dataloader = len(test_data_loader.dataset)  # use this for debugging
        self.prediction_all = torch.zeros(
            (self.num_models, len_test_dataloader, self.num_classes)
        )  
        self.prediction_all_tempscale = torch.zeros(
            (self.num_models, len_test_dataloader, self.num_classes)
        )  
        self.target_all = torch.zeros(len_test_dataloader)
        self.age_all = torch.zeros(len_test_dataloader)
        self.gender_all = torch.zeros(len_test_dataloader)
        self.race_all = torch.zeros(len_test_dataloader)
        self.curr_pos = 0


    def test_step(self, batch, batch_idx):

        data, age, age_mse, gender, race = batch

        if self.target_class == "gender":
            target = gender
        elif self.target_class == "race":
            target = race
        else:  # "age"
            target = age

        outputs = []
        outputs_tempscale = []

        target = target.int()
        batch_size = data.shape[0]

        #for idx, model in enumerate(self.model):  
        for idx, model in enumerate(self.curr_best_model):  
            output_in_softmax_tempscale = F.softmax(model(data) / self.temperature , dim=1) 
            output_in_softmax = F.softmax(model(data) , dim=1) 
            self.prediction_all_tempscale[idx, self.curr_pos : (self.curr_pos + batch_size), :] = output_in_softmax_tempscale
            self.prediction_all[idx, self.curr_pos : (self.curr_pos + batch_size), :] = output_in_softmax
            self.log("test_accuracy_step_model"+str(idx), self.test_accuracy(output_in_softmax, target), on_step=True)
            self.log("test_accuracy_step_model_temp"+str(idx), self.test_accuracy(output_in_softmax_tempscale, target), on_step=True)
            outputs.append(output_in_softmax)
            outputs_tempscale.append(output_in_softmax_tempscale)

        self.target_all[self.curr_pos : (self.curr_pos + batch_size)] = target
        self.age_all[self.curr_pos : (self.curr_pos + batch_size)] = age.int()
        self.gender_all[self.curr_pos : (self.curr_pos + batch_size)] = gender.int()
        self.race_all[self.curr_pos : (self.curr_pos + batch_size)] = race.int()
        self.curr_pos = self.curr_pos + batch_size  # Update position

        self.target_dict_all = {
            "age":self.age_all,
            "gender":self.gender_all,
            "race":self.race_all,
        }

        output_all = torch.sum(torch.stack(outputs, 0), dim=0)
        output_all = output_all / len(self.model)

        output_all_tempscale = torch.sum(torch.stack(outputs_tempscale, 0), dim=0)
        output_all_tempscale = output_all_tempscale / len(self.model)

        # Log Metrics
        self.log("test_accuracy_step", self.test_accuracy(output_all, target), on_step=True)
        self.log("test_accuracy_step_temp", self.test_accuracy(output_all_tempscale, target), on_step=True)

    

    def on_test_epoch_end(self,):
        logger = self.logger
        self.on_test_epoch_end_content(tempscale=False)
        if self.config["calibration"]["temp_scaling"]:
            self.on_test_epoch_end_content(tempscale=True)



    def configure_optimizers(self,):
        return self.optimizers_pre, self.lr_scheduler_pre


    def on_test_epoch_end_content(self, tempscale=False):

        if tempscale:
            prediction_all = self.prediction_all_tempscale
            temp="_temp"
        else:
            prediction_all= self.prediction_all
            temp=""

        self.prediction_mean = torch.mean(prediction_all, dim=0)  # ensemble average prediction
        confidence_maxsoftmax, pred = torch.max(self.prediction_mean, 1)

        accuracies = pred.eq(self.target_all.view_as(pred))  # 1/0 whether pred correct for each input
        ece_all, bin_corrects_all, bins_all = ece_score(confidences=confidence_maxsoftmax, accuracies=accuracies, n_bins=self.n_bins)

        ece_all = self.ece_loss(
            softmax_in=self.prediction_mean, labels=self.target_all
        )
        ece_class_all = self.classwise_ece_loss(
            softmax_in=self.prediction_mean, labels=self.target_all
        )
        ece_ada_all = self.ada_ece_loss(
            softmax_in=self.prediction_mean, labels=self.target_all
        )

        accuracy_all = self.test_accuracy(
            self.prediction_mean, self.target_all.int()
        )

        nll_all = F.cross_entropy(
            torch.log(self.prediction_mean), self.target_all.long()
        )
        conf_all = torch.mean(confidence_maxsoftmax)
        conf_median = torch.median(confidence_maxsoftmax)
        conf_quantile1 = torch.quantile(confidence_maxsoftmax, q=0.25)
        conf_quantile2 = torch.quantile(confidence_maxsoftmax, q=0.75)

        # entropy
        entropy_all = torch.mean(get_entropy(self.prediction_mean, n_classes=self.num_classes))
        

        # Log Metrics
        self.log("test_accuracy"+temp, accuracy_all)
        self.log("test_ece"+temp, ece_all)
        self.log("test_ece_ada"+temp, ece_ada_all)
        self.log("test_nll"+temp, nll_all)
        self.log("test_confidence"+temp, conf_all)
        self.log("test_confidence_median"+temp, conf_median)
        self.log("test_confidence_quantile1"+temp, conf_quantile1)
        self.log("test_confidence_quantile2"+temp, conf_quantile2)
        self.log("test_ece_classwise"+temp, ece_class_all)
        self.log("entropy_all"+temp, entropy_all)
        

        # attribute-wise Accuracy and ECE
        values_attribute_ece = []
        values_attribute_ece_ada = []
        values_attribute_acc = []
        values_attribute_nll = []
        labels_attribute = []
        values_attribute_confidence = []
        for attribute in self.attributes.keys():
            for class_name, class_idx in zip(self.attributes[attribute].keys(), self.attributes[attribute].values()):
                output_attr = self.prediction_mean[self.target_dict_all[attribute]==class_idx]
                target_attr = self.target_all[self.target_dict_all[attribute]==class_idx]
                if output_attr.nelement() != 0:
                    acc = self.accuracy(output_attr, target_attr.int())
                    self.log("test_attribute_accuracy"+temp+"_"+str(attribute)+str(class_name), acc)#, on_epoch=True)
                    confidence_maxsoftmax_attr, pred = torch.max(output_attr, 1)
                    corrects = pred.eq(target_attr.view_as(pred))  # 1/0 whether pred correct for each input
                    ece, bin_corrects, bins = ece_score(confidences=confidence_maxsoftmax_attr.detach(), accuracies=corrects, n_bins=self.n_bins)

                    # ECE table
                    data = [[x, y] for (x, y) in zip(bins, bin_corrects)]

                    ece = self.ece_loss(
                        softmax_in=output_attr, labels=target_attr
                    )
                    ece_ada = self.ada_ece_loss(
                        softmax_in=output_attr, labels=target_attr
                    )
                    nll = F.cross_entropy(torch.log(output_attr),target_attr.long()) 
                    confidence = torch.mean(confidence_maxsoftmax_attr)

                    self.log("test_attribute_acc"+temp+"_"+str(attribute)+str(class_name), acc)#, on_epoch=True)
                    self.log("test_attribute_ece"+temp+"_"+str(attribute)+str(class_name), ece)#, on_epoch=True)
                    self.log("test_attribute_ece_ada"+temp+"_"+str(attribute)+str(class_name), ece_ada)#, on_epoch=True)
                    self.log("test_attribute_nll"+temp+"_"+str(attribute)+str(class_name), nll)
                    self.log("test_confidence"+temp+"_"+str(attribute)+str(class_name), confidence)#, on_epoch=True)

                    values_attribute_acc.append(acc.item())
                    values_attribute_ece.append(ece.item())
                    values_attribute_ece_ada.append(ece_ada.item())
                    values_attribute_nll.append(nll.item())
                    values_attribute_confidence.append(confidence.item())

                    labels_attribute.append(str(attribute)+str(class_name))
        data_attribute = [[label, acc_att, ece_att, ece_ada_att, nll_att, conf_att] for (label, acc_att, ece_att, ece_ada_att, nll_att, conf_att) in zip(labels_attribute, values_attribute_acc, values_attribute_ece, values_attribute_ece_ada, values_attribute_nll, values_attribute_confidence)]
        data_attribute.append(["all",accuracy_all.item(), ece_all.item(), ece_ada_all.item(), nll_all.item(), conf_all.item()])
        data_pred = self.prediction_all
        data_tar = self.target_all
        data_pred_temp = self.prediction_all_tempscale
        
        # # # # # # # # # # # # # # # # # # # # # # #
        #     Save predictions as csv and npy       #
        # # # # # # # # # # # # # # # # # # # # # # #
        if self.config["trainer"]["csv"] == True:
            print("Creating CSV")
            df_helper = pd.DataFrame(data_attribute, columns=["attribute", "acc", "ece", "ece_ada", "nll", "confidence"])
            att_list = df_helper['attribute'].to_list()
            metrics_list = df_helper.columns.to_list()
            colnames = ["experiment"]+["{}_{}".format(b_, a_) for a_ in metrics_list[1:] for b_ in att_list]
            # colnames.append(['ensemble','mixup', 'target'])
            df = pd.DataFrame(columns=colnames)
            experiment = [self.config["group"]]
            data = experiment + df_helper['acc'].to_list() + df_helper['ece'].to_list() + df_helper['ece_ada'].to_list() + df_helper['nll'].to_list() + df_helper['confidence'].to_list()
            df.loc[1]=data

            df['ensemble'] = self.config["method"]["num_ensemble"]
            df['mixup'] = self.config["augmentation"]["mixup"]
            df['target'] = self.config["data_loader"]["args"]["target"]
            df['epochs'] = self.config["trainer"]["epochs"]

            if self.config["method"]["num_ensemble"] > 1: 
                method = "ensemble_"
            elif self.config["augmentation"]["mixup"] : 
                method = "mixup_"
            else:
                method = "vanilla_"
            comment = self.config["comment"]

            cluster_used = True if "HOSTNAME" in os.environ else False
            if cluster_used:
                path = "/output/data-shifts/" + self.config["data_loader"]["args"]["target"] 
            else:
                path = Path(self.config["trainer"]["save_dir"])
            csv_exists = os.path.exists(os.path.join(path, method + comment + temp + ".csv"))
            if csv_exists:
                print("csv exists, so new line is appended")
                df2 = pd.read_csv(filepath_or_buffer=os.path.join(path, method + comment + temp + ".csv"), index_col=0)
                df2 = df2.append(df)
                df2.to_csv(os.path.join(path, method + comment + temp + ".csv"))
            else:
                df.to_csv(os.path.join(path, method + comment + temp + ".csv"))

            # Save of Prediction with/without temperature scale and targets

            # define variables to save
            data_pred = data_pred.numpy()
            data_tar = data_tar.numpy()
            data_pred_temp = data_pred_temp.numpy()
            # save for one type of unique run
            experiment_save = self.config["group"]

            # check if files exist
            save_numpy_exist1 = os.path.exists(os.path.join(path, method + comment + temp + experiment_save + "_pred.npy"))
            save_numpy_exist2 = os.path.exists(os.path.join(path, method + comment + temp + experiment_save + "_tar.npy"))
            save_numpy_exist3 = os.path.exists(os.path.join(path, method + comment + temp + experiment_save + "_pred_tempsc.npy"))
            if save_numpy_exist1 and save_numpy_exist2 and save_numpy_exist3: # all three files exist
                data_pred_load = np.load(os.path.join(path, method + comment + temp + experiment_save + "_pred.npy"))
                data_tar_load = np.load(os.path.join(path, method + comment + temp + experiment_save + "_tar.npy"))
                data_pred_temp_load = np.load(os.path.join(path, method + comment + temp + experiment_save + "_pred_tempsc.npy"))
                # check if loaded targets have the right shape
                if len(data_tar_load.shape) == 1:
                    # for one dimension add another dimension and concatenate
                    data_tar = np.concatenate((data_tar_load[np.newaxis, :], data_tar[np.newaxis, :]), axis=0)
                elif len(data_tar_load.shape) == 2:
                    # for two dimensions do not add one dimension to loaded target and concatenate
                    data_tar = np.concatenate((data_tar_load, data_tar[np.newaxis, :]), axis=0)
                else:
                    raise ValueError('Wrong dimensions for target save: {}'.format(data_tar_load.shape))

                # concatenate predictions and predictions with temperature scale
                data_pred = np.concatenate((data_pred_load, data_pred), axis=0)
                data_pred_temp = np.concatenate((data_pred_temp_load, data_pred_temp), axis=0)
                # save concatenated data
                np.save(os.path.join(path, method + comment + temp + experiment_save + "_pred.npy"), data_pred)
                np.save(os.path.join(path, method + comment + temp + experiment_save + "_tar.npy"), data_tar)
                np.save(os.path.join(path, method + comment + temp + experiment_save + "_pred_tempsc.npy"), data_pred_temp)
            elif not save_numpy_exist1 and not save_numpy_exist2 and not save_numpy_exist3:
                # all three files do not exist, so save data
                np.save(os.path.join(path, method + comment + temp + experiment_save + "_pred.npy"), data_pred)
                np.save(os.path.join(path, method + comment + temp + experiment_save + "_tar.npy"), data_tar)
                np.save(os.path.join(path, method + comment + temp + experiment_save + "_pred_tempsc.npy"), data_pred_temp)
            else:
                raise ValueError('Something is not saved right before')
