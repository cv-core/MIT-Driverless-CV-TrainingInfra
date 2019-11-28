#!/usr/bin/python3

import argparse
import subprocess

import optuna
import pymysql
import numpy as np
import torch

##### section for all random seeds #####
# cv2.setRNGSeed(17)
# torch.manual_seed(17)
# np.random.seed(17)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
########################################

pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
pymysql.converters.conversions = pymysql.converters.encoders.copy()
pymysql.converters.conversions.update(pymysql.converters.decoders)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    def add_bool_arg(name, default, help):
        arg_group = parser.add_mutually_exclusive_group(required=False)
        arg_group.add_argument('--' + name, dest=name, action='store_true', help=help)
        arg_group.add_argument('--no_' + name, dest=name, action='store_false', help=("Do not " + help))
        parser.set_defaults(**{name:default})
    parser.add_argument('--model_cfg', type=str, help='cfg file path',required=True)
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument('--num_epochs', type=int, default=20, help='maximum number of epochs')
    parser.add_argument('--num_trials', type=int, default=100, help="number of optuna trials to run")
    parser.add_argument('--val_tolerance', type=int, default=1, help="tolerance for validation loss decreasing")
    parser.add_argument('--study_name', type=str, default='optuna_test_6', help="cometml / optuna study name")


    ##### tile and scale #####
    add_bool_arg('ts_study', default=False, help="whether to initialize study of whether using tiling dataloader")
    ##########################
    add_bool_arg('optimizer_study', default=False, help="whether to have optuna study between Adam and SGD optimizer")
    add_bool_arg('loss_study', default=False, help="whether to have optuna study on loss constants")

    add_bool_arg('auto_sd', default=False, help='whether to enable automatical instance shutdown after training. default to True')

    opt = parser.parse_args()

    def objective(trial):
        ######################################
        if opt.loss_study:
            xy_loss = trial.suggest_uniform('xy_loss', 1.6, 2.4)
            wh_loss = trial.suggest_uniform('wh_loss', 1.28, 1.92)
            no_object_loss = trial.suggest_uniform('no_object_loss', 20.0, 30.0)
            object_loss = trial.suggest_uniform('object_loss', 0.08, 0.12)
            
        else:
            xy_loss = 2
            wh_loss = 1.6
            no_object_loss = 25
            object_loss = 0.1
        ######################################

        if opt.ts_study:
            tile = trial.suggest_categorical('tile', [False, True])
        else:
            tile = True #Default to use tiling dataloader
        ######################################

        if opt.optimizer_study:
            optimizer_pick = trial.suggest_categorical('optimizer_pick', ["Adam", "SGD"])
        else:
            optimizer_pick = "Adam"    #Default to Adam optimizer
        ######################################

        # build the argstring
        args = {
            "model_cfg": opt.model_cfg,
            "ts": tile,
            "xy_loss": xy_loss,
            "wh_loss": wh_loss,
            "no_object_loss": no_object_loss,
            "object_loss": object_loss,
            "num_epochs": opt.num_epochs,
            "checkpoint_interval": opt.checkpoint_interval,
            "optimizer_pick": optimizer_pick,
            "val_tolerance": opt.val_tolerance,
            "auto_sd": opt.auto_sd
        }
        arglist = ["python3", "train.py"]
        for arg, value in args.items():
            if value is None:
                continue
            if value is False:
                arglist.append(f"--no_{arg}")
                continue
            if value is True:
                arglist.append(f"--{arg}")
                continue
            arglist.append(f"--{arg}={value}")

        statement = " ".join(arglist)
        print(f"statement for this study is: ")
        print(statement)

        # calling through subprocess to ensure that all cuda memory is fully released between experiments
        subprocess.check_call(arglist)

        result_file = open("logs/result.txt","r+")
        score = float(result_file.read())
        print(f"score for this study is {score}")
        return score    # want to return a value to minimize

    try:
        # study = optuna.create_study(study_name=opt.study_name, storage="mysql+pymysql://root:root@35.224.251.208/optuna")
        study = optuna.create_study(study_name=opt.study_name)
        print("Created optuna study")
    except ValueError as e:
        if "Please use a different name" in str(e):
            # study = optuna.Study(study_name=opt.study_name, storage="mysql+pymysql://root:root@35.224.251.208/optuna")
            study = optuna.Study(study_name=opt.study_name)
            print("Joined existing optuna study")
        else:
            raise
    except:
        raise
    study.optimize(objective, n_trials=opt.num_trials)
