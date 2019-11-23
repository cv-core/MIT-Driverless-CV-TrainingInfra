#!/usr/bin/python3

import argparse
import subprocess

import optuna
import pymysql
import numpy as np
import torch

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
    parser.add_argument('--num_trials', type=int, default=100, help="number of optuna trials to run")
    parser.add_argument('--study_name', type=str, default='optuna_keypoints_study', help="cometml / optuna study name")

    ###### geo loss study ######
    add_bool_arg('geo_loss_study', default=False, help="whether to initialize study of vertical and horizontal geo loss")
    ##### loss type study ######
    add_bool_arg('loss_type_study', default=False, help="whether to initialize study of three different loss type: l2_softargmax|l2_heatmap|l1_softargmax")
    ############################

    add_bool_arg('auto_sd', default=False, help='whether to enable automatical instance shutdown after training. default to True')

    opt = parser.parse_args()

    def objective(trial):
        ######################################
        if opt.geo_loss_study:
            geo_loss_gamma_vert = trial.suggest_uniform('geo_loss_gamma_vert', 0, 0.15)
            geo_loss_gamma_horz = trial.suggest_uniform('geo_loss_gamma_horz', 0, 0.15)
        else:
            geo_loss_gamma_vert = 0
            geo_loss_gamma_horz = 0
        ######################################
        if opt.loss_type_study:
            loss_type = trial.suggest_categorical('loss_type', ['l2_softargmax', 'l2_heatmap', 'l1_softargmax'])
        else:
            loss_type = 'l1_softargmax'
        ######################################

        # build the argstring
        args = {
            'geo_loss_gamma_vert': geo_loss_gamma_vert,
            'geo_loss_gamma_horz': geo_loss_gamma_horz,
            'loss_type': loss_type,
            "study_name": opt.study_name,
            "auto_sd": opt.auto_sd
        }
        arglist = ["python3", "-u",  "train_eval.py"]
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

        result_file = open("logs/" + opt.study_name +".txt","r+")
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
