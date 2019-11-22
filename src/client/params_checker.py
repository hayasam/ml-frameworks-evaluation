from models.base_model import EvaluationModel
import importlib

import sys
import numpy
import configargparse
import numpy as np
import subprocess

import trainer
from models.pytorch_models import EvaluationAlex
from models.pytorch_models import alexnet
from models.models_store import ModelStore
import torch
import torch.nn as nn

import unittest

# numpy.set_printoptions(threshold=sys.maxsize)
# torch.set_printoptions(threshold=10000000)

# 8 layers for AlexNet (5*Conv2d and 3*Linear)

class TrainingDiff():

    # file names
    net_diff_name = ""
    vgg_diff_name = ""
    alex_diff_name = ""

    # models
    buggy_net = None
    corrected_net = None
    buggy_vgg = None
    corrected_vgg = None
    buggy_alex = None
    corrected_alex = None
    seed = 1869809928

    def init_models(self):
        trainer.set_local_seed(self.seed)
        self.buggy_net = self.load_model(library="pytorch", name="Net", **{'use_gpu': True})
        self.buggy_net.initialize_weights(self.seed)
        
        trainer.set_local_seed(self.seed)
        self.corrected_net = self.load_model(library="pytorch", name="Net", **{'use_gpu': True})
        self.corrected_net.initialize_weights(self.seed)

        trainer.set_local_seed(self.seed)
        self.buggy_vgg = self.load_model(library="pytorch", name="EvaluationVGG", **{'use_gpu': True})
        self.buggy_vgg.initialize_weights(self.seed)
        
        trainer.set_local_seed(self.seed)
        self.corrected_vgg = self.load_model(library="pytorch", name="EvaluationVGG", **{'use_gpu': True})
        self.corrected_vgg.initialize_weights(self.seed)

        trainer.set_local_seed(self.seed)
        self.buggy_alex = self.load_model(library="pytorch", name="EvaluationAlex", **{'use_gpu': True})
        self.buggy_alex.initialize_weights(self.seed)
        
        trainer.set_local_seed(self.seed)
        self.corrected_alex = self.load_model(library="pytorch", name="EvaluationAlex", **{'use_gpu': True})
        self.corrected_alex.initialize_weights(self.seed)

    # duplicate of models_store.py get_model_for_name()
    def load_model(self, library: str, name: str, **kwargs) -> EvaluationModel:
        qualified_module_name = '{}_models'.format(library).lower()
        module = __import__('{}.{}'.format('models', qualified_module_name), fromlist=[name])
        cl = getattr(module, name)
        return cl(**kwargs)
    

    def find_nbr_params(self, modules):
        params = []

        for m in modules:
            if hasattr(m, 'weight') and isinstance(m.weight, torch.nn.parameter.Parameter):
                params.append(m.weight)
        return params

    def find_str_params(self, modules):
        p = []

        for m in modules:
            if hasattr(m, 'weight') and isinstance(m.weight, torch.nn.parameter.Parameter):
                p.append(str(type(m)))
                p.append(str(len(m.weight)))
                p.append(str(m.weight))
        buggy_params_to_print = '\n'.join(p)
        return buggy_params_to_print

    def do_net_diff(self):
        self.init_models()
        seed_str = str(self.seed)

        buggy_net_params = self.find_str_params(self.buggy_net.modules())

        buggy_name = "pytorch_Net_buggy_" + seed_str + ".txt"
        with open(buggy_name, "w") as layer_output:
            layer_output.write(buggy_net_params)

        corrected_net_params = self.find_str_params(self.corrected_net.modules())

        corrected_name = "pytorch_Net_corrected_" + seed_str + ".txt"
        with open(corrected_name, "w") as layer_output:
            layer_output.write(corrected_net_params)

        # bash diff
        diff_name = "diff_Net" + seed_str + "_file.txt"
        diff_file = open(diff_name, "w")
        subprocess.Popen(["diff", buggy_name, corrected_name])
        diff = subprocess.Popen(["diff", buggy_name, corrected_name], stdout=diff_file)
        diff_file.close()

        self.net_diff_name = diff_name

    def do_vgg_diff(self):
        self.init_models()
        seed_str = str(self.seed)

        buggy_vgg_params = self.find_str_params(self.buggy_vgg.vgg_model.modules())

        buggy_name = "pytorch_EvaluationVGG_buggy_" + seed_str + ".txt"
        with open(buggy_name, "w") as layer_output:
            layer_output.write(buggy_vgg_params)

        corrected_vgg_params = self.find_str_params(self.corrected_vgg.vgg_model.modules())

        corrected_name = "pytorch_EvaluationVGG_corrected_" + seed_str + ".txt"
        with open(corrected_name, "w") as layer_output:
            layer_output.write(corrected_vgg_params)

        # bash diff
        diff_name = "diff_EvaluationVGG" + seed_str + "_file.txt"
        diff_file = open(diff_name, "w")
        subprocess.Popen(["diff", buggy_name, corrected_name])
        diff = subprocess.Popen(["diff", buggy_name, corrected_name], stdout=diff_file)
        diff_file.close()

        self.vgg_diff_name = diff_name

    def do_alex_diff(self):
        self.init_models()
        seed_str = str(self.seed)

        buggy_alex_params = self.find_str_params(self.buggy_alex.alex_model.modules())

        buggy_name = "pytorch_EvaluationAlex_buggy_" + seed_str + ".txt"
        with open(buggy_name, "w") as layer_output:
            layer_output.write(buggy_alex_params)

        corrected_alex_params = self.find_str_params(self.corrected_alex.alex_model.modules())

        corrected_name = "pytorch_EvaluationAlex_corrected_" + seed_str + ".txt"
        with open(corrected_name, "w") as layer_output:
            layer_output.write(corrected_alex_params)

        # bash diff
        diff_name = "diff_EvaluationAlex" + seed_str + "_file.txt"
        diff_file = open(diff_name, "w")
        subprocess.Popen(["diff", buggy_name, corrected_name])
        diff = subprocess.Popen(["diff", buggy_name, corrected_name], stdout=diff_file)
        diff_file.close()

        self.alex_diff_name = diff_name

    # TODO the purpose of this is to run all do_diff, if needed
    # def check_exp_diff():
        

if __name__ == "__main__":
    differentiator = TrainingDiff()
    differentiator.init_models()
    differentiator.do_alex_diff()
