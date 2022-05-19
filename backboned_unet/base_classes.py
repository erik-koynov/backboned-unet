from abc import ABCMeta, abstractmethod
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
import logging
import numpy as np
import sys
sys.path.append('../../datasets')
from datasets.custom_augmentations import standard_scaling, to_rgb
logger = logging.getLogger("models_logger")
import os
import json
import re
from importlib import import_module


class StrictMeta(ABCMeta):
    def __new__(cls,  name, bases, attributes):
        self = super().__new__(cls, name, bases, attributes,)
        logger.info(f"Creating cls, {type(cls), cls, self, type(self)}")
        logger.info(f"Attributes: {attributes}")
        logger.info(f"Metaclass: {self.__abstractmethods__}")
        logger.info(f"MRO: {self.__mro__}")

        for base in self.__mro__[1:]:
            logger.info(f"Base class: {base}")
            if hasattr(base, "__abstractmethods__"):
                for func in base.__abstractmethods__:
                    try:
                        current_func_code = attributes[func].__code__
                    except KeyError:
                        raise TypeError(f"Class must define abstract method {func}")

                    base_func_params = getattr(base, func).__code__.co_varnames
                    logger.info(f"Function {func}'s parameters: {current_func_code.co_varnames[:current_func_code.co_argcount]}")
                    if current_func_code.co_varnames[:current_func_code.co_argcount]!=base_func_params:
                        raise TypeError(f"Class must define {func} with these parameters: {base_func_params}, "
                                        f"but has {current_func_code.co_varnames[:current_func_code.co_argcount]} instead.")


        return self

class BaseModel(metaclass=ABCMeta):
    checkpoint_name = "model.ckpt"
    config_name = "config.json"

    @abstractmethod
    def forward(self, *args, **kwargs):
        """forward method of a nn.Module"""

    @abstractmethod
    def eval(self):
        """set to eval mode"""

    @abstractmethod
    def state_dict(self):
        """return the state dict of the module"""

    @abstractmethod
    def load_state_dict(self, *args, **kwargs):
        """load the state dict"""

    def predict(self, input_: np.array,
                threshold=0.5,
                transformations: Compose=None,
                retrieve_additinal_outputs: bool = False):
        """
        Predict from given input image. The input image can be just loaded image as numpy array with the following
        shape : batch x H x W x C or b x H x W (if the task is binary prediction).
        All preprocessing steps are carried out in this method.
        :param input_:
        :param threshold:
        :param transformations:
        :return: Output shape -> Batch x H x W
        """
        if transformations is None:
            if len(input_.shape) == 3:
                input_ = torch.LongTensor(input_)
                transformations = Compose([standard_scaling, to_rgb, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            else:
                input_ = torch.LongTensor(input_.transpose(0, 3, 1, 2))
                transformations = Compose([standard_scaling, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        # permute the dimensions of the image to C, H, W
        self.eval()
        input_ = transformations(input_)
        out, intermediate_outputs, attention_masks = self.forward(input_, return_attentions=True)
        if threshold is not None:
            out = F.sigmoid(out)

            if out.shape[1] > 1:
                out = torch.argmax(out, dim=1)

            else:
                out = (out >= threshold).type(torch.int32).squeeze(1) # squeeze the channel dimension

        if retrieve_additinal_outputs:

            return out, intermediate_outputs, attention_masks
        return out # shape B x H x W

    @classmethod
    def from_pretrained(cls, pretrained_path: str):

        model_path = os.path.join(pretrained_path, BaseModel.checkpoint_name)
        config_path = os.path.join(pretrained_path, BaseModel.config_name)
        model = cls.from_config(config_path)
        model.load_state_dict(torch.load(model_path))
        return model

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        config = BaseModel.inverse_jsonify(config)
        model = cls(**config)
        return model

    def get_initialization_variables(self):
        """
        Retrieve the values of the parameters of __init__ as  dict with key being the name of the parameter
        and value its value in the object.
        :return:
        """
        var_list = self.__class__.__dict__['__init__'].__code__.co_varnames[1:]
        parameters_dict = {}
        for i in var_list:
            try:
                parameters_dict[i] = self.__dict__[i]

            except KeyError:
                continue
        return parameters_dict

    def save(self, save_path: str):
        os.mkdir(save_path)
        torch.save(self.state_dict(), os.path.join(save_path, BaseModel.checkpoint_name))
        with open(os.path.join(save_path, BaseModel.config_name), 'w') as f:
            json.dump(self.jsonify(), f)


    def jsonify(self) -> dict:
        variables_prepared = {}
        for key, value in self.get_initialization_variables().items():
            if isinstance(value, type):
                variables_prepared[key] = repr(value)
            if isinstance(value, list):
                value_ = []
                for element in value:
                    if isinstance(element, type):
                        value_.append(repr(element))
                    else:
                        value_.append(element)
                variables_prepared[key] = value_
            else:
                variables_prepared[key] = value
        return variables_prepared

    @staticmethod
    def inverse_jsonify(loaded_variables: dict) -> dict:
        variables_postprocessed = {}
        def preprocess_class_repr(string):
            value = re.sub("<class '", "", string)
            value = re.sub("'>", "", value)
            definition_module = value.split('.')[-1]
            definition_package = '.'.join(value.split('.')[:-1])
            if ' ' in value:
                raise Exception(f"Illegal character in {string}")
            value = import_module(definition_package).__dict__[definition_module]
            return value

        for key, value in loaded_variables.items():
            if isinstance(value, str) and value.startswith("<class"):
                variables_postprocessed[key] = preprocess_class_repr(value)

            if isinstance(value, list):
                value_ = []
                for element in value:
                    if isinstance(element, str) and element.startswith("<class"):
                        value_.append(preprocess_class_repr(element))
                    else:
                        value_.append(element)
                variables_postprocessed[key] = value_
            else:
                variables_postprocessed[key] = value
        return variables_postprocessed





    def compute_labels_from_attention_maps(self):
        """
        Compute labels from attention maps
        :return:
        """