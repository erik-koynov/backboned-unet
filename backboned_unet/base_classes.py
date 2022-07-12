from abc import ABCMeta, abstractmethod
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
import logging
import numpy as np
import sys
from pathlib import Path
import os
print(os.path.abspath(__file__))
print(Path(__file__))
print(os.getcwd())
file_path = Path(os.getcwd())/Path(__file__).parent
sys.path.append(str(file_path/Path('../../')))
sys.path.append(str(file_path/Path('../../invertransforms')))
#print(sys.path)
from invertransforms.lib import Invertible
from datasets.custom_augmentations import standard_scaling, to_rgb
from tta_tools import FlipsForTTA
from inference_tools import compute_attention_cam_from_centroid
logger = logging.getLogger("backboned_unet")
import os
import json
import re
from importlib import import_module
from typing import Union
from copy import deepcopy

class StrictMeta(ABCMeta):
    def __new__(mcs,  name: str, bases: tuple, attributes: dict):
        """
        Create a new CLASS!
        Force the child class methods that have been named as abstract functions in a base class
        to have the same signature as in the abstract class
        :param name: name of the class to create
        :param bases: the base classes the new class is inheriting from
        :param attributes: class attributes
        """
        cls = super().__new__(mcs, name, bases, attributes,)
        logger.info(f"Creating cls, {type(mcs), mcs, cls, type(cls)}")
        logger.info(f"Attributes: {attributes}")
        logger.info(f"Metaclass: {mcs}")
        logger.info(f"MRO: {cls.__mro__}")

        return cls

    def __call__(cls, *initialization_args, **initialization_kwargs):
        attributes = cls.__dict__
        for base in cls.__mro__[1:]:
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
        return super().__call__(*initialization_args, **initialization_kwargs)

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

    def mc_dropout_predictions(self,
                               input_: Union[np.array,torch.Tensor],
                               threshold: float = 0.5,
                               transformations: Compose=None,
                               num_samples: int = 20,
                               attention_mask_idx: int = None,
                               scale_attn_map=True,
                               do_preprocess=True):
        """
        return all predictions made in N stochastic passes through the algorithm
        :return: T x C x H x W
        """
        thresholded_predictions = []
        predictions = []
        attention_masks = []
        for sample in range(num_samples):
            out_thresholded, _, attention_masks_, out = self.predict(input_,
                                                                     threshold=threshold,
                                                                     transformations=transformations,
                                                                     retrieve_additinal_outputs=True,
                                                                     keepdim=False,
                                                                     training_mode=True,
                                                                     do_preprocess=do_preprocess)
            thresholded_predictions.append(out_thresholded.detach().cpu())
            predictions.append(out.detach().cpu())
            attn_mask = compute_attention_cam_from_centroid(attention_masks_[attention_mask_idx],
                                                            predictions=out_thresholded,
                                                            scale=scale_attn_map,
                                                            resize_to_input=True)[0] # retain only the attn mask on x
            attention_masks.append(attn_mask[None,None,...].cpu())

        return torch.cat(thresholded_predictions), torch.cat(predictions), torch.cat(attention_masks)


    def tt_augmentation_predictions(self,
                                    input_: Union[np.array,torch.Tensor],
                                    ttest_time_augmentation: Invertible=None,
                                    threshold: float = 0.5,
                                    transformations: Compose=None,
                                    num_samples: int = 12,
                                    attention_mask_idx: int = None,
                                    do_preprocess=True,
                                    training_mode=False,
                                    return_transformed_inputs=False,
                                    fill_training_sub=0.,
                                    ):
        """
        input_: single image (for now)
        :return: B x C x H x W
        """
        tta_flips = FlipsForTTA()
        transformed_inputs = []

        thresholded_predictions = []
        predictions = []
        attention_masks = []
        augmentation_params = []

        fill = ttest_time_augmentation.fill

        for sample in range(num_samples+len(tta_flips)):

            if do_preprocess and sample == 0: # preprocess (once!) before the test time augmentation!
                input_ = self._preprocess(input_, transformations)

            if sample < len(tta_flips):
                transformed_input = tta_flips(input_)
                inv_transform = tta_flips.inverse_transform
            else:

                transformed_input = ttest_time_augmentation(input_)
                inv_transform = ttest_time_augmentation.inverse_transform

            if fill is not None:
                fill_mask = (transformed_input == fill)
            else:
                fill_mask = torch.full_like(transformed_input, False)

            print("shape fill_mask: ", fill_mask.shape)
            do_fill = fill_mask.sum() > 0
            if do_fill: #
                transformed_input[fill_mask] = fill_training_sub
                fill_mask = fill_mask[:, 0:1, ...]
            augmentation_params.append(deepcopy(ttest_time_augmentation)) # save current state of the random variables

            if return_transformed_inputs:
                transformed_inputs.append(transformed_input)

            out_thresholded, _, attention_masks_, out = self.predict(transformed_input,
                                                                     threshold=threshold,
                                                                     transformations=None,
                                                                     retrieve_additinal_outputs=True,
                                                                     keepdim=True,
                                                                     training_mode=training_mode,
                                                                     do_preprocess=False)
            print(f"out shape: {out.shape}")

            aug_out_th = inv_transform(out_thresholded.detach().cpu())
            aug_out = inv_transform(out.detach().cpu())
            attn_mask = compute_attention_cam_from_centroid(attention_masks_[attention_mask_idx],
                                                            predictions=out_thresholded,
                                                            resize_to_input=True)[0] # retain only the attn mask on x
            aug_attn = inv_transform(attn_mask[None, None, ...].cpu())

            if fill_mask.sum() > 0:
                print(f'FILL is {fill}')
                aug_out[fill_mask] = fill
                aug_out_th[fill_mask] = fill
                aug_attn[fill_mask] = fill

            thresholded_predictions.append(aug_out_th.squeeze(1))
            predictions.append(aug_out)
            attention_masks.append(aug_attn)

        if return_transformed_inputs:
            return torch.cat(thresholded_predictions), torch.cat(predictions), torch.cat(attention_masks), augmentation_params, transformed_inputs

        return torch.cat(thresholded_predictions), torch.cat(predictions), torch.cat(attention_masks), augmentation_params





    def predict(self, input_: np.array,
                threshold=0.5,
                transformations: Compose=None,
                retrieve_additinal_outputs: bool = False,
                keepdim = False,
                training_mode = False,
                do_preprocess = True):
        """
        Predict from given input image. The input image can be just loaded image as numpy array with the following
        shape : batch x H x W x C or b x H x W (if the task is binary prediction).
        All preprocessing steps are carried out in this method.
        :param input_:
        :param threshold:
        :param transformations:
        :return: Output shape -> Batch x( 1 )x H x W
        """

        # permute the dimensions of the image to C, H, W
        if do_preprocess:
            input_ = self._preprocess(input_, transformations)
        if not training_mode:
            self.eval()
        else:
            self.train()

        if retrieve_additinal_outputs:
            return self._predict_on_preprocessed_input(input_, threshold, keepdim, retrieve_additinal_outputs=True)
        return self._predict_on_preprocessed_input(input_, threshold, keepdim, retrieve_additinal_outputs=False)

    def _predict_on_preprocessed_input(self,
                                       input_: torch.Tensor,
                                       threshold: float = 0.5,
                                       keepdim=False,
                                       retrieve_additinal_outputs: bool = False
                                       ):
        out, intermediate_outputs, attention_masks = self.forward(input_, return_attentions=True)
        out = F.sigmoid(out)
        out_thresholded = self._apply_treshold(out, threshold, keepdim)
        if retrieve_additinal_outputs:
            return out_thresholded, intermediate_outputs, attention_masks, out
        return out_thresholded, out  # shape B x H x W


    def _apply_treshold(self, model_output: torch.Tensor, threshold: float = 0.5, keepdim=False):
        if threshold is not None:

            if model_output.shape[1] > 1:  # if there are more than 2 classes
                out = torch.argmax(model_output, dim=1, keepdim=keepdim)
                logger.warning("Model output has more than one output channel, threshold will be ignored, argmax will"
                               f"will be output!Output shape: {model_output.shape}")

            else:

                out = (model_output >= threshold).type(torch.int32)
                if not keepdim:
                    out = out.squeeze(1)  # squeeze the channel dimension
            return out
        else:
            return model_output

    @classmethod
    def from_pretrained(cls, pretrained_path: str):

        model_path = os.path.join(pretrained_path, BaseModel.checkpoint_name)
        config_path = os.path.join(pretrained_path, BaseModel.config_name)
        model = cls.from_config(config_path)
        try:
            model.load_state_dict(torch.load(model_path,))
        except RuntimeError:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        return model

    def _preprocess(self,
                    input_: np.array,
                    transformations: Compose = None):
        if transformations is None:
            if len(input_.shape) == 3: # B x H x W
                input_ = torch.LongTensor(input_)
                transformations = Compose([standard_scaling, to_rgb, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            elif len(input_.shape) == 4:
                input_ = torch.LongTensor(input_.transpose(0, 3, 1, 2))
                transformations = Compose([standard_scaling, Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            else:
                raise Exception(f"Input is of type: {type(input_)}, with dimensions: {input_.shape}. Possible causes"
                                f"lack of batch dimension")
        return transformations(input_)


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
                logger.info(f"Parameter {i} not found among the member variables.")
                continue
        return parameters_dict

    def save(self, save_path: str, overwrite = False):
        if not overwrite:
            try:
                os.mkdir(save_path)
            except FileExistsError:
                raise Exception(f"Directory {save_path} exists and overwrite parameter is set to False.")
        else:
            Path(save_path).mkdir(exist_ok=True)

        torch.save(self.state_dict(), os.path.join(save_path, BaseModel.checkpoint_name))
        with open(os.path.join(save_path, BaseModel.config_name), 'w') as f:
            obj_as_dict = self.jsonify()
            json.dump(obj_as_dict, f)


    def jsonify(self) -> dict:
        variables_prepared = {}
        for key, value in self.get_initialization_variables().items():
            print(key, value, isinstance(value, type))
            if isinstance(value, type):
                variables_prepared[key] = repr(value)
            elif isinstance(value, list):
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

            elif isinstance(value, list):
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