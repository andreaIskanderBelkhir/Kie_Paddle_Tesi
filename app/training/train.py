# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import yaml
import paddle
import paddle.distributed as dist
from paddleocr import ppocr
from ppocr.data import build_dataloader
from ppocr.modeling.architectures import build_model
from app.apps.losses import build_loss
from app.apps.optimizer import build_optimizer
from ppocr.postprocess import build_post_process
from app.apps.metrics import build_metric
from ppocr.utils.save_load import load_model
from ppocr.utils.utility import set_seed
from ppocr.modeling.architectures import apply_to_static
import app.apps.tools.program as program


dist.get_world_size()


class Train(object):

    """docstring for Train"""
    def __init__(self,conf_yml):

        config = yaml.load(open(conf_yml, 'rb'),Loader=yaml.Loader)
        self.config, self.device, self.logger, self.vdl_writer = program.preprocess(config,is_train=True)
        self.seed = self.config['Global']['seed'] if 'seed' in self.config['Global'] else 1024
        

    def __call__(self):
        set_seed(self.seed)
        # init dist environment
        if self.config['Global']['distributed']:
            dist.init_parallel_env()
    
        global_config = self.config['Global']
    
            # build dataloader
        train_dataloader = build_dataloader(self.config, 'Train', self.device, self.logger)
        if len(train_dataloader) == 0:
                self.logger.error(
                    "No Images in train dataset, please ensure\n" +
                    "\t1. The images num in the train label_file_list should be larger than or equal with batch size.\n"
                    +
                    "\t2. The annotation file and path in the configuration file are provided normally."
                )
                return
    
        if self.config['Eval']:
                valid_dataloader = build_dataloader(self.config, 'Eval', self.device, self.logger)
        else:
                valid_dataloader = None
    
            # build post process
        post_process_class = build_post_process(self.config['PostProcess'],
                                                    global_config)
    
            # build model
            # for rec algorithm
        if hasattr(post_process_class, 'character'):
                char_num = len(getattr(post_process_class, 'character'))
                if self.config['Architecture']["algorithm"] in ["Distillation",
                                                           ]:  # distillation model
                    for key in self.config['Architecture']["Models"]:
                        if self.config['Architecture']['Models'][key]['Head'][
                                'name'] == 'MultiHead':  # for multi head
                            if self.config['PostProcess'][
                                    'name'] == 'DistillationSARLabelDecode':
                                char_num = char_num - 2
                            # update SARLoss params
                            assert list(self.config['Loss']['loss_config_list'][-1].keys())[
                                0] == 'DistillationSARLoss'
                            self.config['Loss']['loss_config_list'][-1][
                                'DistillationSARLoss']['ignore_index'] = char_num + 1
                            out_channels_list = {}
                            out_channels_list['CTCLabelDecode'] = char_num
                            out_channels_list['SARLabelDecode'] = char_num + 2
                            self.config['Architecture']['Models'][key]['Head'][
                                'out_channels_list'] = out_channels_list
                        else:
                            self.config['Architecture']["Models"][key]["Head"][
                                'out_channels'] = char_num
                elif self.config['Architecture']['Head'][
                        'name'] == 'MultiHead':  # for multi head
                    if self.config['PostProcess']['name'] == 'SARLabelDecode':
                        char_num = char_num - 2
                    # update SARLoss params
                    assert list(self.config['Loss']['loss_config_list'][1].keys())[
                        0] == 'SARLoss'
                    if self.config['Loss']['loss_config_list'][1]['SARLoss'] is None:
                        self.config['Loss']['loss_config_list'][1]['SARLoss'] = {
                            'ignore_index': char_num + 1
                        }
                    else:
                        self.config['Loss']['loss_config_list'][1]['SARLoss'][
                            'ignore_index'] = char_num + 1
                    out_channels_list = {}
                    out_channels_list['CTCLabelDecode'] = char_num
                    out_channels_list['SARLabelDecode'] = char_num + 2
                    self.config['Architecture']['Head'][
                        'out_channels_list'] = out_channels_list
                else:  # base rec model
                    self.config['Architecture']["Head"]['out_channels'] = char_num
    
                if self.config['PostProcess']['name'] == 'SARLabelDecode':  # for SAR model
                    self.config['Loss']['ignore_index'] = char_num - 1
    
        model = build_model(self.config['Architecture'])
    
        use_sync_bn = self.config["Global"].get("use_sync_bn", False)
        if use_sync_bn:
                model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)
                self.logger.info('convert_sync_batchnorm')
    
        model = apply_to_static(model, self.config, self.logger)
    
            # build loss
        loss_class = build_loss(self.config['Loss'])
    
            # build optim
        optimizer, lr_scheduler = build_optimizer(
                self.config['Optimizer'],
                epochs=self.config['Global']['epoch_num'],
                step_each_epoch=len(train_dataloader),
                model=model)
    
            # build metric
        eval_class = build_metric(self.config['Metric'])
    
        self.logger.info('train dataloader has {} iters'.format(len(train_dataloader)))
        if valid_dataloader is not None:
            self.logger.info('valid dataloader has {} iters'.format(
                len(valid_dataloader)))
    
        use_amp = self.config["Global"].get("use_amp", False)
        amp_level = self.config["Global"].get("amp_level", 'O2')
        amp_custom_black_list = self.config['Global'].get('amp_custom_black_list', [])
        if use_amp:
                AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
                if paddle.is_compiled_with_cuda():
                    AMP_RELATED_FLAGS_SETTING.update({
                        'FLAGS_cudnn_batchnorm_spatial_persistent': 1
                    })
                paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)
                scale_loss = self.config["Global"].get("scale_loss", 1.0)
                use_dynamic_loss_scaling = self.config["Global"].get(
                    "use_dynamic_loss_scaling", False)
                scaler = paddle.amp.GradScaler(
                    init_loss_scaling=scale_loss,
                    use_dynamic_loss_scaling=use_dynamic_loss_scaling)
                if amp_level == "O2":
                    model, optimizer = paddle.amp.decorate(
                        models=model,
                        optimizers=optimizer,
                        level=amp_level,
                        master_weight=True)
        else:
                scaler = None
    
            # load pretrain model
        pre_best_model_dict = load_model(self.config, model, optimizer,
                                             self.config['Architecture']["model_type"])
    
        if self.config['Global']['distributed']:
            model = paddle.DataParallel(model)
            # start train
        program.train(self.config, train_dataloader, valid_dataloader, self.device, model,
                          loss_class, optimizer, lr_scheduler, post_process_class,
                          eval_class, pre_best_model_dict, self.logger, self.vdl_writer, scaler,
                          amp_level, amp_custom_black_list)
    



