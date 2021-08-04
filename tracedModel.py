#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
:Description: Pytorch Model conversion into the

:Authors: (c) Artem Lutov <lav@lumais.com>
:Date: 2021-08-03
"""
import torch

PATH = "AnTroD_resnet50.pt"
device = torch.device('cpu')

model = torch.load(PATH, map_location=device)
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("AnTroD_resnet50_traced.pt")
# Verify that the model can be loaded without exceptions
tm = torch.jit.load("AnTroD_resnet50_traced.pt")
