#!/usr/bin/env python3

import wandb
import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm
import json
import numpy as np
from matplotlib import pyplot as plt


run = wandb.init(
    project="automathon",
    name="nom-de-votre-equipe",
    config={
        "learning_rate": 0.001,
        "architecture": "-",
        "dataset": "DeepFake Detection Challenge",
        "epochs": 10,
        "batch_size": 10,
    },
)

def train(model, optimizer, loss_fn, train_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            imgs, labels = batch
            output = model(imgs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            run.log({"training_loss": loss.item()}, step=epoch)