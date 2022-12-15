#!/usr/bin/env python
# coding: utf-8

# In[280]:


import os
import sys
import numpy as np
import torch

from datetime import datetime
from typing import Tuple
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn import KLDivLoss, CrossEntropyLoss, CosineEmbeddingLoss, MSELoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ### Loading Teacher model ---> CLIP Image Extractor

# In[281]:


import clip

model_name = "ViT-B/32"

# model is the torch model.
# preprocess function is for image preprocessing.

model, preprocess = clip.load(model_name)

# Get only the visual model
teacher_model = model.visual
input_resolution = model.visual.input_resolution

print(
    "Model parameters:",
    f"{np.sum([int(np.prod(p.shape)) for p in model.visual.parameters()]):,}",
)
print("Input resolution:", input_resolution)


# ### Instantiating Student model
#
# [VisionTransformer](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L206)

# In[333]:


from clip.model import VisionTransformer
from clip.model import convert_weights  # Make them float16

# Set Student Configuration

patch_size = 32
width = 384
layers = 6
heads = 12
output_dim = 512

student_model = VisionTransformer(
    input_resolution=input_resolution,
    patch_size=patch_size,
    width=width,
    layers=layers,
    heads=heads,
    output_dim=output_dim,
)


convert_weights(student_model)


print(
    "Model parameters:",
    f"{np.sum([int(np.prod(p.shape)) for p in student_model.parameters()]):,}",
)


# In[334]:


import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0, 0, 0), (1, 1, 1)),
    ]
)

cifar100 = torchvision.datasets.CIFAR100(
    "data/", download=True, train=True, transform=transform
)
train_dataloader = torch.utils.data.DataLoader(
    cifar100, batch_size=32, shuffle=True, num_workers=8
)


# In[335]:


class DistillationTrainer:
    def __init__(self, *args, **kwargs):
        self.teacher = teacher_model
        self.student = student_model
        self.train_dataloader = train_dataloader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        self.teacher.eval()

        self.epochs = 30
        self.start_epoch = 1

        # set up optimizer
        self.optimizer = SGD(self.student.parameters(), lr=0.001)

        # Set up LR Scheduler

    #         self.lr_scheduler = ReduceLROnPlateau(self.optimizer, "min")

    def compute_loss(self, images, return_outputs=False):
        images = images.to(self.device).half()

        outputs_student = self.student(images)

        # compute teacher output
        with torch.no_grad():
            outputs_teacher = self.teacher(images)
        #             outputs_teacher = torch.tensor(outputs_teacher.detach().cpu().numpy()).to(self.device)

        # assert size
        assert outputs_student.size() == outputs_teacher.size()

        # Soften probabilities and compute distillation loss

        #         KL Divergence Loss
        kl_loss = KLDivLoss(reduction="batchmean", log_target=True)
        loss = kl_loss(F.log_softmax(outputs_student), F.log_softmax(outputs_teacher))

        # Cosine loss
        loss += CosineEmbeddingLoss()(
            outputs_teacher,
            outputs_student,
            torch.ones(outputs_teacher.size()[0]).to(self.device),
        )

        #         #MSE Loss
        #         mse_loss = MSELoss()
        #         loss += mse_loss(outputs_teacher, outputs_student)

        return loss

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            print(f"Starting Epoch Training for Epoch {epoch}")
            loss_value = self._train_epoch(epoch)
            print(f"Epoch {epoch} complete,\n KLD-CosineLoss is {loss_value}")
            if epoch % 10 == 0:
                torch.save(self.student.state_dict(), f"Run_2_DistilledModel{epoch}.pt")

    def _train_epoch(self, epoch):
        loss_value = 0
        for batch_idx, (images, _) in enumerate(self.train_dataloader):

            self.optimizer.zero_grad()
            #             labels = torch.nn.functional.one_hot(labels,num_classes=100)

            loss = self.compute_loss(images)

            loss_value += loss

            loss.backward()

            #             torch.autograd.set_detect_anomaly(True)

            self.optimizer.step()

            #             print("After",self.student.conv1.weight)

            if batch_idx % 100 == 0:
                print(f"Loss after {batch_idx} Batch is {loss_value/batch_idx+1} ")

        return loss_value.detach().cpu().numpy() / len(self.train_dataloader)


# In[336]:


Trainer = DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    train_dataloader=train_dataloader,
    preprocess=preprocess,
)


# In[ ]:


Trainer.train()


# In[ ]:


torch.save(Trainer.student.state_dict(), f"Final_DistilledModel.pt")


# In[ ]:
