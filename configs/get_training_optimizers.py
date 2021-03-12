import logging
import torch
import torchvision
from torch import nn


from configs.enums_file import OPTIMIZERS,LOSSES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#-------------------------------Training Optimizers---------------------------------------

def get_optimizer(optimizer_type):
    if optimizer_type == OPTIMIZERS.ADAM.value:
        optimizer = torch.optim.Adam
        return optimizer
    elif optimizer_type == OPTIMIZERS.Adam_W.value:
        optimizer = torch.optim.AdamW
        return optimizer
    else:
        logging.error("Wrong optimizer")

def get_loss_function(loss_func):
    if loss_func == LOSSES.Cross_Entropy.value:
        loss_function = nn.CrossEntropyLoss().to(device)
        return loss_function
    else:
        logging.error("Wrong loss function")