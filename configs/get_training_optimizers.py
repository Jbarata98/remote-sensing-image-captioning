import logging
from configs.enums_file import *
#-------------------------------Training Optimizers---------------------------------------

def get_optimizer(optimizer_type):
    if optimizer_type == OPTIMIZER.ADAM.value:
        optimizer = torch.optim.Adam
        return optimizer
    elif optimizer_type == OPTIMIZER.Adam_W.value:
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