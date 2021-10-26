import logging
import torch
from torch import nn
import numpy as np

from src.configs.setters.set_enums import OPTIMIZERS, LOSSES
from src.configs.utils.sup_contrast import SupConLoss
from src.configs.utils.ALS import AsymmetricLoss,AsymmetricLossOptimized,ASLSingleLabel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------Training Optimizers---------------------------------------

# -------------------------------helper functions-----------------------------------------#

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def accuracy_encoder(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# --------------------------------------------------------------------------------------------------#

class Optimizers:
    """
    class to get the optimizers
    """

    def __init__(self, optimizer_type, loss_func, device=DEVICE):
        self.optimizer_type = optimizer_type
        self.device = device
        self.loss = loss_func

    def _get_optimizer(self, params, lr):
        params = (params, lr)
        if self.optimizer_type == OPTIMIZERS.ADAM.value:
            optimizer = torch.optim.Adam(*params)
            return optimizer
        elif self.optimizer_type == OPTIMIZERS.Adam_W.value:
            optimizer = torch.optim.AdamW(*params)
            return optimizer
        else:
            logging.error("Wrong optimizer")

    def _get_loss_function(self):
        if self.loss == LOSSES.Cross_Entropy.value:
            loss_function = nn.CrossEntropyLoss().to(DEVICE)
            return loss_function
        elif self.loss == LOSSES.SupConLoss.value:
            loss_function = SupConLoss(temperature=0.1)
            return loss_function
        elif self.loss == LOSSES.ALS.value:
            loss_function = ASLSingleLabel()
            return loss_function
        else:
            logging.error("Wrong loss function")


class EarlyStopping:
    """
    class that defines EarlyStopping
    """

    def __init__(
            self,
            epochs_limit_without_improvement,
            epochs_since_last_improvement,
            baseline, encoder_optimizer,
            decoder_optimizer,
            period_decay_lr=5,
            mode="loss"
    ):
        self.epochs_limit_without_improvement = epochs_limit_without_improvement
        self.epochs_since_last_improvement = epochs_since_last_improvement
        self.best_loss = baseline
        self.stop_training = False
        self.improved = False
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.period_decay_lr = period_decay_lr

        if mode == "loss":  # loss -> current value needs to be lesser than previous
            self.monitor_op = torch.le
        elif mode == "metric":  # metric -> current value needs to be greater than previous
            self.monitor_op = np.greater
        else:
            raise Exception("unknown mode")

    def check_improvement(self, current_loss):

        if self.monitor_op(current_loss, self.best_loss):

            logging.info("Current %s Best %s",
                         current_loss, self.best_loss)

            self.best_loss = current_loss
            self.epochs_since_last_improvement = 0
            self.improved = True

        else:
            self.epochs_since_last_improvement += 1
            self.improved = False
            logging.info("Val without improvement. Not improving since %s epochs",
                         self.epochs_since_last_improvement)

            if self.epochs_since_last_improvement == self.epochs_limit_without_improvement:
                logging.info("Early stopping")
                self.stop_training = True

            # Decay learning rate if there is no improvement for x consecutive epochs
            if self.epochs_since_last_improvement > 0 and self.epochs_since_last_improvement % self.period_decay_lr == 0:
                logging.info("Decay learning rate")
                if self.decoder_optimizer != None:
                    adjust_learning_rate(self.decoder_optimizer, 0.8)
                if self.encoder_optimizer != None:
                    adjust_learning_rate(self.encoder_optimizer, 0.8)

    def get_number_of_epochs_without_improvement(self):
        return self.epochs_since_last_improvement

    def is_current_val_best(self):
        return self.improved

    def is_to_stop_training_early(self):
        return self.stop_training


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
