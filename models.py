from lib import *
import torch.nn.functional as F
"""
Models and Loss Functions Module

This module combines functionalities from two previously separate model definition files, ensuring a unified
and comprehensive set of model architectures and loss functions for various machine learning tasks. The module
supports models for linear regression, multi-layer perceptrons, GRUs, and multimodal inputs, and includes custom 
loss functions like MSE, CE, and CCC.

Classes:
    - myLin: A linear regression model.
    - myMLP: A multi-layer perceptron model.
    - myLin_multimodal: A linear regression model for multimodal inputs.
    - myGRU: A GRU-based model for sequence data.
    - CustomGRU: A customizable GRU-based model with multiple layers.
    - myGRU_mul: A GRU-based model for multimodal inputs.

Loss Functions:
    - MSEWrapper: A wrapper for Mean Squared Error loss.
    - CEWrapper: A wrapper for Cross Entropy loss using CCC.
    - CCC_Loss: A loss function based on Concordance Correlation Coefficient (CCC).
    - CCC_Loss_b: A batch-based CCC loss function.

Main Features:
    - Various neural network architectures for different tasks.
    - Custom loss functions to handle specific evaluation metrics.
    - Support for multimodal inputs and attention mechanisms.
    - Encapsulation of common machine learning model components.

Usage:
This module should be imported and utilized within the context of a larger experimental framework, where it can 
define and manage the machine learning models and loss functions for different experiments.

"""

class myLin(nn.Module):
    def __init__(self, feat_size, output_size=1):
        super().__init__()
        self.feat_size = feat_size
        self.output_size = output_size
        self.lin = nn.Linear(feat_size, output_size)

    def forward(self, x):
        output = self.lin(x)
        output = torch.mean(output, 1)
        return output

class myMLP(nn.Module):
    def __init__(self, feat_size, hidden_size=50, output_size=1):
        super().__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lin1 = nn.Linear(feat_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.lin1(x)
        output = self.lin2(output)
        output = torch.mean(output, 1)
        return output


class myLin_multimodal(nn.Module):
    def __init__(self, feat_sizeT, feat_sizeA, feat_sizeV, output_size=1):
        super().__init__()
        self.feat_sizeT = feat_sizeT
        self.feat_sizeA = feat_sizeA
        self.feat_sizeV = feat_sizeV
        self.output_size = output_size
        self.linT = nn.Linear(feat_sizeT, output_size)
        self.linA = nn.Linear(feat_sizeA, output_size)
        self.linV = nn.Linear(feat_sizeV, output_size)
        self.lin = nn.Linear(3*output_size, output_size)

    def forward(self, xT, xA, xV):
        outputT = self.linT(xT)
        outputA = self.linA(xA)
        outputV = self.linV(xV)
        outputT = torch.mean(outputT, 1)
        outputA = torch.mean(outputA, 1)
        outputV = torch.mean(outputV, 1)
        output = torch.cat((outputT, outputA, outputV), -1)
        output = self.lin(output)
        return output

class myGRU(nn.Module):
    def __init__(self, feat_size, hidden_size=256, num_layers=3, output_size=1):
        super().__init__()
        self.feat_size = feat_size
        self.output_size = output_size
        self.rnn = nn.GRU(
            input_size=feat_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.lin(output)
        return output


class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, num_layers=1):
        super(CustomGRU, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        self.gru_layers = nn.ModuleList()
        for i in range(self.num_layers):
            input_dim = self.input_size if i == 0 else self.hidden_sizes[i - 1]
            output_dim = self.hidden_sizes[i]
            self.gru_layers.append(nn.GRU(input_dim, output_dim, batch_first=True))

        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)

    def forward(self, x):
        for gru_layer in self.gru_layers:
            x, _ = gru_layer(x)
        output = self.output_layer(x)
        return output

class myGRU_mul(nn.Module):
    def __init__(self, audio_feat_size, video_feat_size, text_feat_size, hidden_size=256, num_layers=1, output_size=1):
        super().__init__()
        self.audio_feat_size = audio_feat_size
        self.video_feat_size = video_feat_size
        self.text_feat_size = text_feat_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        combined_feat_size = audio_feat_size + video_feat_size + text_feat_size
        
        self.rnn = nn.GRU(
            input_size=combined_feat_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            batch_first=True)

        self.lin = nn.Linear(hidden_size, output_size)
        self.weight = torch.nn.Parameter(data=torch.tensor(0.5), requires_grad=True)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = self.lin(output)
        out = torch.mul(output, self.weight)
        return out

# Losses
class MSEWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, outputs, targets, length=None):
        if type(targets) == sb.dataio.batch.PaddedData:
            targets = targets[0]
        if targets.size()[1] == 1:
            targets = targets.squeeze(1)
        outputs = outputs.squeeze(1)
        targets = targets.float()
        predictions = outputs
        loss = self.criterion(predictions, targets)
        return loss

class CEWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = CCC_Loss()

    def forward(self, outputs, targets, length=None):
        outputs = outputs.squeeze(1)
        targets = torch.tensor(targets, device='cuda:0').squeeze(1)
        loss = 1 - self.criterion(outputs, targets)
        return loss

def calc_CCC(tensor1, tensor2):
    """Calculating Concordance Correlation Coefficient (CCC) using pytorch
    This method allows backpropagation and being used as the loss function

    Arguments
    ---------
    tensor1 : torch.FloatTensor
        one dimensional torch tensor representing the list of values for the first tensor.
    tensor2 : torch.FloatTensor
        one dimensional torch tensor representing the list of values for the first tensor.

    Returns
    -------
    torch.FloatTensor
        Single value torch tensor for calculated CCC

    Example
    -------
    >>> ccc = calc_CCC(torch.rand(50), torch.rand(50))
    """
    min_size = min(tensor1.size(0), tensor2.size(0))
    if tensor1.size(0) != tensor2.size(0):
        tensor1 = tensor1[:min_size]
        tensor2 = tensor2[:min_size]
    mean_gt = torch.mean(tensor2, 0)
    mean_pred = torch.mean(tensor1, 0)
    var_gt = torch.var(tensor2, 0)
    var_pred = torch.var(tensor1, 0)
    v_pred = tensor1 - mean_pred
    v_gt = tensor2 - mean_gt
    denominator = var_gt + var_pred + (mean_gt - mean_pred) ** 2
    cov = torch.mean(v_pred * v_gt)
    numerator = 2 * cov
    ccc = numerator / denominator
    return ccc

class CCC_Loss(torch.nn.Module):
    """The torch nn module based class for using CCC as the loss.
    This class allows for better encapsulation than using the CCC loss function.
    """
    def __init__(self):
        super(CCC_Loss, self).__init__()

    def forward(self, prediction, ground_truth):
        loss = calc_CCC(prediction.view(-1), ground_truth.view(-1))
        loss = torch.mean(loss)
        return loss

class CCC_Loss_b(torch.nn.Module):
    """The torch nn module based class for using CCC as the loss.
    This class allows for better encapsulation than using the CCC loss function.
    This class applies CCC across the batch size. Then, averages over different targets.
    """
    def __init__(self):
        super().__init__()

    def forward(self, prediction, ground_truth):
        if type(ground_truth) == sb.dataio.batch.PaddedData:
            ground_truth = ground_truth[0]
        size = prediction.size()
        for i in range(size[1]):
            loss = 1 - calc_CCC(prediction[:, i], ground_truth[:, i])
            loss = torch.mean(loss)
            if i == 0:
                losses = loss
            else:
                losses += loss
        loss = losses / size[1]
        return loss
