import torch
import torch.nn as nn


class MultitaskLoss4OneStructured(nn.Module):
    def __init__(self, hparams, weight=None):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.nll_loss = torch.nn.NLLLoss(weight=weight)

        
        self.alpha1 = hparams.alpha1
        self.alpha2 = hparams.alpha2
        self.eps = 1e-10
    def forward(self, inputs, targets):
        # Implement your customized loss function here
        # Calculate the loss based on the inputs and targets
        # Return the loss value
        main_probs, structured_logits, notes_logits = inputs

        main_loss = self.nll_loss(torch.log(main_probs), targets)
        # aux_loss = sum([self.ce_loss(logits[i], targets) for i in range(len(logits))])
        
        # loss = main_loss + self.alpha * aux_loss
        aux_loss_1 = self.ce_loss(structured_logits, targets)
        aux_loss_2 = self.ce_loss(notes_logits, targets)
           
        return main_loss / (main_loss.detach() + self.eps) + self.alpha1*aux_loss_1 / (aux_loss_1.detach() + self.eps) + self.alpha2*aux_loss_2 / (aux_loss_2.detach() + self.eps)

class MultitaskLoss4OneStructured_detached(nn.Module):
    def __init__(self, hparams, weight=None):
        super().__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=weight)
        self.nll_loss = torch.nn.NLLLoss(weight=weight)

        self.alpha1 = hparams.alpha1
        self.alpha2 = hparams.alpha2
        self.eps = 1e-10
    def forward(self, inputs, targets):
        # Implement your customized loss function here
        # Calculate the loss based on the inputs and targets
        # Return the loss value
        main_probs, structured_logits, notes_logits = inputs

        main_loss = self.nll_loss(torch.log(main_probs), targets)
        # aux_loss = sum([self.ce_loss(logits[i], targets) for i in range(len(logits))])
        
        # loss = main_loss + self.alpha * aux_loss
        aux_loss_1 = self.ce_loss(structured_logits, targets)
        aux_loss_2 = self.ce_loss(notes_logits, targets)
           
        return (main_loss + self.alpha1*aux_loss_1 + self.alpha2*aux_loss_2).detach()
