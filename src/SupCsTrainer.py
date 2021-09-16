"""
General class implementing Supervised Contrastive loss.

Dependencies: losses.py

Version 0.0.1

Initial Release.

Hooman Sedghamiz, July 2021.
"""
from __future__ import print_function
from transformers import Trainer
from transformers import AutoModel
import torch
import torch.nn.functional as FF
import torch.nn as nn
import typing as tp
from transformers import TrainingArguments



class SupCsTrainer(Trainer):
    def __init__(
        self,
        w_drop_out: tp.Optional[tp.List[float]] = [0.0,0.05,0.2],
        temperature: tp.Optional[float] = 0.05,
        def_drop_out: tp.Optional[float]=0.1,
        pooling_strategy: tp.Optional[str]='pooler',
        **kwargs
        ):
        """Initialize Contrastive Learning Module and params.
        
        :param w_drop_out: Drop_out probabilities. THe number of views equals to the length of this list.
        
        :param pooling_strategy: Pooling strategy it can be either `pooler` or `mean`. 
        `pooler` employs the `[CLS]` token if available and `mean` takes the mean of the last hidden layer.
        
        """
        super().__init__(**kwargs)
        self.w_drop_out = w_drop_out
        self.temperature_s = temperature 
        self.def_drop_out = def_drop_out
        self.pooling_strategy = pooling_strategy
        if pooling_strategy == 'pooler':
            print('# Employing pooler ([CLS]) output.')
        else:
            print('# Employing mean of the last hidden layer.')
        
    def compute_loss(
        self,
        model: nn,
        inputs: tp.Dict,
        return_outputs: tp.Optional[bool]=False,
        )-> tp.Tuple[float, torch.Tensor]:
        """ Computes loss given model and its inputs.
        
        :param model: The model to be finetuned.   
        
        :param inputs: The inputs
        
        """
        labels = inputs.pop("labels")
        
        # ----- Default p = 0.1 ---------#
        output = model(**inputs)
        if self.pooling_strategy == 'pooler':
            try:
                logits = output.pooler_output.unsqueeze(1) 
            except:
                logits = output.last_hidden_state.mean(dim=1, keepdim=True)
        else:
            logits = output.last_hidden_state.mean(dim=1, keepdim=True)
        
        # ---- iteratively create dropouts -----#
        for p_dpr in self.w_drop_out:
            # -- Set models dropout --#
            if p_dpr != self.def_drop_out:
                model = self.set_dropout_mf(model, w=p_dpr)
            # ---- concat logits ------#
            if self.pooling_strategy == 'pooler':
                # --------- If model does offer pooler output --------#
                try:
                    logits = torch.cat((logits, model(**inputs).pooler_output.unsqueeze(1)), 1)
                except:
                    logits = torch.cat((logits, model(**inputs).last_hidden_state.mean(dim=1, keepdim=True)), 1)
            else:
                logits = torch.cat((logits, model(**inputs).last_hidden_state.mean(dim=1, keepdim=True)), 1)
            
        # ---- L2 norm ---------#
        logits = FF.normalize(logits, p=2, dim=2)
        
        #----- Set model back to dropout = 0.1 -----#
        if p_dpr != self.def_drop_out: model = self.set_dropout_mf(model, w=0.1)
        
        
        # SupContrast
        loss_fn = SupConLoss(temperature=self.temperature_s) # temperature=0.1

        loss = loss_fn(logits, labels) # added rounding for stsb
        
        return (loss, outputs) if return_outputs else loss
    
    def set_dropout_mf(
        self, 
        model:nn, 
        w:tp.List[float]
        ):
        """Alters the dropouts in the embeddings.
        """
        # ------ set hidden dropout -------#
        if hasattr(model, 'module'):
            model.module.embeddings.dropout.p = w
            for i in model.module.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w        
        else:
            model.embeddings.dropout.p = w
            for i in model.encoder.layer:
                i.attention.self.dropout.p = w
                i.attention.output.dropout.p = w
                i.output.dropout.p = w
            
        return model

    
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss