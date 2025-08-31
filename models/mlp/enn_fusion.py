from typing import Callable, Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
from models.dst_pytorch import Belief_layer, Dempster_Shafer_Module, Dempster_layer, DempsterNormalize_layer, Distance_layer, DistanceActivation_layer, Omega_layer
ModuleType = Union[str, Callable[..., nn.Module]]
ModuleType0 = Union[str, Callable[[], nn.Module]]
from transformers import AutoTokenizer, AutoModel
from peft import inject_adapter_in_model, LoraConfig

def pignistic(mass, n_class):

    probs = mass[:, :n_class] + (1 / n_class) * mass[:, n_class].unsqueeze(1)
    uncertainty = mass[:, n_class]

    return probs, uncertainty
    

class MassGen(nn.Module):
    def __init__(self, n_feature_maps, n_classes, n_prototypes = 1) -> None:
        super().__init__()

        self.ds1 = Distance_layer(n_prototypes, n_feature_maps)
        self.ds1_activate = DistanceActivation_layer(n_prototypes)
        self.ds2 = Belief_layer(n_prototypes, n_classes)
        self.ds2_omega = Omega_layer(n_prototypes, n_classes)

    def forward(self, inputs):
        ED = self.ds1(inputs)
        ED_ac = self.ds1_activate(ED)
        mass_prototypes = self.ds2(ED_ac)
        mass_prototypes_omega = self.ds2_omega(mass_prototypes)
        return mass_prototypes_omega



class MLPENNOneStructured(nn.Module):
    # combine twice
    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        # structured
        structured_d_in_ls = hparams.structured_d_in_ls
        self.backbone = MLP(
                        d_in=sum(structured_d_in_ls),
                        n_blocks=hparams.n_blocks,
                        d_block=hparams.structured_d_hidden,
                        d_out=hparams.structured_d_hidden,
                        dropout=hparams.dropout,
                        )
        self.structured_cls = nn.Sequential(nn.Dropout(hparams.dropout),
                                            nn.Linear(hparams.structured_d_hidden, hparams.n_class),
                                            )
        self.structured_dsm = Dempster_Shafer_Module(hparams.structured_d_hidden, hparams.n_class, hparams.prototype_dim)

        # notes
        
        self.notes_reducer = nn.Sequential(nn.Dropout(hparams.dropout),
                                            nn.Linear(768, hparams.notes_d_hidden),
                                            )
        self.notes_fcs4logits = nn.Sequential(nn.Dropout(hparams.dropout),
                                            nn.Linear(hparams.notes_d_hidden, hparams.n_class),
                                            )
        self.notes_dsm = Dempster_Shafer_Module(hparams.notes_d_hidden,
                                                                hparams.n_class, hparams.prototype_dim)
        
        # fusion
        self.ds_dempster = Dempster_layer(2, hparams.n_class)
        self.ds_normalize = DempsterNormalize_layer()

    def forward(self, inputs):
        """_summary_

        Args:
            inputs (_type_): a list

        Returns:
            _type_: _description_
        """
        cont_data, cat_data, notes_data = inputs

        structured_feats = self.backbone(torch.cat([cont_data, cat_data], dim=1))
        structured_logits = self.structured_cls(structured_feats)
        structured_mass = self.structured_dsm(structured_feats)

        note_reduced = self.notes_reducer(notes_data)
        notes_logits = self.notes_fcs4logits(note_reduced)
        notes_mass = self.notes_dsm(note_reduced)

        # combine all the mass functions
        mass_ls = [structured_mass, notes_mass]

        # mass_stack: [batch_size, 2, n_class+1]
        mass_stack = torch.stack(mass_ls, dim=1)
        mass_Dempster = self.ds_dempster(mass_stack)
        mass_Dempster_normalize = self.ds_normalize(mass_Dempster)

        probs, uncertainty = pignistic(mass_Dempster_normalize, self.hparams.n_class)

        return probs, structured_logits, notes_logits, uncertainty
    
