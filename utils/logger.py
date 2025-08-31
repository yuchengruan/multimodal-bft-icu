import os
from comet_ml import Experiment

import torch.distributed as dist

class CometNewExperiment(Experiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.curr_step = -1

    def new_step(self):
        self.curr_step += 1

def create_logger(hparams):

    login_info = {"api_key": "xxx", 
                  "project_name": "xxx", 
                  "workspace": "xxx"}

    if (dist.is_torchelastic_launched() is True and int(os.environ['LOCAL_RANK']) == 0) or (dist.is_torchelastic_launched() is not True):
        login_info["disabled"] = not hparams.logger
        if hparams.logger:
            print("="*15, " Initializing logger ", "="*15)
    else:
        login_info["disabled"] = True

    experiment = CometNewExperiment(**login_info)
    experiment.log_parameters(hparams)
    experiment.log_code(folder="./")
    
    return experiment
   