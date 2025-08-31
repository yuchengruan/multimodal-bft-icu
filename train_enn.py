import os
import random
from typing import Callable
import numpy as np
from pyparsing import Any, Dict
import torch
from ignite.engine import Events
from ignite.metrics import Loss
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers.stores import EpochOutputStore
from typing import List
import torch
from ignite.metrics import Loss
from ignite.engine import create_supervised_trainer

from data.tabular_enn_fusion import MIMIC_Structured_Notes
from models.mlp.enn_fusion import MLPENNOneStructured
from utils.loss import MultitaskLoss4OneStructured, MultitaskLoss4OneStructured_detached
from utils.options import add_hparams2parser
from utils.logger import create_logger
from utils.metrics import AUPRC, ECE, NLL, AUROC, NPV, BalancedAccuracy, BrierScore, F1, Precision, Recall, Specificity
from utils.torch_utils import count_parameters

hparams_dict = {
# Common arguments
"seed": 10,
"logger": False,
"devices": "7," ,
"batch_size": 32,
"lr": 1.0e-4,
"max_epochs": 100,
"comments": "",


# Dataset arguments
"outcome":  "long_icu",
"n_class": 2,
"data_path": "./data/datasets/mimic/processed",
"structured_d_in_ls": None, # This will be set in the dataset class
"class_weight": None,   # This will be set in the dataset class
"cv_split": 0,




# Model arguments
"model": "mlp_enn_one_structured",
"pretrained_model": "emilyalsentzer/Bio_ClinicalBERT",
"prototype_dim": 20,
"n_blocks": 3,
"structured_d_hidden": 32,
"notes_d_hidden": 128,
"alpha1": 2,
"alpha2": 1,
"dropout":0.1
}
    
def train(hparams: Any, logger: Any) -> None:

    # ********************** prepare for training
    print("="*15, " Preparing for training ", "="*15)
    device = torch.device(int(hparams.devices.split(",")[0]))
    random.seed(hparams.seed)
    np.random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    
    # ********************** dataset
    print("="*15, " Building dataset module ", "="*15 )
    print(f"* Data path: {hparams.data_path}")
    print(f"* Outcome: {hparams.outcome}")
    print(f"* CV split: {hparams.cv_split}")

    train_dataset = MIMIC_Structured_Notes(hparams.data_path, "train", hparams)
    val_dataset = MIMIC_Structured_Notes(hparams.data_path, "val", hparams)
    test_dataset = MIMIC_Structured_Notes(hparams.data_path, "test", hparams)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hparams.batch_size, shuffle=False)

    # ********************** model
    print("="*15, " Building model ", "="*15)
    print(f"* Model: {hparams.model}")
    model = MLPENNOneStructured(hparams)


    model.to(device)
    print("* Trainable model parameters:")
    count_parameters(model)

    # ********************** loss and optimizer
    print("="*15, " Building loss, optimizer, and metrics ", "="*15)
    criterion = MultitaskLoss4OneStructured(hparams, weight=torch.tensor(hparams.class_weight).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.lr)
    # scheduler = MultiStepLR(optimizer, milestones=[100], gamma=0.2, verbose=True)

    def transform_probs2labels(output: List[torch.Tensor]):
        return torch.argmax(output[0][0], dim=1), output[1]
    
    def transform_probs2probs(output: List[torch.Tensor]):
        return output[0][0][:,1], output[1]
    
    # metrics should be a dictionary
    metrics = {"precision": Precision(output_transform=transform_probs2labels),
               "recall": Recall(output_transform=transform_probs2labels),
            "specificity": Specificity(output_transform=transform_probs2labels),
            "npv": NPV(output_transform=transform_probs2labels),

            "bacc": BalancedAccuracy(output_transform=transform_probs2labels), 
               "f1": F1(output_transform=transform_probs2labels),
               "aucroc": AUROC(output_transform=transform_probs2probs),
               "auprc": AUPRC(output_transform=transform_probs2probs),
               
               "brier": BrierScore(output_transform=transform_probs2probs),
               "ece": ECE(output_transform=transform_probs2probs),
               "nll": NLL(output_transform=lambda x: (x[0][0], x[1]))}
    
    # add loss into metrics
    criterion_detached = MultitaskLoss4OneStructured_detached(hparams, weight=torch.tensor(hparams.class_weight).to(device))
    metrics["loss"] = Loss(criterion_detached, output_transform=lambda x: (x[0][:-1], x[1]))
    

    # ********************** trainer and evaluator
    print("="*15, " Building trainer and evaluator ", "="*15)
    trainer = create_supervised_trainer(model, 
                                        optimizer, 
                                        criterion, 
                                        device=device, 
                                        output_transform=lambda x, y, y_pred, loss: criterion_detached(y_pred, y), 
                                        model_transform=lambda x: x[:-1]
                                        )


    evaluator = create_supervised_evaluator(model, 
                                            metrics=metrics, 
                                            device=device)
    eos = EpochOutputStore()
    eos.attach(evaluator, 'output')

    # ********************** add event handlers
    # Log training loss in each iteration
    def log_training_results(engine):
        batch_idx = engine.state.iteration % engine.state.epoch_length if engine.state.iteration  % engine.state.epoch_length != 0 else engine.state.epoch_length
        print(f"Epoch: {engine.state.epoch} | Batch: {batch_idx}/{engine.state.epoch_length} - Train loss: {engine.state.output:.4f}")
        logger.log_metrics({"train_loss": engine.state.output})

    # Log validation and test results in each epoch
    def log_validation_results(engine):
        print("Validating...")
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("val completed")
        logger.log_metrics(metrics, prefix="val")

    
    def log_test_results(engine):
        print("Testing...")
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        print("test completed")
        logger.log_metrics(metrics, prefix="test")

        # scheduler.step()


    trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: logger.set_epoch(engine.state.epoch))
    trainer.add_event_handler(Events.ITERATION_STARTED, lambda engine: logger.new_step())
    trainer.add_event_handler(Events.ITERATION_COMPLETED, log_training_results)
    
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, log_test_results)
    # trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: scheduler.step())
    

    # ********************** training start
    print("="*15, " Start training ", "="*15)
    trainer.run(train_loader, max_epochs=hparams.max_epochs)


if __name__ == "__main__":

    hparams = add_hparams2parser(hparams_dict)

    logger = create_logger(hparams)
    
    train(hparams, logger)