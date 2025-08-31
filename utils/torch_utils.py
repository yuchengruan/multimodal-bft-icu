from prettytable import PrettyTable
import torch

def param_reset(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()
        else:
            params = 0
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"* Total Trainable Params: {total_params}")
    return total_params
