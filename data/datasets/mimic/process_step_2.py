import os
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

def param_reset(model):
    if hasattr(model, 'reset_parameters'):
        model.reset_parameters()
    else:
        if hasattr(model, 'children'):
            for child in model.children():
                param_reset(child)


def process_one_note(data_save_path, pre_trained_model, reset_params = False):

    file_path = os.path.join(data_save_path, "text.csv")
    dataset = pd.read_csv(file_path)

    tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)
    cell_encoder = AutoModel.from_pretrained(pre_trained_model)
    if reset_params is True:
        param_reset(cell_encoder)
    for param in cell_encoder.parameters():
        param.requires_grad = False

    device = torch.device(4)
    cell_encoder.to(device)

    cell_embeddings_ls = []
    for i in tqdm(range(len(dataset))):
        inputs_per_sample = dataset.iloc[i].astype("str")
        input_tokenized = tokenizer(inputs_per_sample["notes"], return_tensors="pt", max_length = 512, padding="max_length", truncation=True)

        for key in input_tokenized.keys():
            input_tokenized[key] = input_tokenized[key].to(device)

        with torch.no_grad():
            patient_emb = cell_encoder(**input_tokenized).pooler_output

        cell_embeddings_ls.append(patient_emb.cpu().numpy())

    cell_embeddings = np.stack(cell_embeddings_ls, axis = 0)


    np.save(os.path.join(data_save_path, "onenote_" + pre_trained_model.split("/")[-1] + ".npy"), cell_embeddings)
    print("text.csv saved!")

if __name__ == "__main__":
    
    data_save_path = "./data/datasets/mimic_v3/processed"


    pre_trained_model = "emilyalsentzer/Bio_ClinicalBERT"
    # pre_trained_model = "dmis-lab/biobert-v1.1"
    # pre_trained_model = "google-bert/bert-base-uncased"
    process_one_note(data_save_path, pre_trained_model)
    
