import pandas as pd
import os
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import simplejson
import pandas as pd
import numpy as np
import torch
import numpy as np
import os
from transformers import BertTokenizerFast,BertModel
import pandas as pd
import simplejson
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 666
np.random.seed(RANDOM_STATE)  


# def z_score_standardization(series):
#     return (series - series.mean()) / series.std()


def process_text_data(df):
    df_data = df.copy()

    # replace [empty] cell in text columns with ""
    # add column name to the cell content if it is not empty
    for col in text_cols:
        df_data[col] = df_data[col].map(lambda x: col.strip() + ": " + x + "\n" if x != "[empty]" else "")

    # concatenate all the columns in text_cols
    df_data["notes"] = df_data["Nursing"]  + df_data["Nursing/other"] + df_data["Physician "] + df_data["Radiology"]
    # strip the leading and trailing whitespaces
    df_data["notes"] = df_data["notes"].str.strip()

    # replace the empty cell in "notes" with "There is no data available."
    df_data["notes"] = df_data["notes"].replace("", "Notes: There is no data available.")

    return df_data, df_data["notes"]

def process_text_data_ds(df):
    df_data = df.copy()

    # replace [empty] cell in text columns with ""
    for col in text_cols:
        df_data[col] = df_data[col].map(lambda x: col.strip() + ": " + x if x != "[empty]" else col.strip() + ": " + "There is no data available.")

    return df_data, df_data[text_cols]


def process_structured_data_cont(df):
    df_data = df.copy()

    return df_data[cont_cols].round(2).astype("float32")



def process_structured_data_cat(df):
    df_data = df.copy()

    return df_data[cat_cols].astype("category")


def process_structured_data(df, col_ls):
    df_data = df.copy()

    # return categorical and continuous features in col_ls
    cont_cols_extracted = sorted(list(set(col_ls) & set(cont_cols)))
    cat_cols_extracted = sorted(list(set(col_ls) & set(cat_cols)))


    return df_data[cont_cols_extracted].round(2).astype("float32"), df_data[cat_cols_extracted].astype("category")


def generate_cv_splits(data_save_path, df_labels, label_col):

    data_path = os.path.join(data_save_path, label_col)
    os.makedirs(data_path, exist_ok=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for i, (train_val_idx, test_idx) in enumerate(skf.split(np.zeros(len(df_labels)), df_labels[label_col])):

        split_save_path = os.path.join(data_path, f"{i}")
        os.makedirs(split_save_path, exist_ok=True)

        train_val_label = df_labels.iloc[train_val_idx][label_col].values
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.25, random_state=RANDOM_STATE, stratify=train_val_label)

        # save the splits
        np.save(os.path.join(split_save_path, "train_idx.npy"), train_idx)
        np.save(os.path.join(split_save_path, "val_idx.npy"), val_idx)
        np.save(os.path.join(split_save_path, "test_idx.npy"), test_idx)

        print(f"cv splits for {label_col} saved in {split_save_path}")


cont_cols = ['admission_age','heart_rate', 'aptt',
            'urea_nitrogen',  'eosinophil', 'lymphocytes', 'neutrophils',
            'rdw', 'bicarbonate', 'chloride', 'creatinine', 'hemoglobin',
            'mean_cell_vol', 'platelet_count', 'potassium', 'sodium',
            'prothrombin_time', 'prothrombin_time_inr','weight','wbc', "plr", "nlr"]

cat_cols = [# categorical features
            'gender', 'ethnicity_grouped', "admission_type",
            
            # binary features (Yes/No)
            "sedatives", "statin", "diuretic", "antibiotics", "hypertension",
            "diabetes", "alcohol_abuse", "cva", "chf", "ihd", "ventilation", "vasopressor"]

text_cols = [# text
            'Nursing', 'Nursing/other', 'Physician ', 'Radiology']

label_cols = ['icu_death', "los_rank", "los_icu"]

usecols = cont_cols + cat_cols + text_cols + label_cols


demographics = ['admission_age', 'gender', 'weight', 'ethnicity_grouped', 'admission_type']

vital = ['heart_rate', 'aptt', 'urea_nitrogen', 'eosinophil',
       'lymphocytes', 'neutrophils', 'rdw', 'bicarbonate', 'chloride',
       'creatinine', 'hemoglobin', 'mean_cell_vol', 'platelet_count',
       'potassium', 'sodium', 'prothrombin_time', 'prothrombin_time_inr'
       , 'wbc', 'plr', 'nlr']

treatment = ['sedatives', 'statin', 'diuretic', 'antibiotics', 'ventilation', 'vasopressor']
cormobidities = ['hypertension', 'diabetes', 'alcohol_abuse', 'cva', 'chf', 'ihd']



if __name__ == "__main__":

    assert len(cont_cols) + len(cat_cols) == len(demographics) + len(vital) + len(treatment) + len(cormobidities)

    data_path = "./data/datasets/mimic_v3"
    data_save_path = os.path.join(data_path, "processed")
    os.makedirs(data_save_path, exist_ok=True)

    data = pd.read_csv(os.path.join(data_path, "data_merged.csv"))[usecols]
    data[text_cols] = data[text_cols].fillna("[empty]")

    # -----------------------------for mutimodal--------------------------------------------
    
    # text features
    df_data, text_data = process_text_data(data)
    # continuous features
    df_cont = df_data[cont_cols].round(2).astype("float32")
    # categorical features
    df_cat = df_data[cat_cols].astype("category")

    # save text features
    text_data.to_csv(os.path.join(data_save_path, "text.csv"), index=False)
    print("text data saved!")

    # save continuous features
    df_cont.to_csv(os.path.join(data_save_path, "cont.csv"), index=False)
    print("cont data saved!")

    # split and save categorical features
    df_cat.to_csv(os.path.join(data_save_path, "cat.csv"), index=False)
    print("cat data saved!")


    # PROCESS THE DATA FOR DATA SOURCES
    # -----------------------------for multimodal_ds--------------------------------------------
    # text features
    df_data, text_data = process_text_data_ds(data)
    # demographic features
    df_cont_demographics, df_cat_demographics = process_structured_data(df_data, demographics)
    # vital features
    df_cont_vital, df_cat_vital = process_structured_data(df_data, vital)
    # treatment features
    df_cont_treatment, df_cat_treatment = process_structured_data(df_data, treatment)
    # cormobidities features
    df_cont_comorbidities, df_cat_comorbidities = process_structured_data(df_data, cormobidities)


    # -----------------------------for mutimodal_ds--------------------------------------------
    # save text features
    text_data.to_csv(os.path.join(data_save_path, "text_ds.csv"), index=False)
    print("text data saved for ds!")

    # save demographic features
    df_cont_demographics.to_csv(os.path.join(data_save_path, "cont_demographics.csv"), index=False)
    df_cat_demographics.to_csv(os.path.join(data_save_path, "cat_demographics.csv"), index=False)
    print("demographics data saved for ds!")

    # save vital features
    df_cont_vital.to_csv(os.path.join(data_save_path, "cont_vital.csv"), index=False)
    df_cat_vital.to_csv(os.path.join(data_save_path, "cat_vital.csv"), index=False)
    print("vital data saved for ds!")

    # save treatment features
    df_cont_treatment.to_csv(os.path.join(data_save_path, "cont_treatment.csv"), index=False)
    df_cat_treatment.to_csv(os.path.join(data_save_path, "cat_treatment.csv"), index=False)
    print("treatment data saved for ds!")

    # save comorbidities features
    df_cont_comorbidities.to_csv(os.path.join(data_save_path, "cont_comorbidities.csv"), index=False)
    df_cat_comorbidities.to_csv(os.path.join(data_save_path, "cat_comorbidities.csv"), index=False)
    print("comorbidities data saved for ds!")


    # --------------------------------for labels--------------------------------------------
    # binarize the los_icu with thresold 7
    data["long_icu"] = (data["los_icu"] > 7).astype(int)
    data["icu_death"] = data["icu_death"].astype(int)

    # save the label
    df_labels = data[["icu_death", "long_icu"]]
    df_labels.to_csv(os.path.join(data_save_path, "label.csv"), index=False)
    print("label data saved!")

    # --------------------------------for generate splits--------------------------------------------
    # create 5-fold cross-validation splits for icu_death
    generate_cv_splits(data_save_path, df_labels, label_col="icu_death")

    # create 5-fold cross-validation splits for long_icu
    generate_cv_splits(data_save_path, df_labels, label_col="long_icu")


