import os
from sklearn.impute import SimpleImputer
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.utils.class_weight import compute_class_weight



class MIMIC_Structured_Notes(Dataset):
    def __init__(self, data_path, split, hparams = None) -> None:
        super().__init__()

        self.hparams = hparams
        self.data_path = data_path
        self.data_outcome_path = os.path.join(data_path, hparams.outcome)
        self.cv_split_path = os.path.join(self.data_outcome_path, str(hparams.cv_split))

        data_split_path = os.path.join(self.cv_split_path, split + ".pickle")
        # check if we have data splits saved
        if not os.path.exists(data_split_path):
            print(f"Data split does not exist. Setting up data splits.")
            self.setup()

        with open(data_split_path, 'rb') as f:
            data = pickle.load(f)

        self.cont = data["cont"].to_numpy()
        self.cat = data["cat"].to_numpy()
        self.notes = data["notes"][hparams.pretrained_model]
        self.label = data["label"].to_numpy()

        if split == "train":
            self.hparams.structured_d_in_ls = [self.cont.shape[1], self.cat.shape[1]]
            print(f"* Structured input dimensions: {self.hparams.structured_d_in_ls}")
            self.hparams.class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(self.label), y=self.label).tolist()

    def setup(self):

        # load split idx
        train_split = np.load(os.path.join(self.cv_split_path, "train_idx.npy"))
        val_split = np.load(os.path.join(self.cv_split_path, "val_idx.npy"))
        test_split = np.load(os.path.join(self.cv_split_path, "test_idx.npy"))

        # load cont and cat data for the splits
        data_cont_path = os.path.join(self.data_path, "cont.csv")
        data_cat_path = os.path.join(self.data_path, "cat.csv")
        df_cont = pd.read_csv(data_cont_path)
        df_cat = pd.read_csv(data_cat_path)

        train_cont = df_cont.iloc[train_split]
        val_cont = df_cont.iloc[val_split]
        test_cont = df_cont.iloc[test_split]
        
        train_cat = df_cat.iloc[train_split]
        val_cat = df_cat.iloc[val_split]
        test_cat = df_cat.iloc[test_split]

        train_notes_dict = {}
        val_notes_dict = {}
        test_notes_dict = {}
        # load notes for the splits
        for pretrained_model in ["emilyalsentzer/Bio_ClinicalBERT", "dmis-lab/biobert-v1.1", "google-bert/bert-base-uncased"]:
            notes = np.load(os.path.join(self.data_path, "onenote_" + pretrained_model.split("/")[-1] + ".npy")).squeeze()

            train_notes_dict[pretrained_model] = notes[train_split]
            val_notes_dict[pretrained_model] = notes[val_split]
            test_notes_dict[pretrained_model] = notes[test_split]


        # load labels for the splits
        data_label_path = os.path.join(self.data_path, "label.csv")

        train_label = pd.read_csv(data_label_path)[self.hparams.outcome][train_split]
        val_label = pd.read_csv(data_label_path)[self.hparams.outcome][val_split]
        test_label = pd.read_csv(data_label_path)[self.hparams.outcome][test_split]

        # -------------------------------- Preprocessing --------------------------------
        # 1. Missing data imputation
        cont_imputer = SimpleImputer(strategy='median')
        train_cont = pd.DataFrame(cont_imputer.fit_transform(train_cont), columns=train_cont.columns)
        val_cont = pd.DataFrame(cont_imputer.transform(val_cont), columns=val_cont.columns)
        test_cont = pd.DataFrame(cont_imputer.transform(test_cont), columns=test_cont.columns)

        cat_inputer = SimpleImputer(strategy='most_frequent')
        train_cat = pd.DataFrame(cat_inputer.fit_transform(train_cat), columns=train_cat.columns)
        val_cat = pd.DataFrame(cat_inputer.transform(val_cat), columns=val_cat.columns)
        test_cat = pd.DataFrame(cat_inputer.transform(test_cat), columns=test_cat.columns)

        # 2. Standardization of continuous features
        ss = StandardScaler()
        train_cont = pd.DataFrame(ss.fit_transform(train_cont), columns=train_cont.columns)
        val_cont = pd.DataFrame(ss.transform(val_cont), columns=val_cont.columns)
        test_cont = pd.DataFrame(ss.transform(test_cont), columns=test_cont.columns)

        # 3. Encoding of categorical features (one-hot encoding)
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        train_cat = pd.DataFrame(ohe.fit_transform(train_cat), columns=ohe.get_feature_names_out())
        val_cat = pd.DataFrame(ohe.transform(val_cat), columns=ohe.get_feature_names_out())
        test_cat = pd.DataFrame(ohe.transform(test_cat), columns=ohe.get_feature_names_out())

        # save data as pickle based on train, val, test splits
        with open(os.path.join(self.cv_split_path, "train.pickle"), 'wb') as f:
            pickle.dump({"cont": train_cont, "cat": train_cat, "notes": train_notes_dict, "label": train_label}, f)
        with open(os.path.join(self.cv_split_path, "val.pickle"), 'wb') as f:
            pickle.dump({"cont": val_cont, "cat": val_cat, "notes": val_notes_dict, "label": val_label}, f)
        with open(os.path.join(self.cv_split_path, "test.pickle"), 'wb') as f:
            pickle.dump({"cont": test_cont, "cat": test_cat, "notes": test_notes_dict, "label": test_label}, f)


    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):

        cont_data = self.cont[index].astype("float32")
        cat_data = self.cat[index].astype("float32")
        notes_data = self.notes[index]
        label_data = self.label[index]


        return (cont_data, cat_data, notes_data), label_data

