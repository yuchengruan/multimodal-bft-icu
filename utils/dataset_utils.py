import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder



def ordinal_transform(x, n_outputs = 4):
    assert x >= 0 and x <= n_outputs, f"{x} is out of the range."
    ordinal_label = np.concatenate((np.ones((x, )), np.zeros((n_outputs - x, ))), axis= 0)
    return ordinal_label.astype("long")


class CustomLabelEncoder:
    def __init__(self):
        self.classes_ = None

    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        return np.array([self._map_value(val) for val in y])

    def _map_value(self, val):
        if val in self.classes_:
            return np.where(self.classes_ == val)[0][0]
        else:
            return len(self.classes_)

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

class CustomOrdinalEncoder(OrdinalEncoder):
    def __init__(self, categories = "auto", handle_unknown = "use_encoded_value", unknown_value = np.nan, **kwargs):
        super().__init__(categories=categories, handle_unknown=handle_unknown, unknown_value=unknown_value, **kwargs)
    
    def fit(self, X, y=None):
        super().fit(X, y)
        return self

    def transform(self, X):
        # all unseen data are encoded as np.nan
        X_trans = super().transform(X)
        cardinalities = [len(categories) for categories in self.categories_]

        # replace np.nan with the cardinality in each column
        for i, col in enumerate(X_trans.T):
            col[np.isnan(col)] = cardinalities[i]

        self.cardinalities = cardinalities

        return X_trans

    def get_cardinalities(self):
        return self.cardinalities  # have to +1 to account for the unseen value

if __name__ == "__main__":
    # Example usage
    encoder = CustomOrdinalEncoder()
    y = np.array([['111', 'dog'], ['111', 'cat'], ["222", 'dog'], ['333', 'cat']])
    transformed_y = encoder.fit_transform(y)
    print("Transformed labels:", transformed_y, sep="\n")
    print("Categories:", encoder.categories_, sep="\n")

    y_test = np.array([['111', 'ham'], ['555', 'cat'], ["222", 'dog'], ['333', 'cat'], ['444', 'dog']])
    transformed_y_test = encoder.transform(y_test)
    print("Transformed test labels:", transformed_y_test, sep="\n")
    print(f"{encoder.get_feature_names_out()}")
    print("Cardinalities:", encoder.get_cardinalities())