import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target, test_size, random_state):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() == 2 else None
    )
