import pandas as pd
import numpy as np
import tensorflow as tf
import keras as k
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def age_dict(data: pd.DataFrame):
    male, female = data[data.Sex == 'male']['Age'].mean(), data[data.Sex == 'female']['Age'].mean()
    return {'1': int(male), '0': int(female)}


def age_encod(value: float):
    if 0 <= value <= 15:
        return 0
    elif 15 < value <= 45:
        return 1
    else:
        return 2


class Data:

    def __init__(self, file):
        self.data = pd.read_csv(file)
        self.age_dict = age_dict(self.data)

    def clear(self):
        le = LabelEncoder()
        mm = MinMaxScaler()
        self.data.Fare = mm.fit_transform(self.data.Fare.values.reshape(-1,1))
        self.data.Embarked = le.fit_transform(self.data.Embarked)
        self.data.Sex = self.data.Sex.apply(lambda val: 1 if val == 'male' else 0)
        self.data.Age = self.data.apply(
            lambda row: age_encod(self.age_dict[str(row.Sex)]) if str(row.Age) == 'nan' else age_encod(row.Age), axis=1).astype(int)
        self.data["Embarked"].fillna(self.data.Embarked.value_counts().max(), inplace=True)
        self.data.Ticket = self.data["Ticket"].apply(lambda x: int(x.isdigit()))
        del self.data["Cabin"], self.data["PassengerId"], self.data["Name"]

    def to_model(self):
        return XY(self.data)

class XY:

    def __init__(self, data: pd.DataFrame):
        inf = data
        self.Y = inf.Survived
        del inf["Survived"]
        self.X = inf

    def get(self):
        return self.X, self.Y


if __name__ == '__main__':
    Data = Data('train.csv')
    Data.clear()
    dat = Data.to_model()
    X, Y = dat.get()
    print(X)
