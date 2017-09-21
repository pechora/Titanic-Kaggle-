import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers

model = Sequential()
model.add(Dense(96, activation='relu', input_dim=11))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='tanh'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked', 'Survived']]
test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

def disp(X):
    print(X.describe())

def prep(X, train = True):
    xnp = X.values
    lnx = len(xnp)
    x_in = np.zeros([lnx, 11], dtype = 'float64')

    x_in[:, 0:3] = pd.get_dummies(X['Pclass'])[[1, 2, 3]].values
    x_in[:, 3:5] = pd.get_dummies(X['Sex'])[['male', 'female']].values
    x_in[:, 5] = (X['Age'].fillna(X['Age'].median())).values/50
    x_in[:, 6] = X['SibSp'].values/10
    x_in[:, 7] = X['Parch'].values/10
    x_in[:, 8:11] = pd.get_dummies(X['Embarked'])[['C', 'Q', 'S']].values

    if(train):
        y_in = pd.get_dummies(X['Survived'])[[0, 1]].values
        return x_in, y_in
    return x_in

def train(X):
    x_in, y_in = prep(X)
    model.fit(x_in, y_in, batch_size=100, epochs=100, shuffle = True, verbose=1, validation_split=0.25)

disp(X)
disp(test)
train(X)
x_in = prep(test, train = False)
y_pred = model.predict(x_in)

lnx = len(test)
ans = np.zeros([lnx, 2], dtype = 'int64')
ids = test_df['PassengerId']

for i in range(0, lnx):
    ans[i][0] = ids[i]
    ans[i][1] = y_pred[i].argmax()

pd.DataFrame(ans).to_csv('ans.csv', header = ['PassengerId', 'Survived'],index = False)
