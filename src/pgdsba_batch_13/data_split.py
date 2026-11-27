import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np

    return np, pd


@app.cell
def _(pd):
    data = pd.read_csv('dataset/housing.csv')
    return (data,)


@app.cell
def _():
    from sklearn.model_selection import train_test_split

    return (train_test_split,)


@app.cell
def _(data, train_test_split):
    train, test = train_test_split(data, test_size=0.25, random_state=42)
    return test, train


@app.cell
def _(test, train):
    print(train.shape)
    print(test.shape)


@app.cell
def _(data):
    data['median_income'].describe()


@app.cell
def _(data):
    data.shape


@app.cell
def _(data, np, pd):
    data['income_cat'] = pd.cut(data['median_income'], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])


@app.cell
def _(data):
    data.shape


@app.cell
def _(data):
    len(data)


@app.cell
def _(data):
    data['income_cat'].value_counts()  / len(data) * 100 


@app.cell
def _(data, train_test_split):
    strat_train, strat_test = train_test_split(data, test_size=0.25, random_state=42, stratify=data['income_cat'])
    return strat_test, strat_train


@app.cell
def _(strat_train):
    strat_train['income_cat'].value_counts()  / len(strat_train) * 100 


@app.cell
def _(strat_test):
    strat_test['income_cat'].value_counts()  / len(strat_test) * 100 


@app.cell
def _(strat_test, strat_train):
    strat_test.to_csv('dataset/housing_test.csv')
    strat_train.to_csv('dataset/housing_train.csv')


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
