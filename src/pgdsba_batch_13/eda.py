import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    housing = pd.read_csv('dataset/housing_train.csv')
    return (housing,)


@app.cell
def _(housing):
    housing


@app.cell
def _(housing):
    housing.columns.tolist()


@app.cell
def _():
    x_columns = [
      "longitude",
      "latitude",
      "housing_median_age",
      "total_rooms",
      "total_bedrooms",
      "population",
      "households",
        "median_income",
      "ocean_proximity"
    ]

    y_column = ["median_house_value"]
    return x_columns, y_column


@app.cell
def _(housing, x_columns, y_column):
    X = housing[x_columns]
    y = housing[y_column]
    return X, y


@app.cell
def _(X, y):
    print(X.shape)
    print(y.shape)


@app.cell
def _(X):
    X.dtypes


@app.cell
def _(X):
    X_num = X.select_dtypes(include=['number']).columns.tolist()
    X_cat = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    return X_cat, X_num


@app.cell
def _(X_cat, X_num):
    print(X_num)
    print(X_cat)


@app.cell
def _():
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn import set_config
    set_config(display='diagram')
    return (Pipeline,)


@app.cell
def _():
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import OneHotEncoder

    return OneHotEncoder, SimpleImputer, StandardScaler


@app.cell
def _(OneHotEncoder, Pipeline, SimpleImputer, StandardScaler):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('std_scaler', OneHotEncoder(handle_unknown='ignore'))
    ])
    return cat_pipeline, num_pipeline


@app.cell
def _():
    from sklearn.compose import ColumnTransformer

    return (ColumnTransformer,)


@app.cell
def _(ColumnTransformer, X_cat, X_num, cat_pipeline, num_pipeline):
    preprocessing_pipeline = ColumnTransformer([
        ('num', num_pipeline, X_num),
        ('cat', cat_pipeline, X_cat)
    ])
    return (preprocessing_pipeline,)


@app.cell
def _(Pipeline, preprocessing_pipeline):
    from sklearn.linear_model import LinearRegression

    lin_reg_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', LinearRegression())
    ])
    lin_reg_pipeline
    return (lin_reg_pipeline,)


@app.cell
def _(X, lin_reg_pipeline, y):
    lin_model = lin_reg_pipeline.fit(X, y)
    return (lin_model,)


@app.cell
def _(lin_model):
    print(lin_model.named_steps['model'].coef_)
    print(lin_model.named_steps['model'].intercept_)


@app.cell
def _(Pipeline, preprocessing_pipeline):
    from sklearn.tree import DecisionTreeRegressor

    tree_reg_pipeline = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('model', DecisionTreeRegressor())
    ])
    tree_reg_pipeline
    return (tree_reg_pipeline,)


@app.cell
def _(X, tree_reg_pipeline, y):
    reg_model = tree_reg_pipeline.fit(X, y)
    return (reg_model,)


@app.cell
def _():
    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import r2_score

    def evaluate_model(model, data, labels):
        predictions = model.predict(data)
        r2 = r2_score(labels, predictions)
        rmse = root_mean_squared_error(labels, predictions)
        # rerun as dict
        return {'r2': r2, 'rmse': rmse}

    return (evaluate_model,)


@app.cell
def _(X, evaluate_model, reg_model, y):
    evaluate_model(reg_model, X, y)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
