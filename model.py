"""Portable feature engineering and frozen regression candidates."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

TARGET = 'TARGET_deathRate'
ID = 'id'
CANDIDATES = ('mean', 'linear', 'ridge', 'histgb', 'blend', 'blend-no-deaths')


class CountyFeatures(BaseEstimator, TransformerMixin):
    """Row-local transformations; column schema learned only from training input."""
    def __init__(self, variant='enhanced'):
        self.variant = variant

    def fit(self, X, y=None):
        if self.variant not in ('baseline', 'enhanced', 'no-deaths'):
            raise ValueError('Unknown feature variant.')
        if not isinstance(X, pd.DataFrame):
            raise TypeError('Expected a pandas DataFrame.')
        excluded = {TARGET, ID}
        if self.variant == 'baseline':excluded.add('Geography')
        if self.variant == 'no-deaths':excluded.add('avgDeathsPerYear')
        self.columns_ = [c for c in X.columns if c not in excluded]
        if not self.columns_:
            raise ValueError('No predictors provided.')
        return self

    def transform(self, X):
        check_is_fitted(self, 'columns_')
        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f'Missing input columns: {sorted(missing)}')
        data = X.loc[:, self.columns_].copy()
        if 'MedianAge' in data:
            data['MedianAge'] = data['MedianAge'].where(data['MedianAge'] <= 100,
                                                       data['MedianAge']/12)
        if self.variant != 'baseline' and 'Geography' in data:
            data['state'] = data['Geography'].str.rsplit(',', n=1).str[-1].str.strip()
        data = data.drop(columns='Geography', errors='ignore')
        if self.variant == 'no-deaths':
            data = data.drop(columns='avgDeathsPerYear', errors='ignore')
        if self.variant != 'baseline':
            for column in ('avgAnnCount', 'avgDeathsPerYear', 'popEst2015', 'studyPerCap'):
                if column in data:
                    data[f'log1p_{column}'] = np.log1p(data[column].clip(lower=0))
            if {'avgAnnCount', 'popEst2015'} <= set(data):
                data['cases_per_100k_population'] = (
                    data['avgAnnCount'].clip(lower=0)/data['popEst2015'].clip(lower=1)*100_000)
            if 'avgDeathsPerYear' in data:
                deaths = data['avgDeathsPerYear'].clip(lower=0)
                if 'avgAnnCount' in data:
                    data['deaths_per_case'] = deaths/data['avgAnnCount'].clip(lower=1)
                if 'popEst2015' in data:
                    data['deaths_per_100k_population'] = deaths/data['popEst2015'].clip(lower=1)*100_000
        return data


def preprocessing(scale=True):
    numeric_steps = [('impute', SimpleImputer(strategy='median', add_indicator=True))]
    if scale:
        numeric_steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(numeric_steps), make_column_selector(dtype_include=np.number)),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), make_column_selector(dtype_exclude=np.number)),
    ], remainder='drop')


def build_model(name, seed=42):
    if name not in CANDIDATES:
        raise ValueError(f'Unknown model {name}; choose from {CANDIDATES}.')
    variant = 'baseline' if name in ('mean', 'linear') else 'no-deaths' if name == 'blend-no-deaths' else 'enhanced'
    ridge = Pipeline([('preprocess', preprocessing()), ('regressor', Ridge(alpha=.75))])
    tree = Pipeline([('preprocess', preprocessing(scale=False)),
                     ('regressor', HistGradientBoostingRegressor(
                         max_leaf_nodes=15, learning_rate=.04, max_iter=350,
                         min_samples_leaf=12, l2_regularization=3, random_state=seed))])
    if name == 'mean':
        estimator = Pipeline([('preprocess', preprocessing()), ('regressor', DummyRegressor())])
    elif name == 'linear':
        estimator = Pipeline([('preprocess', preprocessing()), ('regressor', LinearRegression())])
    elif name == 'ridge':
        estimator = ridge
    elif name == 'histgb':
        estimator = tree
    else:
        estimator = VotingRegressor([('ridge', ridge), ('histgb', tree)], weights=[.35,.65])
    return Pipeline([('features', CountyFeatures(variant)), ('estimator', estimator)])


def read_labeled(path):
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f'Expected label column {TARGET}.')
    y = pd.to_numeric(frame[TARGET], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError('Labels must be finite.')
    if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError('Expected unique, nonmissing county row IDs.')
    return frame.drop(columns=TARGET), y
