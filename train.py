"""Compare fixed candidates; select on validation; report a split-isolated replay."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_validate, train_test_split

from model import CANDIDATES, build_model, read_labeled


def split_indices(n, seed):
    train, rest = train_test_split(np.arange(n), test_size=.30, random_state=seed)
    validation, test = train_test_split(rest, test_size=.50, random_state=seed)
    return train, validation, test


def metrics(y, prediction):
    return {'r2': float(r2_score(y,prediction)),
            'rmse': float(np.sqrt(mean_squared_error(y,prediction))),
            'mae': float(mean_absolute_error(y,prediction))}


def evaluate(data_path, output, seed=42, folds=5, repeats=2, names=CANDIDATES,
             synthetic=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new or empty output directory to preserve existing results.')
    output.mkdir(parents=True,exist_ok=True)
    X,y=read_labeled(data_path)
    if folds<2 or repeats<1 or len(y)<20:
        raise ValueError('Need >=20 observations, >=2 folds, and >=1 repeat.')
    tr,va,te=split_indices(len(y),seed)
    if folds>len(tr):raise ValueError('More folds than training observations.')
    cv=RepeatedKFold(n_splits=folds,n_repeats=repeats,random_state=seed)
    splits=list(cv.split(tr))
    rows, fold_rows=[],[]
    for name in names:
        estimator=build_model(name,seed)
        scores=cross_validate(estimator,X.iloc[tr],y[tr],cv=splits,
                              scoring={'r2':'r2','mse':'neg_mean_squared_error'},
                              n_jobs=1,error_score='raise')
        for i,(r2,mse) in enumerate(zip(scores['test_r2'],-scores['test_mse'])):
            fold_rows.append({'model':name,'repeat':i//folds,'fold':i%folds,
                              'r2':float(r2),'mse':float(mse)})
        fitted=clone(estimator).fit(X.iloc[tr],y[tr])
        row={'model':name,'cv_r2_mean':float(scores['test_r2'].mean()),
             'cv_r2_std_fold':float(scores['test_r2'].std(ddof=1))}
        for prefix,idx in [('train',tr),('validation',va)]:
            row.update({f'{prefix}_{k}':v for k,v in metrics(y[idx],fitted.predict(X.iloc[idx])).items()})
        rows.append(row)
        print(f"{name}: validation R2={row['validation_r2']:.4f}",flush=True)
    # Select before any test scoring. Hyperparameters and weights are fixed.
    chosen=max(rows,key=lambda row:row['validation_r2'])['model']
    development=np.concatenate([tr,va])
    selected_model=None
    for row in rows:
        fitted=build_model(row['model'],seed).fit(X.iloc[development],y[development])
        prediction=fitted.predict(X.iloc[te])
        row.update({f'test_{k}':v for k,v in metrics(y[te],prediction).items()})
        row['selected_on_validation']=row['model']==chosen
        if row['model']==chosen:
            selected_model=fitted
    results=pd.DataFrame(rows)
    results.to_csv(output/'evaluation.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(output/'cv_folds.csv',index=False)
    metadata={
        'schema_version':1,'data_sha256':hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'dataset_name':Path(data_path).name,'synthetic':synthetic,'seed':seed,
        'rows':len(y),'split_sizes':{'train':len(tr),'validation':len(va),'test':len(te)},
        'split_indices':{'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist()},
        'cv_folds':folds,'cv_repeats':repeats,'cv_partition':'training split only',
        'selection':'highest validation R2, before test scoring',
        'test_partition':'same rows for all fixed candidates; models refitted on train+validation',
        'selected_model':chosen,'candidates':list(names),
        'python':platform.python_version(),'numpy':np.__version__,
        'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'joblib':joblib.__version__,
        'scope':'Synthetic execution demonstration.' if synthetic else
            'Split-isolated replay on historically reused coursework data; not prospective validation.',
    }
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    joblib.dump({'schema_version':1,'model':selected_model,'metadata':metadata},output/'model.joblib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    x=np.arange(len(rows))
    ax.bar(x-.19,results.validation_r2,.38,label='Validation: fit on 70%',color='#087f8c')
    ax.bar(x+.19,results.test_r2,.38,label='Test: refit on 85%',color='#385a7c')
    ax.set_xticks(x,results.model,rotation=15,ha='right')
    ax.set_ylabel('R²');ax.set_title('Fixed-model replay · synthetic demo' if synthetic else 'County mortality · fixed-model replay')
    ax.axhline(0,color='#a0a0a0',linewidth=.7);ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)
    print(f'Selected on validation: {chosen}; wrote results to {output}',flush=True)
    return results,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path(__file__).with_name('data')/'train.csv')
    parser.add_argument('--output',type=Path,default=Path('artifacts/evaluation'))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--folds',type=int,default=5)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--models',nargs='+',choices=CANDIDATES,default=list(CANDIDATES))
    args=parser.parse_args()
    evaluate(args.data,args.output,args.seed,args.folds,args.repeats,args.models)


if __name__=='__main__':main()
