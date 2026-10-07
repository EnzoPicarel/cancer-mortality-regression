"""Offline synthetic execution demo; its scores are not real-data benchmarks."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from train import evaluate
from predict import predict
from model import TARGET


def synthetic_counties(n=240,seed=18):
    rng=np.random.default_rng(seed)
    population=rng.integers(5000,200000,size=n)
    age=rng.uniform(25,65,size=n)
    poverty=rng.uniform(4,30,size=n)
    rate=75+1.1*age+1.6*poverty+rng.normal(0,8,n)
    frame=pd.DataFrame({'id':np.arange(n),'avgAnnCount':population*.004,
        'avgDeathsPerYear':population*rate/100_000+rng.normal(0,3,n),
        'popEst2015':population,'MedianAge':age,'povertyPercent':poverty,
        'medIncome':rng.uniform(25000,90000,n),'studyPerCap':rng.uniform(0,100,n),
        'Geography':[f'Synthetic County {i}, State {i%4}' for i in range(n)],
        'binnedInc':['low' if i%2 else 'high' for i in range(n)],TARGET:rate})
    frame.loc[[1,5,9],'MedianAge']*=12
    frame.loc[[4,8,12],'medIncome']=np.nan
    return frame


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('artifacts/demo'))
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh demo output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    frame=synthetic_counties()
    train_data=args.output/'synthetic_train.csv'
    frame.to_csv(train_data,index=False)
    test_data=args.output/'synthetic_predict.csv'
    frame.drop(columns=TARGET).iloc[:20].to_csv(test_data,index=False)
    evaluate(train_data,args.output/'evaluation',folds=3,repeats=1,synthetic=True)
    predict(test_data,args.output/'evaluation/model.joblib',args.output/'predictions.csv')
    print('Synthetic demo complete. These metrics do not describe cancer-model performance.')


if __name__=='__main__':main()
