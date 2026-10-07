"""Export finite predictions with the original row-ID order."""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from model import ID,TARGET


def predict(data_path,checkpoint,output):
    data=pd.read_csv(data_path)
    if ID not in data or data[ID].isna().any() or data[ID].duplicated().any():
        raise ValueError('Expected unique, nonmissing row IDs.')
    if TARGET in data:
        raise ValueError('Prediction input must not contain labels.')
    artifact=joblib.load(checkpoint)
    if artifact.get('schema_version')!=1:
        raise ValueError('Unsupported checkpoint format.')
    predictions=artifact['model'].predict(data)
    if len(predictions)!=len(data) or not np.isfinite(predictions).all():
        raise ValueError('Invalid predictions.')
    result=pd.DataFrame({ID:data[ID].to_numpy(),TARGET:predictions})
    output=Path(output)
    if output.exists():raise FileExistsError(f'Output already exists: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=predict(args.data,args.checkpoint,args.output)
    print(f'Wrote {len(result)} predictions to {args.output}')


if __name__=='__main__':main()
