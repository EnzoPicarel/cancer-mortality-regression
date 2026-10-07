"""Fit a fixed candidate on every labeled row for competition inference."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import sklearn
from model import CANDIDATES,build_model,read_labeled


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--model',choices=CANDIDATES,default='blend')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path,default=Path('artifacts/submission_model.joblib'))
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(f'Output already exists: {args.output}')
    X,y=read_labeled(args.data)
    estimator=build_model(args.model,args.seed).fit(X,y)
    metadata={'purpose':'all-label competition refit; no held-out score',
              'candidate':args.model,'rows':len(y),'seed':args.seed,
              'scikit_learn':sklearn.__version__,
              'data_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump({'schema_version':1,'model':estimator,'metadata':metadata},args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'Fitted {args.model} on all {len(y)} labels; wrote {args.output}')


if __name__=='__main__':main()
