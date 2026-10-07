import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from demo import synthetic_counties
from model import CountyFeatures,ID,TARGET,build_model,read_labeled
from predict import predict
from train import split_indices


class PipelineTests(unittest.TestCase):
    def test_no_target_id_county_or_death_derived_features(self):
        data=synthetic_counties(40)
        builder=CountyFeatures('no-deaths').fit(data)
        result=builder.transform(data)
        for column in [TARGET,ID,'Geography','avgDeathsPerYear','log1p_avgDeathsPerYear',
                       'deaths_per_case','deaths_per_100k_population']:
            self.assertNotIn(column,result)
        changed=data.copy();changed[TARGET]*=100;changed[ID]+=1000
        pd.testing.assert_frame_equal(builder.transform(changed),result)
        pd.testing.assert_frame_equal(builder.transform(data.drop(columns='avgDeathsPerYear')),result)
        self.assertIn('cases_per_100k_population',result)

    def test_training_preprocessing_and_unseen_categories(self):
        frame=synthetic_counties(40)
        X=frame.drop(columns=TARGET);y=frame[TARGET]
        model=build_model('linear').fit(X.iloc[:20],y.iloc[:20])
        feature_frame=model.named_steps['features'].transform(X.iloc[:20])
        numeric=feature_frame.select_dtypes(include=np.number)
        prep=model.named_steps['estimator'].named_steps['preprocess']
        imputer=prep.named_transformers_['numeric'].named_steps['impute']
        np.testing.assert_allclose(imputer.statistics_,numeric.median().to_numpy())
        other=X.iloc[20:].copy();other['medIncome']=1e9;other['binnedInc']='unseen'
        before=imputer.statistics_.copy()
        self.assertTrue(np.isfinite(model.predict(other)).all())
        np.testing.assert_array_equal(imputer.statistics_,before)

    def test_disjoint_reproducible_splits(self):
        a=split_indices(2437,42);b=split_indices(2437,42)
        self.assertEqual([len(x) for x in a],[1705,366,366])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        self.assertEqual(len(set(a[0]) & set(a[1])),0)
        self.assertEqual(len(set(a[0]) & set(a[2])),0)
        self.assertEqual(len(set(a[1]) & set(a[2])),0)
        self.assertEqual(set(np.concatenate(a)),set(range(2437)))

    def test_age_correction_does_not_mutate_input(self):
        frame=synthetic_counties(40)
        before=frame.copy(deep=True)
        result=CountyFeatures().fit_transform(frame)
        self.assertAlmostEqual(result.loc[1,'MedianAge'],frame.loc[1,'MedianAge']/12)
        self.assertTrue(frame.equals(before))

    def test_checkpoint_roundtrip_ids_and_schema(self):
        data=synthetic_counties(40)
        model=clone(build_model('ridge')).fit(data.drop(columns=TARGET),data[TARGET])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);check=root/'model.joblib';incoming=root/'test.csv'
            joblib.dump({'schema_version':1,'model':model},check)
            raw=data.drop(columns=TARGET).iloc[[8,2,15]]
            raw.to_csv(incoming,index=False)
            result=predict(incoming,check,root/'submission.csv')
            self.assertEqual(list(result.columns),[ID,TARGET])
            self.assertEqual(result[ID].tolist(),raw[ID].tolist())
            np.testing.assert_allclose(result[TARGET],model.predict(raw))
            with self.assertRaises(FileExistsError):predict(incoming,check,root/'submission.csv')
            data.to_csv(incoming,index=False)
            with self.assertRaises(ValueError):predict(incoming,check,root/'other.csv')

    def test_input_contract_errors(self):
        data=synthetic_counties(40)
        feature=CountyFeatures().fit(data)
        with self.assertRaises(ValueError):feature.transform(data.drop(columns='MedianAge'))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'train.csv'
            data.drop(columns=TARGET).to_csv(path,index=False)
            with self.assertRaises(ValueError):read_labeled(path)
            data.loc[1,ID]=data.loc[0,ID]
            data.to_csv(path,index=False)
            with self.assertRaises(ValueError):read_labeled(path)


if __name__=='__main__':unittest.main()
