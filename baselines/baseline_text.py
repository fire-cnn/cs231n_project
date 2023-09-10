import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
import xgboost as xgb

def fruit_loops(X_train, 
                y_train,
                save_path,
                cv=10,
                scores=['roc_auc', 'accuracy', 'precision', 'recall']):
    """ Run ML pipeline for a set of classifiers
    """

    # Get cagetory types
    numeric_features = X_train.select_dtypes(include=['float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns
    
    # Define transformers for data type
    numeric_transformer = Pipeline(
            steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(
            steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Create ColumnTransformer to change train and test data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Add pipeline object to add modeling and pre-processing
    pipe = Pipeline(
    [('preprocessor', preprocessor),
     ('clf', LogisticRegression(max_iter=500))
    ])

    # Models to run in 10-kFold CV
    param_grid = [
        {'clf': (RandomForestClassifier(),),
        'clf__n_estimators': [10, 100, 200, 500],
        'clf__max_features': ['sqrt', 'log2'],
        'clf__max_depth' : [1,5,10,20,50,100],
        'clf__criterion' :['gini'],
        'clf__min_samples_split': [2,5,10]
        },
       {
       'clf': (LogisticRegression(max_iter=500),),
       'clf__C': (0.001,0.01,0.1,1,10,100),
       'clf__penalty': ['l1','l2']
       },
        {
        'clf':(xgb.XGBClassifier(objective="binary:logistic"),),
        'clf__gamma': [0, 0.25, 0.5],
        'clf__max_depth': [1,5,10,20,50,100],
        'clf__n_estimators': [200, 500]
        }
    ]
    # Run CV
    CV = GridSearchCV(pipe,
                      param_grid,
                      n_jobs= -1,
                      verbose=2,
                      refit=False,
                      scoring=scores,
                      cv=10)
    CV.fit(X_train, y_train)

    # Save results and pickle best models
    df_cross_val = pd.DataFrame(CV.cv_results_)

    if not os.path.exists(os.path.join(save_path, 'models')):
        os.makedirs(os.path.join(save_path, 'models'))

    for metric in scores:
        print(f'Saving best model in CV grid: {metric}')
        df_cross_val_sort = df_cross_val.sort_values(f'rank_test_{metric}')
        model = df_cross_val_sort.iloc[0].param_clf

        path_model = os.path.join(save_path, 
                'models',
                f'best_model_{metric}.pkl'
                )

        with open(path_model, 'wb') as f:
            pickle.dump(model,f) 

    
    # Save CV results
    df_cross_val.to_csv(f'cv.csv', index=False)

    return CV


def main(label_column, columns, path_to_dataset):
    
    df = pd.read_csv(path_to_dataset)
    X = df[columns]
    Y = df[label_column]

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=42)

    cv = fruit_loops(X_train, y_train, cv=10, scores=["roc_auc", "accuracy", "f1"],
                     save_path='./baseline_trees')

    return X_test, y_test, cv


if __name__ == "__main__":
    columns = ['firename',
               'fire_start',
               'FIPS',
               'aspect',
               'elev',
               'fuelmodel',
               'slope_r25',
               'riskToStru',
               'air_temperature',
               'burning_index_g',
               'dead_fuel_moisture_1000hr',
               'dead_fuel_moisture_100hr',
               'mean_vapor_pressure_deficit',
               'potential_evapotranspiration',
               'precipitation_amount',
               'relative_humidity',
               'specific_humidity',
               'surface_downwelling_shortwave_flux_in_air',
               'wind_from_direction',
               'wind_speed',
               'age',
               'band_red_rgb_nanmax',
               'band_red_rgb_nanmin',
               'band_red_rgb_std',
               'band_red_rgb_nanmedian',
               'band_red_rgb_nanmean',
               'band_green_rgb_nanmax',
               'band_green_rgb_nanmin',
               'band_green_rgb_std',
               'band_green_rgb_nanmedian',
               'band_green_rgb_nanmean',
               'band_blue_rgb_nanmax',
               'band_blue_rgb_nanmin',
               'band_blue_rgb_std',
               'band_blue_rgb_nanmedian',
               'band_blue_rgb_nanmean'
               ]

    main(label_column='destroyed', columns=columns, 
         path_to_dataset="/mnt/sherlock/cnn_fire_fuel/data/text/all_features_cleaned.csv")


