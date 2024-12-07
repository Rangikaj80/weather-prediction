import streamlit as st
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.offline import iplot
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest,RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,f1_score


header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

with header:
    st.title('Welcome to my Weather Prediction data science project.')
    st.text('In this project I looking into Weather prediction with features.')
    
with dataset:
    st.header('Weather dataset.')
    st.text('I found this dataset on kaggle website.')

    df = pd.read_csv("weather.csv")
    st.write(df.head())

    st.subheader('Temperature destribution.')
    temperature = pd.DataFrame(df['Temperature'].value_counts()).head(50)
    st.bar_chart(temperature)
    
    
with features:
    st.header('The features I created.')
    
    st.markdown('* **first feature:** I created the feature because of this.... I calculated it using')
    st.markdown('* **second feature:** I createthis feature because of this....I calculated it using')
    
with model_training:
    st.header('Time to train the model.')
    st.text('Here you get to choose the hyperparameters of the model and see how the performance change')
    
    sel_col,disp_col = st.columns(2)
    max_depth = sel_col.slider('What should be the max_depth of the model?',min_value=10,max_value=100,value=20,step=10)
    n_estimators = sel_col.selectbox('How many trees should there be?',options=[100,200,300,'No limit'])
    input_feature = sel_col.text_input('Which feature should be used as the input featur?','')

    regc = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    
    X = df.drop(["Weather Type"],axis=1)
    y = df['Weather Type']
    
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import OneHotEncoder,StandardScaler,LabelEncoder
    from sklearn.feature_selection import SelectPercentile,chi2

    # Define numerical and categorical columns
    numerical_features = X.select_dtypes(exclude='object').columns.tolist()
    categorical_features = X.select_dtypes(include='object').columns.tolist()


    # Encode the target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Define transformas
    numerical_transformer = Pipeline(steps=[
        ('imputer',SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())

    ])
    categorical_transformer = Pipeline(steps=[
        ('onehot',OneHotEncoder(handle_unknown='ignore')),
        ('selector',SelectPercentile(chi2,percentile=50)),
    ])

    # create the column transformer
    preprocessor = ColumnTransformer(
        transformers =[
            ('num',numerical_transformer,numerical_features),
            ('cat',categorical_transformer,categorical_features)
        ]

      )
    np.random.seed(0)
    
    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def evaluate_model(y_true, y_pred,y_proba):
      accuracy = accuracy_score(y_true, y_pred)
      precision = precision_score(y_true, y_pred,average='weighted')
      recall = recall_score(y_true, y_pred,average='weighted')
      f1 = f1_score(y_true,y_pred,average='weighted')
      if y_proba is not None:
        roc = roc_auc_score(pd.get_dummies(y_true),y_proba,multi_class='ovr',average='weighted')
      else:
        roc = float('nan') # Cannot calculate ROC AUC without probabilities
      return accuracy,precision,recall,f1,roc

    # Model and their hyperparameters for GridSearchCV
    models = {
        'lr' : (LogisticRegression(max_iter=100),{'classifier__C':[0.1,1,10]}),
        'knn' : (KNeighborsClassifier(), {'classifier__n_neighbors':[3,5,7]}),
        'svm' : (SVC(probability=True),{'classifier__C':[0.1,1,10],'classifier__kernel': ['linear','rbf']}),
        'rf' : (RandomForestClassifier(),{'classifier__n_estimators':[50, 100, 200]}),
       # 'xgb' : (XGBClassifier(use_label_encoder=False,eval_metric='logloss'),{'classifier__n_estimators':[50, 100,200]})
    }
    model_list = []
    train_results = []
    test_results = []

    # Stratified K-Flod cross validator
    sk = StratifiedKFold(n_splits=5)
    #for name, (model, params) in models.items():
    for name,(model,params) in models.items():
      # Create the full pipeline with preprocessing and the model
      pipeline = Pipeline(steps=[
          ('preprocessor', preprocessor),
          ('classifier', model)
      ])
      # GridSearchCV
      grid_seach = GridSearchCV(pipeline,param_grid=params,cv=sk,scoring='accuracy',n_jobs=-1)
      grid_seach.fit(X_train,y_train)

      # get the best model
      best_model = grid_seach.best_estimator_

      # Make predictions
      y_train_pred = best_model.predict(X_train)
      if hasattr(best_model, "predict_proba"):
        y_train_proba = best_model.predict_proba(X_train)
      else:
        y_train_proba = None
      y_test_pred = best_model.predict(X_test)
      if hasattr(best_model,"predict_proba"):
        y_test_proba = best_model.predict_proba(X_test)
      else:
        y_test_proba = None

      # Evaluate on the train set
      train_accuracy,train_precision,train_recall,train_f1,train_roc_auc = evaluate_model(y_train,y_train_pred,y_train_proba)
      test_accuracy,test_precision,test_recall,test_f1,test_roc_auc = evaluate_model(y_test,y_test_pred,y_test_proba)

      print(name)
      model_list.append(name)

      print("Model performance for training set")
      print(f'- accuracy :{train_accuracy:.4f}')
      print(f'- precesion :{train_precision:.4f}')
      print(f'- recall :{train_recall:.4f}')
      print(f'- f1 :{train_f1:.4f}')
      print(f'- roc :{train_roc_auc:.4f}')

      print("-----------------------------------")

      print("Model performance for test set")
      print(f'- accuracy :{test_accuracy:.4f}')
      print(f'- precesion :{test_precision:.4f}')
      print(f'- recall :{test_recall:.4f}')
      print(f'- f1 :{test_f1:.4f}')
      print(f'- roc :{test_roc_auc:.4f}')

      print("="*35)
      print("\n")

      # Store results
      train_results.append({
          'model':name,
          'accuracy':train_accuracy,
          'precision':train_precision,
          'recall':train_recall,
          'f1':train_f1,
          'roc':train_roc_auc
      })
      test_results.append({
          'model':name,
          'accuracy':test_accuracy,
          'precision':test_precision,
          'recall':test_recall,
          'f1':test_f1,
          'roc':test_roc_auc
      })

    # Optionally convert results to DataFrame for better readability
    train_results_df = pd.DataFrame(train_results)
    test_results_df = pd.DataFrame(test_results)

    print(train_results_df)
    print(test_results_df)

    disp_col.subheader("training results is:")
    disp_col.write(train_results_df)
    disp_col.subheader("test results is:")
    disp_col.write(test_results_df)
    
    
