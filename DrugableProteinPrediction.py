import pandas as pd
import os
from sklearn.metrics import confusion_matrix, f1_score, precision_score, accuracy_score
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import subprocess
import sys


def run_feature_extraction(feature_type, dataset):
    if feature_type == 'PAAC':
        command = f"python iFeature-master\iFeature-master\codes\PAAC.py {dataset}.txt 5 encoding.tsv"
    else:
        command = f"python iFeature-master\iFeature-master\iFeature.py --file {dataset}.txt --type {feature_type}"
    
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return_code = process.returncode

def extract_features(feature_types, datasets):
  for feature_type in feature_types:
    outdir = '{}'.format(feature_type)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    for dataset in datasets:
      run_feature_extraction(feature_type, f"{dataset}")

      df = pd.read_table('encoding.tsv')

      df = df.drop(['#'], axis = 1)

      df.to_csv('{}/{}.csv'.format(outdir, dataset), index = False)

    # print('{} extraction is Done! \n'.format(feature_type))

def select_k_top_features(train_X, train_y, k):
  mi_scores = mutual_info_classif(train_X, train_y)

  mi_series = pd.Series(mi_scores, index = train_X.columns)

  mi_series = mi_series.sort_values(ascending=False)

  top_k_features = mi_series.index.to_list()[:k]
  
  return top_k_features


def scale_dataset(train_X, test_X, test_pos_X, test_neg_X):
  scaler = StandardScaler()
  num_cols = train_X.select_dtypes('number').columns
  train_X[num_cols] = scaler.fit_transform(train_X[num_cols])
  test_X[num_cols] = scaler.transform(test_X[num_cols])
  test_pos_X[num_cols] = scaler.transform(test_pos_X[num_cols])
  test_neg_X[num_cols] = scaler.transform(test_neg_X[num_cols])

  return train_X, test_X, test_pos_X, test_neg_X


def train_test_split(train, test):
  train_X = train.drop(['pos'], axis = 1)
  train_y = train['pos']

  test_X = test.drop(['pos'], axis = 1)
  test_y = test['pos']

  return train_X, test_X, train_y, test_y

def dataset_split(datasetXandY):
  dataset_X = datasetXandY.drop(['pos'], axis = 1)
  dataset_y = datasetXandY['pos']
  return dataset_X, dataset_y


def create_train_test_dfs(feature_type, outdir, data_list):
  train_pos = pd.read_csv('{}/{}/{}.csv'.format(outdir, feature_type, data_list[0]))
  train_pos = train_pos.assign(pos=1)

  train_neg = pd.read_csv('{}/{}/{}.csv'.format(outdir, feature_type, data_list[1]))
  train_neg = train_neg.assign(pos=0)

  train_df = pd.concat([train_pos, train_neg], axis=0)
  train_df = train_df.sample(frac=1)

  test_pos = pd.read_csv('{}/{}/{}.csv'.format(outdir, feature_type, data_list[2]))
  test_pos = test_pos.assign(pos=1)

  test_neg = pd.read_csv('{}/{}/{}.csv'.format(outdir, feature_type, data_list[3]))
  test_neg = test_neg.assign(pos=0)

  test_df = pd.concat([test_pos, test_neg], axis=0)
  test_df = test_df.sample(frac=1, random_state = 69)


  #train_test split
  train_X, train_y = dataset_split(train_df)
  test_X, test_y = dataset_split(test_df)
  test_pos_X, test_pos_Y = dataset_split(test_pos)
  test_neg_X, test_neg_Y = dataset_split(test_neg)

  train_X, test_X, test_pos_X, test_neg_X = scale_dataset(train_X, test_X, test_pos_X, test_neg_X)

  return train_X, test_X, test_pos_X, test_neg_X, train_y, test_y, test_pos_Y, test_neg_Y


def evaluate(classifier, test_X, test_y):
  preds = classifier.predict(test_X)

  #confusion matrix
  cf_matrix = confusion_matrix(test_y, preds)
  sns.heatmap(cf_matrix, annot=True)

  accuracy = accuracy_score(test_y, preds)
  precision = precision_score(test_y, preds)
  f1 = f1_score(test_y, preds)

  # Calculate confusion matrix
  tn, fp, fn, tp = cf_matrix.ravel()

  # Calculate sensitivity and specificity
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)

  # Print results
  print('Accuracy:', accuracy)
  print('Sensitivity:', sensitivity)
  print('Specificity:', specificity)
  print('Precision:', precision)
  print('F1_score:', f1)


TR_pos_SPIDER = sys.argv[1]
TR_neg_SPIDER = sys.argv[2]
TS_pos_SPIDER = sys.argv[3]
TS_neg_SPIDER = sys.argv[4]

feature_types = ['AAC', 'GDPC', 'DPC', 'PAAC']
datasets = ['TR_pos_SPIDER', 'TR_neg_SPIDER', 'TS_pos_SPIDER', 'TS_neg_SPIDER']
outdir_parent = './'
extract_features(feature_types, datasets)

# Classfier using AAC feature type

# print("\nClassfier using AAC feature type")
feature_type = 'AAC'
AAC_train_X, AAC_test_X, AAC_test_pos_X, AAC_test_neg_X, AAC_train_y, AAC_test_y, AAC_test_pos_Y, AAC_test_neg_Y = create_train_test_dfs(feature_type, outdir_parent, datasets)
xgb_cl_AAC = xgb.XGBClassifier()
xgb_cl_AAC.fit(AAC_train_X, AAC_train_y)
# evaluate(xgb_cl_AAC, AAC_test_X, AAC_test_y)

# Classfier using GDPC feature type

# print("\nClassfier using GDPC feature type")
feature_type = 'GDPC'
GDPC_train_X, GDPC_test_X, GDPC_test_pos_X, GDPC_test_neg_X, GDPC_train_y, GDPC_test_y, GDPC_test_pos_Y, GDPC_test_neg_Y = create_train_test_dfs(feature_type, outdir_parent, datasets)
xgb_cl_GDPC = xgb.XGBClassifier()
xgb_cl_GDPC.fit(GDPC_train_X, GDPC_train_y)
# evaluate(xgb_cl_GDPC, GDPC_test_X, GDPC_test_y)

# Classfier using DPC feature type

# print("\nClassfier using DPC feature type")
feature_type = 'DPC'
DPC_train_X, DPC_test_X, DPC_test_pos_X, DPC_test_neg_X, DPC_train_y, DPC_test_y, DPC_test_pos_Y, DPC_test_neg_Y = create_train_test_dfs(feature_type, outdir_parent, datasets)
xgb_cl_DPC = xgb.XGBClassifier()
xgb_cl_DPC.fit(DPC_train_X, DPC_train_y)
# evaluate(xgb_cl_DPC, DPC_test_X, DPC_test_y)

# Classfier using PAAC feature type

# print("\nClassfier using PAAC feature type")
feature_type = 'PAAC'
PAAC_train_X, PAAC_test_X, PAAC_test_pos_X, PAAC_test_neg_X, PAAC_train_y, PAAC_test_y, PAAC_test_pos_Y, PAAC_test_neg_Y = create_train_test_dfs(feature_type, outdir_parent, datasets)
xgb_cl_PAAC = xgb.XGBClassifier()
xgb_cl_PAAC.fit(PAAC_train_X, PAAC_train_y)
# evaluate(xgb_cl_PAAC, PAAC_test_X, PAAC_test_y)




# Ensemble part - Voting classifier

print("\nVoting classifier")

def getEnasmblePreds(model_list, test_df_index):
   proba_df = pd.DataFrame()
   for i, model_data in enumerate(model_list):
    model = model_data[1]
    test_df = model_data[test_df_index]
    predictions = model.predict_proba(test_df)
    model_proba_df = pd.DataFrame(predictions, columns=[f'Model_{i}_Class_0', f'Model_{i}_Class_1'])
    proba_df = pd.concat([proba_df, model_proba_df], axis=1)

    proba_df['Result'] = proba_df.apply(lambda row: 1 if row.filter(like='Class_1').sum() > row.filter(like='Class_0').sum() else 0, axis=1)

    ensemble_preds = proba_df['Result'].values
    return ensemble_preds

model_list = [(AAC_test_X, xgb_cl_AAC, AAC_test_pos_X, AAC_test_neg_X),(DPC_test_X, xgb_cl_DPC, DPC_test_pos_X, DPC_test_neg_X)]


ensemble_preds = getEnasmblePreds(model_list, 0)

cf_matrix = confusion_matrix(AAC_test_y, ensemble_preds)
sns.heatmap(cf_matrix, annot=True)

accuracy = accuracy_score(AAC_test_y, ensemble_preds)
precision = precision_score(AAC_test_y, ensemble_preds)
f1 = f1_score(AAC_test_y, ensemble_preds)

# Calculate confusion matrix
tn, fp, fn, tp = cf_matrix.ravel()

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print results
print('Accuracy:', accuracy)
print('Sensitivity:', sensitivity)
print('Specificity:', specificity)
print('Precision:', precision)
print('F1_score:', f1)


positive_ensemble_preds = getEnasmblePreds(model_list, 2)
# Save predictions to a file
with open('predictions_pos.txt', 'w') as file:
    for prediction in positive_ensemble_preds:
        file.write(str(prediction) + '\n')

negetive_ensemble_preds = getEnasmblePreds(model_list, 3)
# Save predictions to a file
with open('predictions_neg.txt', 'w') as file:
    for prediction in negetive_ensemble_preds:
        file.write(str(prediction) + '\n')



# python Bio_Ass.py TR_pos_SPIDER TR_neg_SPIDER TS_pos_SPIDER TS_neg_SPIDER