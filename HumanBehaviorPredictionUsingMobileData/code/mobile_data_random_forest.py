#Header files
import pandas as pd
import pylab as pl
import randomforests as rf
import sklearn.metrics as scikit_metrics

#Read the csv file containing the data
mobile_gyro_data_frame = pd.read_csv("/Users/sriram/Downloads/UCIHARDataset/samsungdata.csv")
mobile_data_data_frame_backup = mobile_gyro_data_frame

#Clean the data
#Drop duplicate columns
# mobile_gyro_data_frame.drop_duplicates(['name'], inplace=True)
# mobile_gyro_data_frame.name= mobile_gyro_data_frame.name.str.replace('[()]','')
# mobile_gyro_data_frame.name= mobile_gyro_data_frame.name.str.replace('[0-9]','')
# mobile_gyro_data_frame.drop_duplicates(['name'], inplace=True)
# mobile_gyro_data_frame= mobile_gyro_data_frame.name[mobile_gyro_data_frame.name.str.contains('-X|-Y|-Z|min|max|mad|sma|iqr|entropy|energy|band|Coeff') == False]
# mobile_gyro_data_frame = mobile_gyro_data_frame.str.replace('Body','')
# mobile_gyro_data_frame = mobile_gyro_data_frame.str.replace('Mag','')
# mobile_gyro_data_frame = mobile_gyro_data_frame.str.replace('mean', 'Mean')
# mobile_gyro_data_frame = mobile_gyro_data_frame.str.replace('std', 'SD')
# if len(mobile_gyro_data_frame[mobile_gyro_data_frame.str.contains('mean|std')]) == 0: print "All mean and std have been converted!"
# mobile_gyro_data_frame = mobile_gyro_data_frame.str.replace('-','')
# mobile_gyro_data_frame = mobile_gyro_data_frame.str.replace(',','')

cols=list(mobile_gyro_data_frame.columns)
newcols = ["x%d" %(k) for k in range(0,len(cols))]
newcols[-2:] = cols[-2:]
mobile_gyro_data_frame.columns = newcols
mobile_gyro_data_frame = mobile_gyro_data_frame[mobile_gyro_data_frame.columns[1:-2]]
mobile_data_data_frame_backup['subject'].value_counts()
print len(mobile_gyro_data_frame)
#37

# mobile_gyro_train_data = pd.read_csv('/Users/sriram/Downloads/UCIHARDataset/samtrain.csv')
# mobile_gyro_validation_data = pd.read_csv('/Users/sriram/Downloads/UCIHARDataset/samval.csv')
# mobile_gyro_test_data = pd.read_csv('/Users/sriram/Downloads/UCIHARDataset/samtest.csv')
mobile_gyro_train_data = mobile_data_data_frame_backup[mobile_data_data_frame_backup['subject'] <= 6]
mobile_gyro_validation_data_2 = mobile_data_data_frame_backup[mobile_data_data_frame_backup['subject'] < 27]
mobile_gyro_validation_data = samval2[mobile_data_data_frame_backup['subject'] >=21]
mobile_gyro_test_data = mobile_data_data_frame_backup[mobile_data_data_frame_backup['subject'] >= 27]

# print mobile_gyro_train_data['subject'].unique()
# print mobile_gyro_validation_data['subject'].unique()
# print mobile_gyro_test_data['subject'].unique()

# Remove the spare column ('unnamed') created by pandas from train, vals and test
mobile_gyro_train_data.drop(mobile_gyro_train_data.columns[0], axis=1, inplace=True)
mobile_gyro_validation_data.drop(mobile_gyro_validation_data.columns[0], axis=1, inplace=True)
mobile_gyro_test_data.drop(mobile_gyro_test_data.columns[0], axis=1, inplace=True)

# print mobile_gyro_train_data.shape
# print mobile_gyro_validation_data.shape
# print mobile_gyro_test_data.shape

mobile_gyro_train_data = rf.remap_col(mobile_gyro_train_data,'activity')
mobile_gyro_validation_data = rf.remap_col(mobile_gyro_validation_data,'activity')
mobile_gyro_test_data = rf.remap_col(mobile_gyro_test_data,'activity')
rfc = sk.RandomForestClassifier(n_estimators=500, oob_score=True)
train_data = mobile_gyro_train_data[mobile_gyro_train_data.columns[0:-2]]
train_truth = mobile_gyro_train_data['activity']
model = rfc.fit(train_data, train_truth)
print rfc.oob_score_

fi = enumerate(rfc.feature_importances_)
cols = mobile_gyro_train_data.columns
[(value,cols[i]) for (i,value) in fi if value > 0.035]
val_data = mobile_gyro_validation_data[mobile_gyro_validation_data.columns[0:-2]]
val_truth = mobile_gyro_validation_data['activity']
val_pred = rfc.predict(val_data)
test_data = mobile_gyro_test_data[mobile_gyro_test_data.columns[0:-2]]
test_truth = mobile_gyro_test_data['activity']
test_pred = rfc.predict(test_data)
print("mean accuracy score for validation set = %f" %(rfc.score(val_data, val_truth)))
#mean accuracy score for validation set = 0.835436
print("mean accuracy score for test set = %f" %(rfc.score(test_data, test_truth)))
#mean accuracy score for test set = 0.895623

test_cm = scikit_metrics.confusion_matrix(test_truth,test_pred)
pl.matshow(test_cm)
pl.title('Confusion matrix for test data')
pl.colorbar()
pl.show()
print("Accuracy = %f" %(scikit_metrics.accuracy_score(test_truth,test_pred)))
#Accuracy = 0.895623
print("Precision = %f" %(scikit_metrics.precision_score(test_truth,test_pred)))
'''
/Users/sriram/anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
  sample_weight=sample_weight)
Precision = 0.897822
'''
print("Recall = %f" %(scikit_metrics.recall_score(test_truth,test_pred)))
'''
/Users/sriram/anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
  sample_weight=sample_weight)
Recall = 0.895623
'''
print("F1 score = %f" %(scikit_metrics.f1_score(test_truth,test_pred)))
'''
/Users/sriram/anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:756: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
  sample_weight=sample_weight)
F1 score = 0.896006
'''
