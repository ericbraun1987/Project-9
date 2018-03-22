
# coding: utf-8

# In[176]:


import sys
import cPickle as pickle
import numpy as np
import pandas as pd
from copy import copy
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import scale
from sklearn.ensemble import RandomForestClassifier
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier


# In[177]:

features_list =  ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi'] 


# In[178]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# dict to dataframe
df = pd.DataFrame.from_dict(data_dict, orient='index')
df.replace('NaN', np.nan, inplace = True)
df.drop('email_address', axis=1, inplace=True)


# In[179]:

df[financial_features] = df[financial_features].fillna(df[financial_features].median())
df[email_features] = df[email_features].fillna(df[email_features].median())


# In[180]:

df.drop('TOTAL', inplace = True)
df.drop('THE TRAVEL AGENCY IN THE PARK', inplace=True)
df.drop('LOCKHART EUGENE E', inplace=True)


# In[181]:

df['rate_from_poi'] = df['from_poi_to_this_person'] / df['to_messages']
df['rate_to_poi'] = df['from_this_person_to_poi'] / df['from_messages']
df['total_assets']= df['salary']+df['bonus']+df['total_stock_value']+df['exercised_stock_options']+df['long_term_incentive']+df['restricted_stock']


# In[182]:

my_dataset = df.to_dict('index')

feature_list = [u'poi',u'salary', u'to_messages', u'deferral_payments', u'total_payments', u'exercised_stock_options', u'bonus',
                u'restricted_stock', u'shared_receipt_with_poi', u'restricted_stock_deferred', u'total_stock_value', u'expenses',
                u'loan_advances', u'from_messages', u'other', u'from_this_person_to_poi', u'director_fees', u'deferred_income', 
                u'long_term_incentive', u'from_poi_to_this_person', u'total_assets', u'rate_from_poi', u'rate_to_poi']


# In[183]:


from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
import tester


# In[184]:

from sklearn import preprocessing
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


# In[185]:

# dt_clf number of features
n_features = np.arange(1, len(feature_list))

dt_pipe = Pipeline([
    ('select_features', SelectKBest()),
    ('classify', DecisionTreeClassifier())
])

param_grid = [
    {
        'select_features__k': n_features
    }
]


dt_clf= GridSearchCV(dt_pipe, param_grid=param_grid, scoring='f1', cv = 10)
dt_clf.fit(features, labels);

dt_clf.best_params_



# In[186]:


#ab_clf number of features

n_features = np.arange(1, len(feature_list))


ab_pipe = Pipeline([
    ('select_features', SelectKBest()),
    ('classify', AdaBoostClassifier())
])

param_grid = [
    {
        'select_features__k': n_features
    }
]

ab_clf= GridSearchCV(ab_pipe, param_grid=param_grid, scoring='f1', cv = 10)
ab_clf.fit(features, labels);

ab_clf.best_params_


# In[187]:

# Tests Gaussian NB
#gnb_clf = GaussianNB()
#tester.dump_classifier_and_data(gnb_clf, my_dataset, feature_list)
#tester.main();

# Tests Decision Tree Classifier
#dt_clf = DecisionTreeClassifier()
#tester.dump_classifier_and_data(dt_clf, my_dataset, feature_list)
#tester.main();

# Tests KMeans classifier 
#km_clf = KMeans(n_clusters=2)
#tester.dump_classifier_and_data(km_clf, my_dataset, feature_list)
#tester.main();

# Tests AdaBoost classifier
#ab_clf =  AdaBoostClassifier(algorithm='SAMME')
#tester.dump_classifier_and_data(ab_clf, my_dataset, feature_list)
#tester.main();


# In[188]:

#dt_clf = DecisionTreeClassifier()
#dt_clf.fit(features, labels)

#dt_feature_importances = (dt_clf.feature_importances_)
#dt_features = zip(dt_feature_importances, features_list[1:])
#dt_features = sorted(dt_features, key= lambda x:x[0], reverse=True)

#print('Decision Tree Features:\n')
#for i in range(6):
 #   print('{} : {:.4f}'.format(dt_features[i][1], dt_features[i][0]))



# In[189]:

feature_list_2 = [u'poi',u'exercised_stock_options', u'deferred_income', u'shared_receipt_with_poi', 
                  u'expenses', u'restricted_stock', u'from_poi_to_this_person']
print len(feature_list_2)
df2 = df[['poi','exercised_stock_options', 'deferred_income', 'shared_receipt_with_poi', 'expenses', 'restricted_stock',
          'from_poi_to_this_person'
]].copy()

my_dataset_2 = df2.to_dict('index')
data = featureFormat(my_dataset_2, feature_list_2, sort_keys = True)
labels, features = targetFeatureSplit(data)
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


# In[190]:

#n_features = np.arange(1, len(feature_list))
#
#dt_pipe = Pipeline([
 #   ('select_features', SelectKBest()),
  #  ('classify', DecisionTreeClassifier())
#])

#param_grid = [
 #   {
  #      'select_features__k': n_features
   # }
#]


#dt_clf= GridSearchCV(dt_pipe, param_grid=param_grid, scoring='f1', cv = 10)
#dt_clf.fit(features, labels);

#dt_clf.best_params_


# In[ ]:




# In[191]:

dt_pipe = Pipeline([
    ('select_features', SelectKBest(k=6)),
    ('classify', DecisionTreeClassifier()),
])


param_grid = dict(classify__criterion = ['gini', 'entropy'] , 
                  classify__min_samples_split = [2, 4, 6, 8, 10, 20],
                  classify__max_depth = [None, 5, 10, 15, 20],
                  classify__max_features = [None, 'sqrt', 'log2', 'auto'])


dt_clf = GridSearchCV(dt_pipe, param_grid = param_grid, scoring='f1', cv=10)
dt_clf.fit(features, labels)
dt_clf.best_params_


# In[192]:

#ab_pipe = Pipeline([('select_features', SelectKBest(k=6)),
 #                    ('classify', AdaBoostClassifier())
  #                  ])

#param_grid = dict(classify__base_estimator=[DecisionTreeClassifier(), RandomForestClassifier(), GaussianNB()],
 #                 classify__n_estimators = [30, 50, 70, 120],
  #                classify__learning_rate = [0.5, 1, 1.5, 2, 4])

#ab_clf = GridSearchCV(ab_pipe, param_grid=param_grid, scoring='f1', cv=10)
#ab_clf.fit(features, labels)
#ab_clf.best_params_


# In[193]:

dt_clf = Pipeline([
    ('select_features', SelectKBest(k=6)),
    ('classify', DecisionTreeClassifier(criterion='gini', max_depth= 6, max_features= None, min_samples_split=2))
])

tester.dump_classifier_and_data(dt_clf, my_dataset_2, feature_list_2)
tester.main()


# In[ ]:



