import pandas as pd

df = pd.read_csv("diabetes.csv")

type(df)

print("There are %d rows and %d columns in the data." % (df.shape[0], df.shape[1]))


df.isnull().any()


from sklearn.preprocessing import RobustScaler

y = df['Outcome']
X = df.drop('Outcome', axis=1)
X_scaled = pd.DataFrame(RobustScaler().fit_transform(X), columns=X.columns)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


from sklearn import tree

# initializing the tree model and training it
tree_model = tree.DecisionTreeClassifier()
tree_model = tree_model.fit(X_train, y_train)


from sklearn import metrics

# Predict test set:
y_pred = tree_model.predict(X_test)
# test_predprob = tree_model.predict_proba(X_test)[:,1]

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['1', '0']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


from sklearn.linear_model import LogisticRegression

# initializing the tree model and training it
lr_model = LogisticRegression()
lr_model = lr_model.fit(X_train, y_train)

# Predict test set:
y_pred = lr_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['1', '0']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


from sklearn.naive_bayes import GaussianNB

# initializing the tree model and training it
nb_model = GaussianNB()
nb_model = nb_model.fit(X_train, y_train)

# Predict test set:
y_pred = nb_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['1', '0']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


from sklearn.svm import SVC

# initializing the tree model and training it
sv_model = SVC()
sv_model = sv_model.fit(X_train, y_train)

# Predict test set:
y_pred = sv_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['1', '0']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


from sklearn.ensemble import RandomForestClassifier

# initializing the tree model and training it
rf_model = RandomForestClassifier()
rf_model = rf_model.fit(X_train, y_train)

# Predict test set:
y_pred = rf_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['1', '0']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))


from sklearn.neighbors import KNeighborsClassifier

# initializing the tree model and training it
kn_model = KNeighborsClassifier()
kn_model = kn_model.fit(X_train, y_train)

# Predict test set:
y_pred = kn_model.predict(X_test)

auc_scorer = metrics.roc_auc_score(y_test, y_pred)

print("AUC score for test test is %f" % auc_scorer)
print("Confusion Matrix is : \n  {} ".format(metrics.confusion_matrix(y_test, y_pred)))
target_names = ['1', '0']
print(metrics.classification_report(y_test, y_pred, target_names=target_names))
