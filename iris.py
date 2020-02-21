import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

import pdb

# -----------------------------------
# LOAD DATASET 
dataset=pd.read_csv("iris.csv")
print(dataset.head())

# ---------------------------------------
# DATASET INFOS 
print(dataset.shape)  # (150, 5)
print(dataset.describe())

# -------------------------------------------------
# TYPES OF CATEGORIES
categories=dataset.ix[:,4].values
print(np.unique(categories)) #----------------------------- ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

# -------------------------------------------------------
# VISUALIZATON
# UNI VARIATE PLOTS
# box and whisker plots
dataset.plot(kind='box',subplots=True,layout=(2,2))
plt.show()

# histogram
dataset.hist()
plt.show()

# MULTIVARIATE PLOT
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

# -------------------------------------------------
# SPLITTING DATASET
from sklearn.model_selection import train_test_split
x=dataset.ix[:,0:4].values
y=dataset.ix[:,4].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2,random_state=0)
train_y=train_y.reshape(-1,1)
test_y=test_y.reshape(-1,1)

print(train_y.shape,test_y.shape)
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.compose import ColumnTransformer

le=LabelEncoder()
train_y=le.fit_transform(train_y).reshape(-1,1)

le2=LabelEncoder()
test_y=le2.fit_transform(test_y).reshape(-1,1)
# # ---------------------------------------------------------------------------------

# DIMENSIONALITY REDUCTION USING LDA FOR VISUALIZATION
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
train_x=lda.fit_transform(train_x,train_y)
test_x=lda.transform(test_x)
# -----------------------------------------------------------------------------------
# pdb.set_trace()

# COMPARING MODEL
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC

models=[] #--- models will contain tuples whose fist element will be name and second element will be model

def getin(name,model):
    ''' this function will pass a tuple to models. Tuple consist of name and model.'''
    return models.append((name,model))

getin('logistic regression',LogisticRegression(random_state=0))
getin('knn',KNeighborsClassifier())
getin('tree',DecisionTreeClassifier(random_state=0))
getin('naive_bayes',GaussianNB())
getin('svc',SVC(kernel='rbf',degree=3))

model_score=[]
for name , model in models:
    kfold = StratifiedKFold(n_splits=10,random_state=1)
    score= cross_val_score(model,train_x,train_y,scoring='accuracy',cv=kfold)
    model_score.append((name,score.mean()))

from pprint import pprint
pprint(model_score)
        # [('logistic regression', 0.9333566433566434),
        #  ('knn', 0.9327156177156178),
        #  ('tree', 0.9320745920745921),
        #  ('naive_bayes', 0.9494988344988344),
        #  ('svc', 0.9564335664335666)]

# Since svc has maximum score so we choose svc
classifier=SVC()

# to find suitable values of argument use Gridsearch
from sklearn.model_selection import GridSearchCV
parameters=[
    {'C':[1,10,100,1000],'kernel':['linear']},
    {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
]

grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10,n_jobs=-1)
grid_search.fit(train_x,train_y)
print(grid_search.best_params_) # {'C': 1, 'gamma': 0.4, 'kernel': 'rbf'}

classifier=SVC(C=1,kernel='rbf',)
classifier.fit(train_x,train_y)
pred_y=classifier.predict(test_x)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(pred_y,test_y)
print(cm)

# ------------------------------------------------------------------

# VISUALIZATION
x_set,y_set=test_x,test_y
x1,x2=np.meshgrid(
    np.arange(x_set[:,0].min()-1,x_set[:,0].max()+1,0.01),
    np.arange(x_set[:,1].min()-1,x_set[:,1].max()+1,0.01)
)

plt.contourf(
    x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
    alpha=0.75,cmap=ListedColormap(('red','green','blue'))
)

flowers=['Iris-setosa','Iris-versicolor','Iris-virginica']
print(x_set.shape,y_set.shape)
for i,j in enumerate(np.unique(y_set)):
    y=[]
    for z in y_set:
        if z==j:
            y.append(True)
        else:
            y.append(False)
    plt.scatter(x_set[y,0],x_set[y,1],
    c=ListedColormap(('red','green','blue'))(i),label=flowers[j])
plt.legend()
plt.show()