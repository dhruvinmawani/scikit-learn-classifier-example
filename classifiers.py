from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# Accuracy Score Dependency
from sklearn.metrics import accuracy_score

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

_X = [[184, 84, 44], [198, 92, 48], [183, 83, 44], [166, 47, 36],
      [170, 60, 38], [172, 64, 39], [182, 80, 42], [180, 80, 43]]
_Y = ['male', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

clf_list = [KNeighborsClassifier(3),
            SVC(kernel="rbf", C=0.025, probability=True),
            NuSVC(probability=True),
            LinearSVC(random_state=0, tol=1e-5),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            AdaBoostClassifier(),
            GradientBoostingClassifier(),
            GaussianNB(),
            LinearDiscriminantAnalysis(),
            QuadraticDiscriminantAnalysis()]

# CHALLENGE - ...and train them on our data
for clfx in clf_list:
    clf = clfx.fit(X, Y)
    prediction = clf.predict(_X)
    name = clfx.__class__.__name__
    print("="*30)
    
    print(name)
    print('****Results****')
    acc = accuracy_score(_Y, prediction)
    print("Accuracy: {:.4%}".format(acc))
