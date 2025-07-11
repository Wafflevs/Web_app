############## CLASIFICACION ##############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge,Lasso,LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn import set_config
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier #se le puede considerar como un modelo ensamblador (ensambla modelos)
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from pathlib import Path
from joblib import dump







warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class ML:
    
    def __init__(self, X,y, verbose=True, plot=True, regression=None, classification=None, undersample=None,oversample=None,dump=None ):
        self.X=X
        self.y=y
        self.verbose=verbose
        self.regression=regression
        self.classification=classification
        self.undersample=undersample
        self.oversample=oversample
        self.dump=dump
        self.plot=plot
        
        
    def dumpfolder(self, file, type="model", filename=None):

        output_dir = Path("artifacts") / type
        output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{type}.pkl"

        try:
            dump(file, output_dir / filename)
            print(f"{type} se guardo correctamente en {output_dir / filename}")
        except Exception as e:
            print(f"Error al guardar {type} {e}")
            
            
    def preprocess(self):
        xtrain, xtest, ytrain, ytest = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        num_cols = self.X.select_dtypes(exclude="object").columns
        cat_cols = self.X.select_dtypes(include="object").columns
        
        preprocessor = ColumnTransformer(
            [
                ('num',StandardScaler(), num_cols), ('cat', OneHotEncoder(), cat_cols)
            ]   
        )
        xtrain = preprocessor.fit_transform(xtrain)
        xtest = preprocessor.transform(xtest)
        
        if self.dump:
            self.dumpfolder(
                preprocessor, type='preprocessor', filename="preprocessor.pkl"
            )
        
        if self.classification:
        
            le = LabelEncoder()
            ytrain = le.fit_transform(ytrain)
            ytest = le.transform(ytest)
            
            if self.dump:
                self.dumpfolder(le, type="preprocessor", filename="label_encoder.pkl")
                
        elif self.regression:
            pass
        if self.oversample:
            smote = SMOTE(random_state=42, sampling_strategy="minority")
            xtrain, ytrain = smote.fit_resample(xtrain,ytrain)
            
        elif self.undersample:
            rus = RandomUnderSampler(random_state=42)
            xtrain,ytrain = rus.fit_resample(xtrain, ytrain)

        ###NOTA: smote y rus son para ajustar que sea la misma cantidad de datos en entrenamiento...NO SE TOCA test 
        return xtrain, xtest, ytrain, ytest

    def LogReg(self, xtrain, xtest, ytrain, ytest):
        lr = LogisticRegression()
        
        lr_grid = [
            {
                'penalty': ["l1", "l2", "none"],
                "C":[0.01,0.1,1,10,100],
                "max_iter":[100,1000, 2000]
            }
        ]
        
        grid_search = GridSearchCV(lr, lr_grid, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        lrfinal = LogisticRegression(**grid_best_params)
        lrfinal.fit(xtrain, ytrain)
        
        if self.dump:
            self.dumpfolder(
                lrfinal, type="model", filename="logreg.pkl"
            )
        
        ypred = lrfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)
        if self.verbose:
            
        
            print("\n")
            print("---------------------------------\n")
            print("Regresion logistica")
            print(f"accuracy: {accuracy}")
            
            class_report = classification_report(ytest, ypred)
            print(class_report)
        
        if self.plot:
            
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
        
        

    def SVMLin(self, xtrain, xtest, ytrain, ytest):
        
        svc = LinearSVC()
        
        svc_grid=[
            {
                "C":[0.01,0.1, 1,10,100],
                "class_weight":["balanced", None],
                "fit_intercept":[True, False],
                "penalty":["l1","l2",None],
            }
        ]
        
        grid_search = GridSearchCV(svc, svc_grid, cv = 5, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        svcfinal = LinearSVC(**grid_best_params)
        svcfinal.fit(xtrain, ytrain)
        
        if self.dump:
            self.dumpfolder(
                svcfinal, type="model", filename="svm_lineal.pkl"
            )
        
        ypred = svcfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)
        
        if self.verbose:
            
            print("\n")
            print("---------------------------------\n")
            print("SVMlinear")
            print(f"accuracy: {accuracy}")
            
            class_report = classification_report(ytest, ypred)
            print(class_report)
        
        
        if self.plot:
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
        

        
    def SVM(self, xtrain, xtest, ytrain, ytest):
        
        svc = SVC()
        
        svc_grid=[
            {
                "C":[0.01,0.1, 1,10,100],
                "kernel":["linear", "rbf","sigmoid"],
                "gamma":["scale", "auto"]
                
            }
        ]
        
        grid_search = GridSearchCV(svc, svc_grid, cv = 5, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        svcfinal = SVC(**grid_best_params)
        svcfinal.fit(xtrain, ytrain)
        
        
        
        if self.dump:
            self.dumpfolder(
                svcfinal, type="model", filename="SVM.pkl"
            )
        ypred = svcfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)
        if self.verbose:
            
            print("\n")
            print("---------------------------------\n")
            print("SVM")
            print(f"accuracy: {accuracy}")
            
            class_report = classification_report(ytest, ypred)
            print(class_report)
        
        if self.plot:
            
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
        
        

    def DecisionTree(self, xtrain, xtest, ytrain, ytest):
        
        DTC = DecisionTreeClassifier()
        
        
        DTC_grid=[
            {
                "criterion": ["gini", "entropy"],
                "splitter":["best", "random"],
                "class_weight":["balaced", None],
                
            }
        ]
        
        grid_search = GridSearchCV(DTC, DTC_grid, cv = 5, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        DTCfinal = DecisionTreeClassifier(**grid_best_params)
        DTCfinal.fit(xtrain, ytrain)
        
        
        
        if self.dump:
            self.dumpfolder(
                DTCfinal, type="model", filename="Dtree.pkl"
            )
        ypred = DTCfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)    
        if self.verbose:
        
            print("\n")
            print("---------------------------------\n")
            print("Decision Tree")
            print(f"accuracy: {accuracy}")
            
            class_report = classification_report(ytest, ypred)
            print(class_report)
        
        if self.plot:
            
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
        

        
        # from six import StringIO
        # from IPython.display import Image
        # from sklearn.tree import export_graphviz
        # import pydotplus 
        
        # dot_data = StringIO() 
        # export_graphviz(
        #     DTCfinal, out_file=dot_data, filled=True, rounded=True, special_characters=True
        # )
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        # Image(graph.create_png())
        # graph.write_png("arbol.png")
        
    def RandomForest(self, xtrain, xtest, ytrain, ytest):
        
        RFC = RandomForestClassifier()
        
        
        RFC_grid=[
            {
                "criterion": ["gini", "entropy"],
                "class_weight":["balaced", None],
                "warm_start":[True,False],
                
            }
        ]
        
        grid_search = GridSearchCV(RFC, RFC_grid, cv = 5, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        RFCfinal = RandomForestClassifier(**grid_best_params)
        RFCfinal.fit(xtrain, ytrain)
        
        
        
        if self.dump:
            self.dumpfolder(
                RFCfinal, type="model", filename="Random_Forest.pkl"
            )
        ypred = RFCfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)    
        if self.verbose:
            
            print("\n")
            print("---------------------------------\n")
            print("Random Forest")
            print(f"accuracy: {accuracy}")
            
            class_report = classification_report(ytest, ypred)
            print(class_report)
            
            
        if self.plot:
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
        
        
        

    def Adaboost(self, xtrain, xtest, ytrain, ytest):
        
        AB = AdaBoostClassifier()
        
        
        AB_grid=[
            {
                "learning_rate":[0.1,0.5,1],
                "n_estimators":[10,50,100],
                "algorithm":["SAMME", "SAMME.R"]
                
            }
        ]
        
        grid_search = GridSearchCV(AB, AB_grid, cv = 5, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        ABfinal = AdaBoostClassifier(**grid_best_params)
        ABfinal.fit(xtrain, ytrain)
        

        
        if self.dump:
            self.dumpfolder(
                ABfinal, type="model", filename="Adaboost.pkl"
            )
        ypred = ABfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)
        if self.verbose:
            
            print("\n")
            print("---------------------------------\n")
            print("AdaBoost")
            
            print(f"accuracy: {accuracy}")
            class_report = classification_report(ytest, ypred)
            print(class_report)
        
        if self.plot:
            
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
            
        


    def XGBoost(self, xtrain, xtest, ytrain, ytest):
        XGB = xgb.XGBClassifier()
        XGB_grid=[
            {
                "learning_rate":[0.1, 0.5,1],
                "n_estimators":[10,50,100],
                "subsample":[0.8,1.0],
                # "max_depth":[3,5,7]
                
            }
        ]
        
        grid_search = GridSearchCV(XGB, XGB_grid, cv = 5, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        XGBfinal = xgb.XGBClassifier(**grid_best_params)
        XGBfinal.fit(xtrain, ytrain)
        
        
        
        if self.dump:
            self.dumpfolder(
                XGBfinal,type="model", filename="XGBoost.pkl"
            )
        ypred = XGBfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)
        if self.verbose:
            
            print("\n")
            print("---------------------------------\n")
            print("Xtreme Gradient Boost")
            print(f"accuracy: {accuracy}")
            class_report = classification_report(ytest, ypred)
            print(class_report)
        
        if self.plot:
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
        
        


    def GradientBoost(self, xtrain, xtest, ytrain, ytest):
        
        GB = GradientBoostingClassifier()
        
        
        GB_grid=[
            {
                "learning_rate":[0.1,0.5,1],
                "n_estimators":[10,50,100],
                "subsample":[0.8,1.0]
                # "max_depth":[3,5,7]
            }
        ]
        
        grid_search = GridSearchCV(GB, GB_grid, cv = 5, verbose=1)
        grid_search.fit(xtrain, ytrain)
        grid_best_params = grid_search.best_params_
        
        GBfinal = GradientBoostingClassifier(**grid_best_params)
        GBfinal.fit(xtrain, ytrain)
        
        
        
        if self.dump:
            self.dumpfolder(
                GBfinal, type="model", filename="Gradient_Boost.pkl"
            )
        ypred = GBfinal.predict(xtest)
        accuracy = accuracy_score(ytest,ypred)
        if self.verbose:
            
            print("\n")
            print("---------------------------------\n")
            print("Gradient Boost")
            print(f"accuracy: {accuracy}")
            
            class_report = classification_report(ytest, ypred)
            print(class_report)
        
        if self.plot:
            cm = confusion_matrix(ytest, ypred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues')
            plt.show()
            
    def Run(self):
        xtrain, xtest, ytrain, ytest = self.preprocess()
        
        self.LogReg(xtrain, xtest, ytrain, ytest)
        self.SVMLin(xtrain, xtest, ytrain, ytest)
        self.SVM(xtrain, xtest, ytrain, ytest)
        self.DecisionTree(xtrain, xtest, ytrain, ytest)
        self.RandomForest(xtrain, xtest, ytrain, ytest)
        self.Adaboost(xtrain, xtest, ytrain, ytest)
        self.GradientBoost(xtrain, xtest, ytrain, ytest)
        self.XGBoost(xtrain, xtest, ytrain, ytest)
        
        
        
    
   
# df = pd.read_csv(
#     "datasets/used_car_price_dataset_extended.csv"
# )
# df = pd.read_csv(
#     "C:/Users/casham/Documents/RP/datasets/diabetes_prediction.csv"
# )
# print(df.head())
# X =df.drop(["brand"], axis=1)
# y = df["brand"]
# xtrain, xtest, ytrain, ytest = preprocess(X, y)
# LogReg(xtrain, xtest, ytrain, ytest)
#la base que esta comentada no esta hecha para clasii=ficacion, esta mas para regresion...o algo asi


test_pd = pd.read_csv(
    "datasets/test.csv"
)

train_pd = pd.read_csv(
    "datasets/train.csv"
)

df2 = pd.concat([test_pd, train_pd], axis=0)

# print(df2.head())


df2.drop(["id"],axis=1, inplace=True)
df2.dropna(inplace=True)


X = df2.drop(["price_range"], axis=1)
y = df2["price_range"] #la respuesta correcta...



# xtrain, xtest, ytrain, ytest = preprocess(X, y)
# # SVMLin(xtrain, xtest, ytrain, ytest)
# # LogReg(xtrain, xtest, ytrain, ytest)
# # SVM(xtrain, xtest, ytrain, ytest)
# # DecisionTree(xtrain, xtest, ytrain, ytest)
# # RandomForest(xtrain,xtest, ytrain,ytest)
# # Adaboost(xtrain, xtest, ytrain, ytest)
# # GradientBoost(xtrain, xtest, ytrain, ytest)
# # XGBoost(xtrain, xtest, ytrain, ytest)


MLtrain = ML(X,y, classification=True, dump=True, plot=False, verbose=True)
MLtrain.Run()





