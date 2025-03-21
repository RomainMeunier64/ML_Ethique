import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_predict,GridSearchCV
from sklearn.metrics import confusion_matrix,  precision_score, recall_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
import anonypy



class Preprocessing:
    def __init__(self,parameters=None):
        self.parameters=parameters
        self.run_preprocess()


    def run_preprocess(self):
        self.stop=False

        # import dataset
        self.dataset=self.import_data(self.parameters['filepath'])
        self._dataset_original=self.dataset.copy()

        # anonymise specific columns in dataset
        if not self.stop and 'anonymiser' in self.parameters.keys():
            self.dataset=self.anonymiser(self.dataset,self.parameters['anonymiser']['k'],self.parameters['anonymiser']['feature_columns'])
            self._dataset_anonymized=self.dataset.copy()

        # generate dummies for multicategorical_features
        # split dataset in X and y 
        if not self.stop:
            features=self.parameters['features']
            if 'multicategorical_features' in features.keys():
                self.dataset=self.get_dummies(self.dataset,features['multicategorical_features'])
                self._dataset_dummies=self.dataset.copy()
            self._X,self._y=self.split_X_y(self.dataset,features['target'])

        # run train test _split
        tts_params=self.parameters['tts_params']
        if not self.stop:
            self._X_train, self._X_test, self._y_train, self._y_test = self.train_test_split(
                self._X,self._y,tts_params['random_state'],tts_params['test_size']
                )

        # run train test _split
        if not self.stop:
            self._X_train_scaled, self._X_test_scaled = self.scaler(self._X_train,self._X_test,features['continuous_features'])
        
        # output
        if not self.stop:
            self.preprocess_dataset={
                'X_train': self._X_train_scaled, 
                'X_test' : self._X_test_scaled,
                'y_train': self._y_train, 
                'y_test': self._y_test}

    def import_data(self,filepath):
        try : 
            return pd.read_csv(filepath)
        except FileNotFoundError:
            self.stop=True
            print('File not found, please check filepath')
    
    def get_dummies(self,dataset,multicategorical_features):
        return pd.get_dummies(dataset,columns=multicategorical_features,dtype=int)

    def plot_hist(self,dataset):
        rcParams['figure.figsize'] = 16,12
        dataset.hist(bins=20)
        plt.show()

    def split_X_y(self,dataset,target):
        y = dataset[target]
        X = dataset.drop(target,axis=1)
        return X,y
    
    def train_test_split(self,X,y,random_state=41, test_size=0.3,stratify_value=None):
        if stratify_value == None : 
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state,test_size = test_size)
        return X_train, X_test, y_train, y_test
    
    def scaler(self,X_train,X_test,continuous_features):
        #Copie les données dans un nouveau jeu de données
        X_train_scaled=X_train.copy()
        X_test_scaled=X_test.copy()

        #Normalise sur les valeurs continues avec standard scale
        scaler=StandardScaler()
        X_train_scaled[continuous_features]=scaler.fit_transform(X_train[continuous_features])
        X_test_scaled[continuous_features]=scaler.transform(X_test[continuous_features])
        # print("X_train size = {0}; X_train_scaled size = {1}\nX_test size = {2}; X_test_scaled size = {3}".format(X_train.size,X_train_scaled.size,X_test.size,X_test_scaled.size))

        return X_train_scaled,X_test_scaled
    
    def anonymiser(self,dataset,k,feature_columns,sensitive_columns='ano_index'):
        # Copie les colonnes à anonymiser dans un df_local
        df_ano=dataset[feature_columns].copy()

        #Crée une colonne 'index' de la longeur du df_ano
        if sensitive_columns=='ano_index':
            df_ano['ano_index']=range(len(dataset))

        # Genere l'anonymisation des colonnes 'features_columns'
        p=anonypy.Preserver(df_ano,feature_columns,sensitive_columns)
        rows = p.anonymize_k_anonymity(k=k)
        dfn = pd.DataFrame(rows)

        # remplace les colonnes anonymisées dans le dataset d'origine
        # ajoute les colonnes anonymisées dans la catégorie 'multicategorical_features' et les retire de la catégorie 'continuous_features'
        for column in feature_columns:
            dataset[column]=dfn[column].astype('str')
            dataset[column]=dataset[column].astype('category')
            try :
                self.parameters['features']['multicategorical_features'].append(column)
            except:
                None
            try:
                self.parameters['features']['continuous_features'].remove(column)
            except:
                None
        return dataset


class Classification:
    def __init__(self,dataset,model):
        self.model=model
        self.X_train=dataset['X_train']
        self.X_test=dataset['X_test']
        self.y_train=dataset['y_train']
        self.y_test=dataset['y_test']
        
        self.model_fit_and_predict()
        self.metrics()

    def model_fit_and_predict(self):
        self.model.fit(self.X_train,self.y_train)
        self.y_test_pred = self.model.predict(self.X_test)
    
    def metrics(self):
        self.metrics_results={}
        self.metrics_results['conf_matrix']= confusion_matrix(self.y_test,self.y_test_pred)
        self.metrics_results['precision']=precision_score(self.y_test,self.y_test_pred)
        self.metrics_results['recall']=recall_score(self.y_test,self.y_test_pred)
        self.metrics_results['accuracy']=accuracy_score(self.y_test,self.y_test_pred)
        self.metrics_results['f1']=f1_score(self.y_test,self.y_test_pred)
        return self.metrics_results
    
    def results(self):
        return pd.DataFrame({str(self.model):self.metrics_results}).T
    
class GridSearch:
    def __init__(self,gs_parameters,model):
        self.gs_parameters=gs_parameters
        self.model=model
        self.run_gs()

    def run_gs(self):
        self.gs_results={}
        for key in self.gs_parameters:
            pp=Preprocessing(self.gs_parameters[key])
            knn = KNeighborsClassifier(n_neighbors=15,n_jobs=-1)
            model=Classification(pp.preprocess_dataset,knn)
            self.gs_results[str(key)]=model.metrics_results
            self.df=pd.DataFrame(self.gs_results).T

    def results(self):
        return self.df

    def plot_results(self):
        self.df['accuracy'].plot()
        self.df['precision'].plot()
        self.df['recall'].plot()
        self.df['f1'].plot()
        plt.title(str(self.model))
        plt.xlabel('k')
        plt.ylabel('Score')
        plt.legend()
        plt.show()

    def plot_results_2(self):
        fig,ax=plt.subplots(1,4,figsize=(16,4))
        for i,key in enumerate(['accuracy','precision','recall','f1']):
            ax[i].plot(self.df[key])
            ax[i].set_title(key)
            ax[i].set_xlabel('k')
            ax[i].set_ylabel('Score')
        plt.tight_layout()
        plt.show()


def generate_preprocess_parameters(filepath,target,multicategorical_features,continuous_features,
                                   anonymiser_feature_columns=None,anonymiser_k=None,random_state=41,test_size=0.3):

    preprocess_parameters= {
            'filepath':filepath,
            'features':{
                    'target':target,
                    'multicategorical_features' : multicategorical_features,
                    'continuous_features' : continuous_features
                    },
            'tts_params':{
                    'random_state':random_state,
                    'test_size':test_size,
                    },
           
    }
    if not anonymiser_feature_columns==None and not anonymiser_k==None:
        preprocess_parameters['anonymiser']={
            'feature_columns':anonymiser_feature_columns,
            'k':anonymiser_k
            }
    return preprocess_parameters

def generate_gs_preprocess_parameters(gs_parameters,filepath,target,multicategorical_features,continuous_features,
                                   anonymiser_feature_columns=None,anonymiser_k=None,random_state=41,test_size=0.3):
    gs_preprocess_parameters={}
    if anonymiser_k==None:
        for anonymiser_k in gs_parameters['anonymiser_k']:
            gs_preprocess_parameters[anonymiser_k]=generate_preprocess_parameters(filepath,target,multicategorical_features,continuous_features,
                                                                                anonymiser_feature_columns,anonymiser_k)
    else : 
        print("please remove anonymiser_k for permanent value")

    return gs_preprocess_parameters


    



    

        