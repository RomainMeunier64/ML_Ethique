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
    def __init__(self,dataset,parameters):
        self.__dataset=dataset.copy()
        self.dataset_original=dataset.copy()
        self.parameters=parameters
        self.run_preprocess()

    def run_preprocess(self):
        self.stop=False

        # anonymise specific columns in dataset
        if not self.stop and 'anonymiser' in self.parameters.keys():
            try:
                sensitive_columns=self.parameters['anonymiser']['sensitive_columns']
            except:
                sensitive_columns='ano_index'
            self.anonymiser(self.parameters['anonymiser']['k'],self.parameters['anonymiser']['feature_columns'],sensitive_columns)
            self.dataset_anonymized=self.__dataset.copy()

        # generate dummies for multicategorical_features
        if not self.stop:
            if 'multicategorical_features' in self.parameters['features'].keys():
                self.get_dummies(self.parameters['features']['multicategorical_features'])
                self.dataset_dummies=self.__dataset.copy()
            
        # split dataset in X and y 
        if not self.stop:
            self.split_X_y(self.parameters['features']['target'])

        # run train test _split
        if not self.stop:
            self.train_test_split(self.parameters['tts_params']['random_state'],self.parameters['tts_params']['test_size'])

        # run scaler
        if not self.stop:
            self.scaler(self.parameters['features']['continuous_features'])
        
        # output
        if not self.stop:
            self.dataset_preprocessed={
                'X_train': self._X_train_scaled, 
                'X_test' : self._X_test_scaled,
                'y_train': self._y_train, 
                'y_test': self._y_test}
    
    def get_dummies(self,multicategorical_features):
        # Transpose chacune des colonnes multicatégories en dummies
        self.__dataset=pd.get_dummies(self.__dataset,columns=multicategorical_features,dtype=int)

    def split_X_y(self,target):
        # Split le dataset en X et y datasets
        self._y = self.__dataset[target]
        self._X = self.__dataset.drop(target,axis=1)

    def train_test_split(self,random_state=41, test_size=0.3,stratify_value=None):
        # Definit les variables locales à partir du fichier de paramètres
        if stratify_value == None : 
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
                self._X, self._y, random_state = random_state,test_size = test_size)
    
    def scaler(self,continuous_features):
        #Copie les données dans un nouveau jeu de données
        self._X_train_scaled=self._X_train.copy()
        self._X_test_scaled=self._X_test.copy()

        #Normalise sur les valeurs continues avec standard scale
        scaler=StandardScaler()
        self._X_train_scaled[continuous_features]=scaler.fit_transform(self._X_train[continuous_features])
        self._X_test_scaled[continuous_features]=scaler.transform(self._X_test[continuous_features])
    
    def anonymiser(self,k,feature_columns,sensitive_columns):
        # Copie les colonnes à anonymiser dans un df_local
        df_ano=self.__dataset[feature_columns].copy()

        #Crée une colonne 'index' de la longeur du df_ano
        df_ano['ano_index']=range(len(df_ano))

        # Genere l'anonymisation des colonnes 'features_columns'
        p=anonypy.Preserver(df_ano,feature_columns,sensitive_columns)
        rows = p.anonymize_k_anonymity(k=k)
        dfn = pd.DataFrame(rows)

        # remplace les colonnes anonymisées dans le dataset d'origine
        self.anonymized_columns={}
        for column in feature_columns:
            self.__dataset[column]=dfn[column].astype('str')
            self.__dataset[column]=self.__dataset[column].astype('category')
            # ajoute les colonnes anonymisées dans la catégorie 'multicategorical_features' et les retire de la catégorie 'continuous_features'
            try :
                self.parameters['features']['multicategorical_features'].append(column)
            except:
                None
            try:
                self.parameters['features']['continuous_features'].remove(column)
            except:
                None
            # ajoute les colonnes anonymisées et le nombre de valeur par catégories pour output
            self.anonymized_columns[column]=self.__dataset[column].value_counts().sort_index().to_frame()

    def plot_hist(self,dataset):
        rcParams['figure.figsize'] = 16,12
        dataset.hist(bins=20)
        plt.show()


class Classification:
    def __init__(self,dataset,model):
        self.model=model
        self._X_train=dataset['X_train']
        self._X_test=dataset['X_test']
        self._y_train=dataset['y_train']
        self._y_test=dataset['y_test']
        
        self.model_fit_and_predict()
        self.metrics()

    def model_fit_and_predict(self):
        self.model.fit(self._X_train,self._y_train)
        self.y_test_pred = self.model.predict(self._X_test)
    
    def metrics(self):
        self.metrics_results={}
        self.metrics_results['conf_matrix']= confusion_matrix(self._y_test,self.y_test_pred)
        self.metrics_results['precision']=precision_score(self._y_test,self.y_test_pred)
        self.metrics_results['recall']=recall_score(self._y_test,self.y_test_pred)
        self.metrics_results['accuracy']=accuracy_score(self._y_test,self.y_test_pred)
        self.metrics_results['f1']=f1_score(self._y_test,self.y_test_pred)
        return self.metrics_results
    
    def results(self):
        return pd.DataFrame({str(self.model):self.metrics_results}).T
    
class GridSearch:
    def __init__(self,dataset,gs_parameters,model):
        self.dataset=dataset.copy()
        self.gs_parameters=gs_parameters
        self.model=model
        self.run_gs()

    def run_gs(self):
        self.gs_results={}
        for key in self.gs_parameters:
            pp=Preprocessing(self.dataset,self.gs_parameters[key])
            knn = KNeighborsClassifier(n_neighbors=15,n_jobs=-1)
            model=Classification(pp.dataset_preprocessed,knn)
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


def generate_preprocess_parameters(target,multicategorical_features,continuous_features,
                                   anonymiser_feature_columns=None,anonymiser_k=None,random_state=41,test_size=0.3):

    preprocess_parameters= {
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

def generate_gs_preprocess_parameters(gs_parameters,target,multicategorical_features,continuous_features,
                                   anonymiser_feature_columns=None,anonymiser_k=None,random_state=41,test_size=0.3):
    gs_preprocess_parameters={}
    if anonymiser_k==None:
        for anonymiser_k in gs_parameters['anonymiser_k']:
            gs_preprocess_parameters[anonymiser_k]=generate_preprocess_parameters(target,multicategorical_features,continuous_features,
                                                                                anonymiser_feature_columns,anonymiser_k)
    else : 
        print("please remove anonymiser_k for permanent value")

    return gs_preprocess_parameters


    



    

        