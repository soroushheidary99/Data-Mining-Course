import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import pickle 
import numpy as np
from tensorflow import keras
import tensorflow
from scipy import spatial
import random 




PATH = 'Models/'
FEATURES = ['danceability', 'energy', 'loudness', 'speechiness','acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo','duration_ms',]
OUPUT_FEATURES = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms',
       'time_signature']



def getUserInput() : 
    return pd.read_csv('input_tracks.csv')



def getDatabase() : 
    global database_df       
    database_df = pickle.load(open(PATH + 'database_df', 'rb'))



def getRecommendations(label, label_name): 
    dominant_label = max(set(label), key = list(label).count)
    results = database_df[database_df[label_name] == dominant_label]
    return results.sample(5)[OUPUT_FEATURES]




def loadPCA(x, PATH=None) :
    sc_scaler = pickle.load(open(PATH + 'sc_scaler', 'rb'))
    fa_model = pickle.load(open(PATH + 'pca_model', 'rb'))
    fa_model_clusterer = pickle.load(open(PATH + 'pca_model_clusterer', 'rb'))
    x_transformed = sc_scaler.transform(x) 
    x_fa = fa_model.transform(x_transformed)
    label = fa_model_clusterer.predict(x_fa)
    return getRecommendations(label, 'pca_labels')
    



def loadFA(x, PATH=None) :
    sc_scaler = pickle.load(open(PATH + 'sc_scaler', 'rb'))
    pca_model = pickle.load(open(PATH + 'fa_model', 'rb'))
    pca_model_clusterer = pickle.load(open(PATH + 'fa_model_clusterer', 'rb'))
    x_transformed = sc_scaler.transform(x) 
    x_pca = pca_model.transform(x_transformed)
    label = pca_model_clusterer.predict(x_pca)
    return getRecommendations(label, 'fa_labels')




def loadAE(x, PATH=None) : 
    normal_scaler = pickle.load(open(PATH + 'normal_scaler', 'rb'))
    ae_model = keras.models.load_model(PATH + 'ae_model')
    ae_model_clusterer = pickle.load(open(PATH + 'ae_model_clusterer', 'rb'))
    x_transformed = normal_scaler.transform(x) 
    x_ae = ae_model.predict(x_transformed)
    label = ae_model_clusterer.predict(x_ae)
    return getRecommendations(label, 'ae_labels')




def loadBCPCA(x, PATH=None) :
    normal_scaler = pickle.load(open(PATH + 'normal_scaler', 'rb'))
    bc_model = pickle.load(open(PATH + 'bc_pca_model_0', 'rb'))
    pca_model = pickle.load(open(PATH + 'bc_pca_model_1', 'rb'))
    bc_pca_model_clusterer = pickle.load(open(PATH + 'bc_pca_model_clusterer', 'rb'))
    x_transformed = normal_scaler.transform(x) + 1
    x_temp = pd.DataFrame()
    for i in range(x_transformed.shape[1]):
        x_temp[str(i)] = bc_model(x_transformed[:, i])[0]
    x_bc_pca = pca_model.transform(x_temp)
    label = bc_pca_model_clusterer.predict(x_bc_pca)
    return getRecommendations(label, 'bc_pca_labels')




def loadVAE(x, PATH=None) : 
    input_data = tensorflow.keras.layers.Input(shape=(10))
    encoder = tensorflow.keras.layers.Dense(10, activation='tanh')(input_data)
    encoder = tensorflow.keras.layers.Dense(8, activation='tanh')(encoder)
    def sample_latent_features(distribution):
        distribution_mean, distribution_variance = distribution
        batch_size = tensorflow.shape(distribution_variance)[0]
        random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
        return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random
    distribution_mean = tensorflow.keras.layers.Dense(5, name='mean')(encoder)
    distribution_variance = tensorflow.keras.layers.Dense(5, name='log_variance')(encoder)
    latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])
    vae_model = tensorflow.keras.Model(input_data, latent_encoding)
    vae_model.load_weights(PATH + 'vae_model_weights')
    
    normal_scaler = pickle.load(open(PATH + 'normal_scaler', 'rb'))
    vae_model_clusterer = pickle.load(open(PATH + 'vae_model_clusterer', 'rb'))
    x_transformed = normal_scaler.transform(x) 
    x_vae = vae_model.predict(x_transformed)
    label = vae_model_clusterer.predict(x_vae)
    return getRecommendations(label, 'vae_labels')



def euclideanSimilarity(X, x) : 
    X = pd.DataFrame(X)
    x = pd.Series(x)
    X['cos'] = X.apply(lambda r :1 - spatial.distance.euclidean(r, x), axis=1)
    res = X.sort_values('cos', ascending=False).index[:3].to_list()
    return res


def cosineSimilarity(X, x) : 
    X = pd.DataFrame(X)
    x = pd.Series(x)
    X['cos'] = X.apply(lambda r :1 - spatial.distance.cosine(r, x), axis=1)
    res = X.sort_values('cos', ascending=False).index[:3].to_list()
    return res


def dotProductSimilaritiy(X, x) : 
    similarities = X.dot(x)
    similarities = pd.Series(similarities)
    res = similarities.sort_values(ascending=False).index[:3].to_list()
    return res


def useSimilarities(x) : 
    index_list_cos = []
    index_list_dot = []
    index_list_euc = []
    
    for r in range(len(x)) : 
        row = x.iloc[r]
        res1 = cosineSimilarity(database_df[FEATURES], row)
        res2 = euclideanSimilarity(database_df[FEATURES], row)
        res3 = dotProductSimilaritiy(database_df[FEATURES], row)

        for i in res1 :
            index_list_cos.append(i)
        for j in res2 :
            index_list_euc.append(j)
        for k in res3 :
            index_list_dot.append(k)
        
    index_list_cos = np.array(index_list_cos)
    index_list_euc = np.array(index_list_euc)
    index_list_dot = np.array(index_list_dot)

    random.seed(420)    #:D
    r1 = random.sample(range(0, len(index_list_cos)), 10)
    r2 = random.sample(range(0, len(index_list_cos)), 10)
    r3 = random.sample(range(0, len(index_list_cos)), 10)


    universal_mixes = (index_list_cos[r1], index_list_euc[r2], index_list_dot[r3])
    um1 = pd.DataFrame(database_df.iloc[universal_mixes[0]], columns=OUPUT_FEATURES)
    um1.to_csv('OutputMixes/univerasl_mix_cosine_dis.csv')
    um2 = pd.DataFrame(database_df.iloc[universal_mixes[1]], columns=OUPUT_FEATURES)
    um2.to_csv('OutputMixes/univerasl_mix_euclidian_dis.csv')
    um3 = pd.DataFrame(database_df.iloc[universal_mixes[2]], columns=OUPUT_FEATURES)
    um3.to_csv('OutputMixes/univerasl_mix_dot_product.csv')


