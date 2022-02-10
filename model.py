import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from nba_api.stats.static import teams,players
import basketball_reference_scraper as bc
from basketball_reference_scraper.teams import get_roster
import re
import unicodedata
Cleaned_data=pd.read_csv('Cleaned_data1.csv').drop(columns =['Unnamed: 0'])
players_dict5=players.get_players()
teams_dict5=teams.get_teams()

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    text = strip_accents(text.lower())
    text = re.sub('[ ]+', '', text)
    text = re.sub('[^0-9a-zA-Z_-]', '', text)
    return text

Cleaned_data['CLOSEST_DEFENDER']=Cleaned_data['CLOSEST_DEFENDER'].apply(text_to_id)
Cleaned_data['PLAYER_NAME']=Cleaned_data['PLAYER_NAME'].apply(text_to_id)

all_players=[]
for i in Cleaned_data['PLAYER_NAME']:
    if i not in all_players:
        all_players.append(i)
for j in Cleaned_data['CLOSEST_DEFENDER']:
    if j not in all_players:
        all_players.append(j)
label_enc = LabelEncoder()
label_enc.fit(all_players)
a = np.unique(Cleaned_data['CLOSEST_DEFENDER'])
@st.cache(suppress_st_warning=True)
def unique_deffplayers(save_constant):
    return a
Cleaned_data['CLOSEST_DEFENDER']=label_enc.transform(Cleaned_data['CLOSEST_DEFENDER'])
@st.cache(suppress_st_warning=True)
def get_encoder(x):
    return label_enc.transform(x)


k = np.unique(Cleaned_data['PLAYER_NAME'])
@st.cache(suppress_st_warning=True)
def unique_attplayers(save_constant):
    return k


@st.cache(suppress_st_warning=True)
def apis(save_constant):
    Teams5=[]
    abb5=[]
    for x in teams_dict5:
        Teams5.append(x['full_name'])
        abb5.append(str.lower(x['abbreviation']))
    images_dict5= dict(zip(Teams5, abb5))
    return teams_dict5,Teams5,abb5,images_dict5,players_dict5
@st.cache(suppress_st_warning=True)
def rosters(abbre):
    if abbre == 'phx':
        qer = list(bc.teams.get_roster(str.upper('pho'), 2015)['PLAYER'])
    elif abbre == 'cha':
        qer = list(bc.teams.get_roster(str.upper('cho'), 2015)['PLAYER'])
    elif abbre == 'bkn':
        qer = list(bc.teams.get_roster(str.upper('brk'), 2015)['PLAYER'])
    else:
        qer = list(bc.teams.get_roster(str.upper(abbre), 2015)['PLAYER'])
    return qer
@st.cache(suppress_st_warning=True)
def dataset_ids(opt,pdict):
    ids15 = []
    new_names15 = []
    error15 = 0
    for j in opt:
        j_ = text_to_id(j)
        if j_ not in k:
            st.write(j, 'is Unavaiable. Select a Different Player')
            error15 = 1
        new_names15.append(j_)
        for h in pdict:
            k_ = text_to_id(h['full_name'])
            if k_ == j_:
                ids15.append(h['id'])
    pimages_id15 = dict(zip(new_names15, ids15))
    return pimages_id15,new_names15,error15
@st.cache(suppress_st_warning=True)
def dataset_ids2(opt,pdict):
    ids151 = []
    new_names151 = []
    error151 = 0
    for j5 in opt:
        j5_ = text_to_id(j5)
        if j5_ not in a:
            st.write(j5, 'is Unavaiable. Select a Different Player')
            error151 = 1
        new_names151.append(j5_)
        for k5 in pdict:
            k5_ = text_to_id(k5['full_name'])
            if k5_ == j5_:
                ids151.append(k5['id'])
    pimages_id151 = dict(zip(new_names151, ids151))
    return pimages_id151,new_names151,error151
@st.cache(suppress_st_warning=True)
def get_ppictures(opt,cc,dict):
    images=[]
    for i in range(len(opt)):
        link='https://cdn.nba.com/headshots/nba/latest/1040x760/'+ str(dict.get(cc[i]))+".png"
        images.append(link)
    return images



scaler=MinMaxScaler()
def normalization(df):
  scaler.fit(Cleaned_data.drop(columns= ['PLAYER_NAME','SHOT_RESULT']))
  data = scaler.transform(df)
  data = pd.DataFrame(data)
  c = (Cleaned_data.drop(columns= ['PLAYER_NAME','SHOT_RESULT'])).columns
  columns = []
  for i in range(6):
    columns.append(c[i])
  data.columns = columns
  return data

new_df = normalization(Cleaned_data.drop(columns= ['PLAYER_NAME','SHOT_RESULT']))
new_df['SHOT_RESULT'] =  np.array(Cleaned_data['SHOT_RESULT'])
new_df['PLAYER_NAME'] =  np.array(Cleaned_data['PLAYER_NAME'])

# splitting data
#global x_train, x_test, x_val, y_train, y_test , y_val, players

L = len(new_df)
train = int(L*0.8)
test = train + int(L*0.1)
val = test + int(L*0.1) + 1

x_train = new_df[0:train].drop(columns = 'SHOT_RESULT')
y_train = new_df[0:train].drop(columns = ['Home/Away',	'PERIOD',	'GAME_CLOCK',	'SHOT_DIST','CLOSEST_DEFENDER',	'CLOSE_DEF_DIST'])

x_test = new_df[train:test].drop(columns = 'SHOT_RESULT')
y_test = new_df[train:test].drop(columns = ['Home/Away',	'PERIOD',	'GAME_CLOCK',	'SHOT_DIST','CLOSEST_DEFENDER',	'CLOSE_DEF_DIST'])

x_val = new_df[test:val].drop(columns = 'SHOT_RESULT')
y_val = new_df[test:val].drop(columns = ['Home/Away',	'PERIOD',	'GAME_CLOCK',	'SHOT_DIST','CLOSEST_DEFENDER',	'CLOSE_DEF_DIST'])

x_train = x_train.sort_values(by = 'PLAYER_NAME') # player names in alphabetical order
x_test = x_test.sort_values(by = 'PLAYER_NAME')
x_val = x_val.sort_values(by = 'PLAYER_NAME')

y_train = (y_train.sort_values(by = 'PLAYER_NAME'))['SHOT_RESULT'] # only have Shots Result
y_test = (y_test.sort_values(by = 'PLAYER_NAME'))['SHOT_RESULT']
y_val = (y_val.sort_values(by = 'PLAYER_NAME'))['SHOT_RESULT']

players = np.unique(new_df['PLAYER_NAME'])




@st.cache(suppress_st_warning=True)

def neural_network(features, player):

    features = {'Home/Away': [features[0]], 'PERIOD': [features[1]], 'GAME_CLOCK': [features[2]],
                'SHOT_DIST': [features[3]], 'CLOSEST_DEFENDER': [features[4]], 'CLOSE_DEF_DIST': [features[5]]}

    features_df = normalization(pd.DataFrame(features))

    player_indexes_train, = np.where(x_train['PLAYER_NAME'] == player)
    player_indexes_test, = np.where(x_test['PLAYER_NAME'] == player)
    player_indexes_val, = np.where(x_val['PLAYER_NAME'] == player)

    FFNN_train_x = (x_train[:][player_indexes_train[0]:player_indexes_train[-1]]).drop(columns=['PLAYER_NAME'])
    FFNN_train_y = y_train[:][player_indexes_train[0]:player_indexes_train[-1]]

    FFNN_test_x = (x_test[:][player_indexes_test[0]:player_indexes_test[-1]]).drop(columns=['PLAYER_NAME'])
    FFNN_test_y = y_test[:][player_indexes_test[0]:player_indexes_test[-1]]

    FFNN_val_x = (x_val[:][player_indexes_val[0]:player_indexes_val[-1]]).drop(columns=['PLAYER_NAME'])
    FFNN_val_y = y_val[:][player_indexes_val[0]:player_indexes_val[-1]]

    model = Sequential()
    model.add(Dense(4, input_dim=6, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


    history = model.fit(FFNN_train_x, FFNN_train_y, epochs=50,
                        batch_size=15, validation_data=(FFNN_val_x, FFNN_val_y),
                        verbose=2)

    return model.predict(features_df)





