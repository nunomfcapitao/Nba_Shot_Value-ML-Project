import streamlit as st
import model
import numpy as np
import matplotlib.pyplot as plt
from datetime import time
cache_constant=1
att_p=model.unique_attplayers(cache_constant)
deff_p=model.unique_deffplayers(cache_constant)

teams_dict,Teams,abb,images_dict,players_dict=model.apis(cache_constant)
Teams2=[]
st.sidebar.header('About This Project:')
st.sidebar.markdown('This work consists in exploring data-driven basketball training by using a Feedforward Neural Network model and calculating the probability of players scoring '
                    'from  pre-selected open offense plays and to see whom should we pass the ball to and go for the shot')
st.sidebar.subheader('Dataset, Github')
st.sidebar.markdown("https://www.kaggle.com/dansbecker/nba-shot-logs")
st.sidebar.markdown("https://github.com/nunomfcapitao/Nba_Shot_Value-ML-Project")
st.sidebar.markdown("The article for this project can be found on the github link")
st.sidebar.subheader('Authors:')
st.sidebar.markdown('Nuno Miguel Ferreira Capitão')
st.sidebar.markdown('João Coelho Lima Folhadela')
st.sidebar.subheader('Select Your Teams')
selected_oteam = st.sidebar.selectbox(
    "Home Team",
    Teams)
off=0
deff=st.sidebar.checkbox('Defense')
for x in teams_dict:
    if x['full_name']!= selected_oteam:
        Teams2.append(x['full_name'])
url="https://www.nba.com/.element/img/1.0/teamsites/logos/teamlogos_80x64/"+images_dict.get(selected_oteam)+".gif"
selected_dteam = st.sidebar.selectbox(
    "Away Team",
    Teams2)
st.header('Practical Example:  Who Should Shoot the Ball?')
zone_play=st.sidebar.radio('Select Your Zone Offense Play',['3-2','1-3-1','2-1-2'])
per=st.sidebar.number_input('Period',1,4)
gc=st.sidebar.slider('Game Clock (mins)',time(00),time(12))
gc=int(gc.hour)+float(gc.minute/100)
url2="https://www.nba.com/.element/img/1.0/teamsites/logos/teamlogos_80x64/"+images_dict.get(selected_dteam)+".gif"
col1, col2, col3 = st.columns([0.3,0.4,0.6])
with col2:
    st.write('Attacking')
    if deff==0:
        st.image(url)
    else:
        st.image(url2)
with col3:
    st.write('Defending')
    if deff==0:
        st.image(url2)
    else:
        st.image(url)
if zone_play=='3-2':
    st.image('https://media.discordapp.net/attachments/692368724996391004/937511883219566632/unknown.png?width=1060&height=675')
elif zone_play=='1-3-1':
    st.image('https://cdn.discordapp.com/attachments/692368724996391004/937784384570937404/pos_1_3_1.PNG')
else:
    st.image('https://cdn.discordapp.com/attachments/692368724996391004/937784410269433887/pos_2_1_2.PNG')
oteam=selected_oteam
dteam=selected_dteam
if deff!=0:
    oteam=selected_dteam
    dteam=selected_oteam

players=model.rosters(images_dict.get(oteam))
options1=st.multiselect('Select the Attacking Players', players)
pimages_id1, new_names1, error1 = model.dataset_ids(options1, players_dict)
d1, d2= st.columns([1,1])
players2=model.rosters(images_dict.get(dteam))
options2=st.multiselect('Select the Defending Players', players2)
pimages_id2, new_names2, error2 = model.dataset_ids2(options2, players_dict)
d1, d2, d3= st.columns([1,1,1])
with d1:
   st.write('Attacking positions:', options1)
with d2:
    st.write('')
with d3:
   st.write('Defending positions:', options2)
v1, v2, v3= st.columns([1,1,1])
if len(options1)==5 and len(options2)==5:
    I1 = model.get_ppictures(options1, new_names1, pimages_id1)
    with v1:
        st.image(I1[1], width=200)
        st.write(options1[1], 'has the ball')
    I2 = model.get_ppictures(options2, new_names2,pimages_id2)
    with v2:
        st.image('https://cdn.discordapp.com/attachments/692368724996391004/941002082398404698/VS-Versus-logo-letters-for-sports-icon-Graphics-5352736-1-1-580x387.png')
    with v3:
        st.image(I2[1], width=200)
        st.write(options2[1], 'is the closest defender')

if st.button('EVALUATE THE BEST SHOT') and len(options1)==5 and len(options2)==5 and error1==0 and error2==0:
    input1 = model.get_encoder(new_names1)
    input2 = model.get_encoder(new_names2)
    results=np.zeros(5)
    st.write('Calculating....')
    H=0
    #Shot distances
    if zone_play=='3-2':
        P1= 7.34
        V1= 6.90
        P0 = 7.34
        V0= 7.00
        P2= 7.30
        V2= 7.20
        P3= 3.60
        V3= 3.40
        P4= 3.30
        V4= 2.30
        if deff==0:
            H=1
        eva_0=np.array([H,per,gc,P0,input2[0],P0-V0])
        results[0]=model.neural_network(eva_0, new_names1[0])[0][0]
        eva_1 = np.array([H, per, gc, P1, input2[1], P1 - V1])
        results[1] = model.neural_network(eva_1, new_names1[1])[0][0]
        eva_2 = np.array([H, per, gc, P2, input2[2], P2 - V2])
        results[2] = model.neural_network(eva_2, new_names1[2])[0][0]
        eva_3 = np.array([H, per, gc, P3, input2[3], P3 - V3])
        results[3] = model.neural_network(eva_3, new_names1[3])[0][0]
        eva_4 = np.array([H, per, gc, P4, input2[4], P4 - V4])
        results[4] = model.neural_network(eva_4, new_names1[4])[0][0]
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        y_pos =-np.arange(len(options1))
        plt.barh(y_pos, results, align="center", alpha=0.5)
        plt.yticks(y_pos, options1)
        plt.xlabel("Probability of Scoring")
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        index=np.argmax(results)
        if index==0:
            st.write(options1[1],'should pass the ball to',options1[0],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[0],width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/3_2_0to1.gif')
        elif index==1:
            st.write(options1[1], 'should try and shoot since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[1], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/3_2_1.gif')
        elif index==2:
            st.write(options1[1], 'should pass the ball to', options1[2], 'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[2], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/3_2_1to2.gif')
        elif index==3:
            st.write(options1[1], 'should pass the ball to', options1[3], 'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[3], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/3_2-_1to3.gif')
        else:
            st.write(options1[1], 'should pass the ball to', options1[4], 'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[4], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/3_2_1to4.gif')
    elif zone_play=='1-3-1':
        P1 = 11.34
        V1 = 7.10
        P0 = 8.34
        V0 = 7.00
        P2 = 8.30
        V2 = 7.20
        P3 = 4.57
        V3 = 4.10
        P4 = 2.70
        V4 = 1.50
        if deff == 0:
            H = 1
        eva_0 = np.array([H, per, gc, P0, input2[0], P0 - V0])
        results[0] = model.neural_network(eva_0, new_names1[0])[0][0]
        eva_1 = np.array([H, per, gc, P1, input2[1], P1 - V1])
        results[1] = model.neural_network(eva_1, new_names1[1])[0][0]
        eva_2 = np.array([H, per, gc, P2, input2[2], P2 - V2])
        results[2] = model.neural_network(eva_2, new_names1[2])[0][0]
        eva_3 = np.array([H, per, gc, P3, input2[3], P3 - V3])
        results[3] = model.neural_network(eva_3, new_names1[3])[0][0]
        eva_4 = np.array([H, per, gc, P4, input2[4], P4 - V4])
        results[4] = model.neural_network(eva_4, new_names1[4])[0][0]
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        y_pos = -np.arange(len(options1))
        plt.barh(y_pos, results, align="center", alpha=0.5)
        plt.yticks(y_pos, options1)
        plt.xlabel("Probability of Scoring")
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        index = np.argmax(results)
        if index == 0:
            st.write(options1[1], 'should pass the ball to', options1[0],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[0], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/1-3-1_1to0.gif')
        elif index == 1:
            st.write(options1[1], 'should try and shoot since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[1], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/1-3-1_1.gif')
        elif index == 2:
            st.write(options1[1], 'should pass the ball to', options1[2],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[2], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/1-3-1_1to2.gif')
        elif index == 3:
            st.write(options1[1], 'should pass the ball to', options1[3], 'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[3], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/1-3-1_1to3.gif')
        else:
            st.write(options1[1], 'should pass the ball to', options1[4],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[4], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/1-3-1_1to4.gif')


    else:
        P1 = 11.34
        V1 = 6.40
        P0 = 11.34
        V0 = 6.40
        P2 = 6.00
        V2 = 4.10
        P3 = 6.00
        V3 = 4.10
        P4 = 6.50
        V4 = 7.34
        # np.array([Home, Peroid, Game_Clock, Shot_Dist, Closest_Defender, Closest_Defender_Dist]), 'jamesharden')
        if deff == 0:
            H = 1
        eva_0 = np.array([H, per, gc, P0, input2[0], P0 - V0])
        results[0] = model.neural_network(eva_0, new_names1[0])[0][0]
        eva_1 = np.array([H, per, gc, P1, input2[1], P1 - V1])
        results[1] = model.neural_network(eva_1, new_names1[1])[0][0]
        eva_2 = np.array([H, per, gc, P2, input2[2], P2 - V2])
        results[2] = model.neural_network(eva_2, new_names1[2])[0][0]
        eva_3 = np.array([H, per, gc, P3, input2[3], P3 - V3])
        results[3] = model.neural_network(eva_3, new_names1[3])[0][0]
        eva_4 = np.array([H, per, gc, P4, input2[4], P4 - V4])
        results[4] = model.neural_network(eva_4, new_names1[4])[0][0]
        fig, ax = plt.subplots()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_ticks([])
        y_pos = -np.arange(len(options1))
        plt.barh(y_pos, results, align="center", alpha=0.5)
        plt.yticks(y_pos, options1)
        plt.xlabel("Probability of Scoring")
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        index = np.argmax(results)
        if index == 0:
            st.write(options1[1], 'should pass the ball to', options1[0],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[0], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/2_1_2_1to07f58e684e8168d2b.gif')
        elif index == 1:
            st.write(options1[1], 'should try and shoot since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[1], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/2_1_2_1to11eae9dc0736dda8c.gif')
        elif index == 2:
            st.write(options1[1], 'should pass the ball to', options1[2],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[2], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/2_1_2_1to2a25af3ee63768be7.gif')
        elif index == 3:
            st.write(options1[1], 'should pass the ball to', options1[3],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[3], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/2_1_2_1to39b17f46548a85769.gif')
        else:
            st.write(options1[1], 'should pass the ball to', options1[4],'since he has the highest probability of scoring')
            r1, r2, r3 = st.columns([0.6, 1, 1])
            with r1:
                st.write('')
            with r2:
                st.image(I1[4], width=300)
            with r3:
                st.write('')
            st.image('https://s10.gifyu.com/images/2_1_2_1to4e7351d5a43492aa0.gif')