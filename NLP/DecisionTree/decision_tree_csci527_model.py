#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
from joblib import dump, load

# In[2]:


intent = 'open'
entity = 'SHD_Microwave'

# In[3]:


euclidean_distance = {"2-SHD_CounterDrawer": 20.50502587119681,
                      "20-SHD_Freezer": 24.453726432477534,
                      "13-SHD_Toaster": 24.768514546222804,
                      "19-SHD_Camera": 20.681118880622744,
                      "6-SHD_ExtractorFan": 22.61665282314009,
                      "11-SHD_RubbishBin": 24.426732386024344,
                      "12-SHD_CoffeeMachine": 24.775815544244967,
                      "8-SHD_CounterSink": 24.49545772053363,
                      "7-SHD_DishWasher": 24.491005958483015,
                      "17-SHD_Lightswitch": 24.997402587197946,
                      "1-SHD_Cabinet": 20.641921569138145,
                      "21-SHD_Boombox": 20.470055883325053,
                      "18-SHD_Light": 22.125494492337353,
                      "15-SHD_Microwave": 23.770410609210153,
                      "10-SHD_Oven": 22.57053157766672,
                      "3-SHD_CounterDrawer": 21.577536932153553,
                      "9-SHD_CounterDrawer": 24.474609314521256,
                      "5-SHD_Cabinet": 24.709140016326966,
                      "14-SHD_CounterDrawer": 20.475813970107527,
                      "4-SHD_Cabinet": 24.71312866851242,
                      "16-SHD_Camera": 24.657978525468195}

# In[4]:


cosine_distance = {"2-SHD_CounterDrawer": 0.7190903498318748,
                   "20-SHD_Freezer": 0.7916317494595606,
                   "13-SHD_Toaster": 0.7720335446810074,
                   "19-SHD_Camera": 0.787069237669655,
                   "6-SHD_ExtractorFan": 0.7632196062954557,
                   "11-SHD_RubbishBin": 0.8143255090627395,
                   "12-SHD_CoffeeMachine": 0.7859739855226258,
                   "8-SHD_CounterSink": 0.7337352215078303,
                   "7-SHD_DishWasher": 0.7740440305020966,
                   "17-SHD_Lightswitch": 0.7997821573829855,
                   "1-SHD_Cabinet": 0.7600667575842822,
                   "21-SHD_Boombox": 0.7818957087460939,
                   "18-SHD_Light": 0.8184256311974465,
                   "15-SHD_Microwave": 0.7200796664748723,
                   "10-SHD_Oven": 0.7025308143605704,
                   "3-SHD_CounterDrawer": 0.6958494139883896,
                   "9-SHD_CounterDrawer": 0.750485986084034,
                   "5-SHD_Cabinet": 0.8230823663086807,
                   "14-SHD_CounterDrawer": 0.7466674273376548,
                   "4-SHD_Cabinet": 0.7999301641520163,
                   "16-SHD_Camera": 0.7240878031603076}


# In[7]:


def decisiontree_prediction(cosine_distance, euclidean_distance, intent, entity, utterance):
    # cosine_distance_values = cosine_distance.values()
    # cosine_distance_values_list = list(cosine_distance_values)

    # euclidean_distance_values = euclidean_distance.values()
    # euclidean_distance_values_list = list(euclidean_distance_values)

    f = open("availability.json")
    availability_list = json.load(f)
    f.close()

    f = open("DecisionTree/entity_temp.json")
    entity_json = json.load(f)
    f.close()

    availability_val_list = [availability_list[intent]]
    if "microwave" in utterance:
        # print("microwave")
        entity_list = [entity_json["microwave"]]
    elif "MW" in utterance:
        # print("microwave")
        entity_list = [entity_json["microwave"]]
    elif "oven" in utterance:
        entity_list = [entity_json["oven"]]
    elif "allen" in utterance:
        entity_list = [entity_json["oven"]]
    elif "dishwasher" in utterance:
        # print("dishwasher")
        entity_list = [entity_json["dishwasher"]]
    elif "washer" in utterance:
        # print("washer")
        entity_list = [entity_json["dishwasher"]]
    elif "cabinet" in utterance:
        # print("washer")
        entity_list = [entity_json["cabinet"]]
    elif "freezer" in utterance:
        # print("washer")
        entity_list = [entity_json["freezer"]]
    else:
        entity_list = [entity_json["no_entity"]]

    # if "microwave" in entity or "Microwave" in entity:
    # print("microwave")
    # entity_list = [entity_json["microwave"]]
    # elif "oven" in entity or "Oven" in entity:
    # print("oven")
    # entity_list = [entity_json["oven"]]
    # else:
    # entity_list = [entity_json["no_entity"]]

    availability_val_list = availability_val_list[0]
    entity_list = entity_list[0]

    df1 = pd.DataFrame(columns=['2-SHD_CounterDrawer_cosine_distance_list',
                                '20-SHD_Freezer_cosine_distance_list',
                                '13-SHD_Toaster_cosine_distance_list',
                                '19-SHD_Camera_cosine_distance_list',
                                '6-SHD_ExtractorFan_cosine_distance_list',
                                '11-SHD_RubbishBin_cosine_distance_list',
                                '12-SHD_CoffeeMachine_cosine_distance_list',
                                '8-SHD_CounterSink_cosine_distance_list',
                                '7-SHD_DishWasher_cosine_distance_list',
                                '17-SHD_Lightswitch_cosine_distance_list',
                                '1-SHD_Cabinet_cosine_distance_list',
                                '21-SHD_Boombox_cosine_distance_list',
                                '18-SHD_Light_cosine_distance_list',
                                '15-SHD_Microwave_cosine_distance_list',
                                '10-SHD_Oven_cosine_distance_list',
                                '3-SHD_CounterDrawer_cosine_distance_list',
                                '9-SHD_CounterDrawer_cosine_distance_list',
                                '5-SHD_Cabinet_cosine_distance_list',
                                '14-SHD_CounterDrawer_cosine_distance_list',
                                '4-SHD_Cabinet_cosine_distance_list',
                                '16-SHD_Camera_cosine_distance_list',
                                '2-SHD_CounterDrawer_euclidean_distance_list',
                                '20-SHD_Freezer_euclidean_distance_list',
                                '13-SHD_Toaster_euclidean_distance_list',
                                '19-SHD_Camera_euclidean_distance_list',
                                '6-SHD_ExtractorFan_euclidean_distance_list',
                                '11-SHD_RubbishBin_euclidean_distance_list',
                                '12-SHD_CoffeeMachine_euclidean_distance_list',
                                '8-SHD_CounterSink_euclidean_distance_list',
                                '7-SHD_DishWasher_euclidean_distance_list',
                                '17-SHD_Lightswitch_euclidean_distance_list',
                                '1-SHD_Cabinet_euclidean_distance_list',
                                '21-SHD_Boombox_euclidean_distance_list',
                                '18-SHD_Light_euclidean_distance_list',
                                '15-SHD_Microwave_euclidean_distance_list',
                                '10-SHD_Oven_euclidean_distance_list',
                                '3-SHD_CounterDrawer_euclidean_distance_list',
                                '9-SHD_CounterDrawer_euclidean_distance_list',
                                '5-SHD_Cabinet_euclidean_distance_list',
                                '14-SHD_CounterDrawer_euclidean_distance_list',
                                '4-SHD_Cabinet_euclidean_distance_list',
                                '16-SHD_Camera_euclidean_distance_list',
                                '2-SHD_CounterDrawer_availability_val_list',
                                '20-SHD_Freezer_availability_val_list',
                                '13-SHD_Toaster_availability_val_list',
                                '19-SHD_Camera_availability_val_list',
                                '6-SHD_ExtractorFan_availability_val_list',
                                '11-SHD_RubbishBin_availability_val_list',
                                '12-SHD_CoffeeMachine_availability_val_list',
                                '8-SHD_CounterSink_availability_val_list',
                                '7-SHD_DishWasher_availability_val_list',
                                '17-SHD_Lightswitch_availability_val_list',
                                '1-SHD_Cabinet_availability_val_list',
                                '21-SHD_Boombox_availability_val_list',
                                '18-SHD_Light_availability_val_list',
                                '15-SHD_Microwave_availability_val_list',
                                '10-SHD_Oven_availability_val_list',
                                '3-SHD_CounterDrawer_availability_val_list',
                                '9-SHD_CounterDrawer_availability_val_list',
                                '5-SHD_Cabinet_availability_val_list',
                                '14-SHD_CounterDrawer_availability_val_list',
                                '4-SHD_Cabinet_availability_val_list',
                                '16-SHD_Camera_availability_val_list',
                                '2-SHD_CounterDrawer_entity',
                                '20-SHD_Freezer_entity',
                                '13-SHD_Toaster_entity',
                                '19-SHD_Camera_entity',
                                '6-SHD_ExtractorFan_entity',
                                '11-SHD_RubbishBin_entity',
                                '12-SHD_CoffeeMachine_entity',
                                '8-SHD_CounterSink_entity',
                                '7-SHD_DishWasher_entity',
                                '17-SHD_Lightswitch_entity',
                                '1-SHD_Cabinet_entity',
                                '21-SHD_Boombox_entity',
                                '18-SHD_Light_entity',
                                '15-SHD_Microwave_entity',
                                '10-SHD_Oven_entity',
                                '3-SHD_CounterDrawer_entity',
                                '9-SHD_CounterDrawer_entity',
                                '5-SHD_Cabinet_entity',
                                '14-SHD_CounterDrawer_entity',
                                '4-SHD_Cabinet_entity',
                                '16-SHD_Camera_entity'])

    df1 = df1.append({'2-SHD_CounterDrawer_cosine_distance_list': cosine_distance['2-SHD_CounterDrawer'],
                      '20-SHD_Freezer_cosine_distance_list': cosine_distance['20-SHD_Freezer'],
                      '13-SHD_Toaster_cosine_distance_list': cosine_distance['13-SHD_Toaster'],
                      '19-SHD_Camera_cosine_distance_list': cosine_distance['19-SHD_Camera'],
                      '6-SHD_ExtractorFan_cosine_distance_list': cosine_distance['6-SHD_ExtractorFan'],
                      '11-SHD_RubbishBin_cosine_distance_list': cosine_distance['11-SHD_RubbishBin'],
                      '12-SHD_CoffeeMachine_cosine_distance_list': cosine_distance['12-SHD_CoffeeMachine'],
                      '8-SHD_CounterSink_cosine_distance_list': cosine_distance['8-SHD_CounterSink'],
                      '7-SHD_DishWasher_cosine_distance_list': cosine_distance['7-SHD_DishWasher'],
                      '17-SHD_Lightswitch_cosine_distance_list': cosine_distance['17-SHD_Lightswitch'],
                      '1-SHD_Cabinet_cosine_distance_list': cosine_distance['1-SHD_Cabinet'],
                      '21-SHD_Boombox_cosine_distance_list': cosine_distance['21-SHD_Boombox'],
                      '18-SHD_Light_cosine_distance_list': cosine_distance['18-SHD_Light'],
                      '15-SHD_Microwave_cosine_distance_list': cosine_distance['15-SHD_Microwave'],
                      '10-SHD_Oven_cosine_distance_list': cosine_distance['10-SHD_Oven'],
                      '3-SHD_CounterDrawer_cosine_distance_list': cosine_distance['3-SHD_CounterDrawer'],
                      '9-SHD_CounterDrawer_cosine_distance_list': cosine_distance['9-SHD_CounterDrawer'],
                      '5-SHD_Cabinet_cosine_distance_list': cosine_distance['5-SHD_Cabinet'],
                      '14-SHD_CounterDrawer_cosine_distance_list': cosine_distance['14-SHD_CounterDrawer'],
                      '4-SHD_Cabinet_cosine_distance_list': cosine_distance['4-SHD_Cabinet'],
                      '16-SHD_Camera_cosine_distance_list': cosine_distance['16-SHD_Camera'],
                      '2-SHD_CounterDrawer_euclidean_distance_list': euclidean_distance['2-SHD_CounterDrawer'],
                      '20-SHD_Freezer_euclidean_distance_list': euclidean_distance['20-SHD_Freezer'],
                      '13-SHD_Toaster_euclidean_distance_list': euclidean_distance['13-SHD_Toaster'],
                      '19-SHD_Camera_euclidean_distance_list': euclidean_distance['19-SHD_Camera'],
                      '6-SHD_ExtractorFan_euclidean_distance_list': euclidean_distance['6-SHD_ExtractorFan'],
                      '11-SHD_RubbishBin_euclidean_distance_list': euclidean_distance['11-SHD_RubbishBin'],
                      '12-SHD_CoffeeMachine_euclidean_distance_list': euclidean_distance['12-SHD_CoffeeMachine'],
                      '8-SHD_CounterSink_euclidean_distance_list': euclidean_distance['8-SHD_CounterSink'],
                      '7-SHD_DishWasher_euclidean_distance_list': euclidean_distance['7-SHD_DishWasher'],
                      '17-SHD_Lightswitch_euclidean_distance_list': euclidean_distance['17-SHD_Lightswitch'],
                      '1-SHD_Cabinet_euclidean_distance_list': euclidean_distance['1-SHD_Cabinet'],
                      '21-SHD_Boombox_euclidean_distance_list': euclidean_distance['21-SHD_Boombox'],
                      '18-SHD_Light_euclidean_distance_list': euclidean_distance['18-SHD_Light'],
                      '15-SHD_Microwave_euclidean_distance_list': euclidean_distance['15-SHD_Microwave'],
                      '10-SHD_Oven_euclidean_distance_list': euclidean_distance['10-SHD_Oven'],
                      '3-SHD_CounterDrawer_euclidean_distance_list': euclidean_distance['3-SHD_CounterDrawer'],
                      '9-SHD_CounterDrawer_euclidean_distance_list': euclidean_distance['9-SHD_CounterDrawer'],
                      '5-SHD_Cabinet_euclidean_distance_list': euclidean_distance['5-SHD_Cabinet'],
                      '14-SHD_CounterDrawer_euclidean_distance_list': euclidean_distance['14-SHD_CounterDrawer'],
                      '4-SHD_Cabinet_euclidean_distance_list': euclidean_distance['4-SHD_Cabinet'],
                      '16-SHD_Camera_euclidean_distance_list': euclidean_distance['16-SHD_Camera'],
                      '2-SHD_CounterDrawer_availability_val_list': availability_val_list['2-SHD_CounterDrawer'],
                      '20-SHD_Freezer_availability_val_list': availability_val_list['20-SHD_Freezer'],
                      '13-SHD_Toaster_availability_val_list': availability_val_list['13-SHD_Toaster'],
                      '19-SHD_Camera_availability_val_list': availability_val_list['19-SHD_Camera'],
                      '6-SHD_ExtractorFan_availability_val_list': availability_val_list['6-SHD_ExtractorFan'],
                      '11-SHD_RubbishBin_availability_val_list': availability_val_list['11-SHD_RubbishBin'],
                      '12-SHD_CoffeeMachine_availability_val_list': availability_val_list['12-SHD_CoffeeMachine'],
                      '8-SHD_CounterSink_availability_val_list': availability_val_list['8-SHD_CounterSink'],
                      '7-SHD_DishWasher_availability_val_list': availability_val_list['7-SHD_DishWasher'],
                      '17-SHD_Lightswitch_availability_val_list': availability_val_list['17-SHD_Lightswitch'],
                      '1-SHD_Cabinet_availability_val_list': availability_val_list['1-SHD_Cabinet'],
                      '21-SHD_Boombox_availability_val_list': availability_val_list['21-SHD_Boombox'],
                      '18-SHD_Light_availability_val_list': availability_val_list['18-SHD_Light'],
                      '15-SHD_Microwave_availability_val_list': availability_val_list['15-SHD_Microwave'],
                      '10-SHD_Oven_availability_val_list': availability_val_list['10-SHD_Oven'],
                      '3-SHD_CounterDrawer_availability_val_list': availability_val_list['3-SHD_CounterDrawer'],
                      '9-SHD_CounterDrawer_availability_val_list': availability_val_list['9-SHD_CounterDrawer'],
                      '5-SHD_Cabinet_availability_val_list': availability_val_list['5-SHD_Cabinet'],
                      '14-SHD_CounterDrawer_availability_val_list': availability_val_list['14-SHD_CounterDrawer'],
                      '4-SHD_Cabinet_availability_val_list': availability_val_list['4-SHD_Cabinet'],
                      '16-SHD_Camera_availability_val_list': availability_val_list['16-SHD_Camera'],
                      '2-SHD_CounterDrawer_entity': entity_list['2-SHD_CounterDrawer'],
                      '20-SHD_Freezer_entity': entity_list['20-SHD_Freezer'],
                      '13-SHD_Toaster_entity': entity_list['13-SHD_Toaster'],
                      '19-SHD_Camera_entity': entity_list['19-SHD_Camera'],
                      '6-SHD_ExtractorFan_entity': entity_list['6-SHD_ExtractorFan'],
                      '11-SHD_RubbishBin_entity': entity_list['11-SHD_RubbishBin'],
                      '12-SHD_CoffeeMachine_entity': entity_list['12-SHD_CoffeeMachine'],
                      '8-SHD_CounterSink_entity': entity_list['8-SHD_CounterSink'],
                      '7-SHD_DishWasher_entity': entity_list['7-SHD_DishWasher'],
                      '17-SHD_Lightswitch_entity': entity_list['17-SHD_Lightswitch'],
                      '1-SHD_Cabinet_entity': entity_list['1-SHD_Cabinet'],
                      '21-SHD_Boombox_entity': entity_list['21-SHD_Boombox'],
                      '18-SHD_Light_entity': entity_list['18-SHD_Light'],
                      '15-SHD_Microwave_entity': entity_list['15-SHD_Microwave'],
                      '10-SHD_Oven_entity': entity_list['10-SHD_Oven'],
                      '3-SHD_CounterDrawer_entity': entity_list['3-SHD_CounterDrawer'],
                      '9-SHD_CounterDrawer_entity': entity_list['9-SHD_CounterDrawer'],
                      '5-SHD_Cabinet_entity': entity_list['5-SHD_Cabinet'],
                      '14-SHD_CounterDrawer_entity': entity_list['14-SHD_CounterDrawer'],
                      '4-SHD_Cabinet_entity': entity_list['4-SHD_Cabinet'],
                      '16-SHD_Camera_entity': entity_list['16-SHD_Camera']},
                     ignore_index=True)

    decisiontree_clf = load('DecisionTree/decisiontree_99accu_1126_besttuned.joblib')
    y_pred = decisiontree_clf.predict(df1)
    if y_pred == [0]:
        predicted_answer = '7-SHD_DishWasher'

    elif y_pred == [1]:
        predicted_answer = '15-SHD_Microwave'

    elif y_pred == [2]:
        predicted_answer = '10-SHD_Oven'

    elif y_pred == [3]:
        predicted_answer = '20-SHD_Freezer'

    elif y_pred == [4]:
        predicted_answer = '1-SHD_Cabinet'

    elif y_pred == [5]:
        predicted_answer = '5-SHD_Cabinet'

    else:
        predicted_answer = '4-SHD_Cabinet'

    return predicted_answer
