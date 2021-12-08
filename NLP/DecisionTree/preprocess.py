import json


object_name_list = ["-41710-SHD_Camera", "-41612-SHD_Boombox", "23468-SHD_CounterDrawer", "23782-SHD_CounterDrawer", "24078-SHD_Light", "24374-SHD_Cabinet", "24428-SHD_Refrigerator", "24640-SHD_Oven", "25794-SHD_Lightswitch", "28418-SHD_CounterSink", "28588-SHD_Toaster", "29242-SHD_ExtractorFan", "29452-SHD_Cabinet", "29826-SHD_Microwave", "30340-SHD_DishWasher", "31064-SHD_RubbishBin", "31700-SHD_CounterDrawer", "34768-SHD_CounterDrawer", "36362-SHD_CoffeeMachine", "36814-SHD_Cabinet"]

def preprocess():

    f = open("input_data-all-1011.json")
    input_data = json.load(f)#["data"]
    f.close()

    f = open("availability.json")
    availability_list = json.load(f)
    f.close()

    f = open("all_devices_0925.json")
    device_info = json.load(f)["objects"]
    f.close()

    f = open("entity_temp.json")
    entity_json = json.load(f)
    f.close()

    spacial_val_list = []
    intent_list = []
    correct_answer_list = []
    utterance_list = []
    availability_val_list = []
    entity_list = []
    training_data =[]

    for i in input_data:
        cosine_distance_list = [i["cosine_distance"]]

        euclidean_distance_list = [i["euclidean_distance"]]

        intent = i["target_operation"]
        availability_val_list = [availability_list[intent]]
        correct_answer = i["target_object"]
        utterance = i["utterance"]
        #print(utterance)
        if "microwave" in utterance:
            #print("microwave")
            entity_list = [entity_json["microwave"]]
        elif "oven" in utterance:
            #print("oven")
            entity_list = [entity_json["oven"]]
        else:
            entity_list = [entity_json["no_entity"]]

        #print(spacial_val_list)
        #print(availability_val_list)
        #print(entity_list)
        #print(correct_answer)
        row = [cosine_distance_list, euclidean_distance_list, availability_val_list, entity_list, correct_answer]
        training_data.append(row)



    return training_data



