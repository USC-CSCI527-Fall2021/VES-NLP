import re
import json

import pandas as pd
from tqdm import tqdm

from spatial_algorithm import *


def data_converted(df):
    print(len(df))
    # timestamp	headPosx	headPosy	headPosz
    # headRotx	headRoty	headRotz
    # headRotQx	headRotQy	headRotQz	headRotQw
    # handRPosx	handRPosy	handRPosz
    # handRRotx	handRRoty	handRRotz
    # handRRotQx	handRRotQy	handRRotQz	handRRotQw
    # handLPosx	handLPosy	handLPosz
    # handLRotx	handLRoty	handLRotz
    # handLRotQx	handLRotQy	handLRotQz	handLRotQw
    # tracker1Posx	tracker1Posy	tracker1Posz
    # tracker1Rotx	tracker1Roty	tracker1Rotz
    # tracker1RotQx	tracker1RotQy	tracker1RotQz	tracker1RotQw
    # targetType	targetId	utterance	gestureStatus
    df_dict = df.to_dict(orient="records")

    collected_data = []

    # head_pos, head_rotation, utterance, target_object, target_operation
    idx = 0
    for item in tqdm(df_dict):
        if item['gestureStatus'] == "completed" or item['gestureStatus'] == "started" or item['gestureStatus'] == "recognized":
            head_pos = [item['headPosx'], item['headPosy'], item['headPosz']]
            head_rotation = [item['headRotx'], item['headRoty'], item['headRotz']]

            # TODO need to update for new version data => improve the efficiency
            if item['gestureStatus'] == "recognized":
                utterance = item['utterance']
                # if "microwave" in utterance or "MW" in utterance:
                #     target_object = "15-SHD_Microwave"
                # else:
                #     target_object = "10-SHD_Oven"
                #     # open the album
                #     # codes the island
                #     # find alan
                #     # open the outlook
                #     print(utterance)
                target_object = "7-SHD_DishWasher"

                if "open" in utterance:
                    target_operation = "open"
                else:
                    target_operation = "close"
                    # codes the island
                    # find alan

                    # coastal water
                    # cortana washer
                    # coastal ship
                    print(utterance)

                collected_data.append(
                    [head_pos, head_rotation, "recognized", utterance, target_object, target_operation])
                for data in collected_data:
                    if data[-3] == idx:
                        data[-3] = utterance
                    if data[-2] == idx:
                        data[-2] = target_object
                    if data[-1] == idx:
                        data[-1] = target_operation
                idx += 1
            else:
                collected_data.append([head_pos, head_rotation, item['gestureStatus'], idx, idx, idx])

    print(len(collected_data))
    new_df = pd.DataFrame(collected_data, columns=["head_pos", "head_rotation", "status", "utterance",
                                                   "target_object", "target_operation"])
    new_df.to_csv("data/dishwasher-1101/converted_input-1101-3.csv", index=None)


def main():
    # note: download the data into this folder
    df_1 = pd.read_csv("data/microwave-oven-1011/converted_input-1011.csv", index_col=False)
    df_2 = pd.read_csv("data/dishwasher-1101/converted_input-1101.csv", index_col=False)
    df_dict = pd.concat([df_1, df_2]).to_dict(orient="records")
    print(len(df_dict))

    # final data format
    # note: passing cosine_distance and euclidean_distance instead of gaze object list
    # [{cosine_distance: {}, euclidean_distance: {}, utterance: "", target_object: "", target_operation: ""}]

    with open("data/all_devices_1101.json", "r") as fin:
        all_devices = json.load(fin)

    all_completed_data = []
    for item in tqdm(df_dict):
        cosine_distance = {}
        euclidean_distance = {}
        for device in all_devices:
            distance_key = f'{device["unique_id"]}-{device["appliance_type"]}'
            head_pos = [float(pos) for pos in re.split(", |\[|\]|\(|\)", item["head_pos"]) if pos != '']
            head_rotation = [float(pos) for pos in re.split(", |\[|\]|\(|\)", item["head_rotation"]) if pos != '']
            equipment_pos = [float(pos) for pos in re.split(", |\[|\]|\(|\)", device["position"]) if pos != '']

            cosine_distance[distance_key] = get_cosine_distance(head_pos, rotation2direction(head_rotation),
                                                                equipment_pos)
            euclidean_distance[distance_key] = get_euclidean_distance(head_pos, equipment_pos)

        all_completed_data.append({
            "status": item["status"],
            "cosine_distance": cosine_distance,
            "euclidean_distance": euclidean_distance,
            "utterance": item["utterance"],
            "target_object": item["target_object"],
            "target_operation": item["target_operation"]
        })

    print(all_completed_data[0])
    print(len(all_completed_data))

    with open("data/data-1101/input_data-all-1101.json", "w") as fout:
        json.dump(all_completed_data, fout, indent=4)


if __name__ == '__main__':
    # df_1 = pd.read_csv("data/dishwasher-1101/SmartHome_Exp_2021-10-25-04-45-39.csv", index_col=False)
    # df_2 = pd.read_csv("data/dishwasher-1101/SmartHome_Exp_2021-10-25-07-46-38.csv", index_col=False)
    # df_3 = pd.read_csv("data/dishwasher-1101/SmartHome_Exp_2021-10-25-07-55-15.csv", index_col=False)

    # print(list(df_1.dropna(subset=['gestureStatus'])["gestureStatus"]))

    # data_converted(df_3)

    # df_1 = pd.read_csv("data/dishwasher-1101/converted_input-1101-1.csv")
    # df_2 = pd.read_csv("data/dishwasher-1101/converted_input-1101-2.csv")
    # df_3 = pd.read_csv("data/dishwasher-1101/converted_input-1101-3.csv")
    #
    # pd.concat([df_1, df_2, df_3]).to_csv("data/dishwasher-1101/converted_input-1101.csv", index=None)

    main()
    # test = re.split(", |\[|\]|\(|\)", "(8.696393, 235.2555, 354.9054)")
    # data = [float(pos) for pos in test if pos != '']
    # print(data)
