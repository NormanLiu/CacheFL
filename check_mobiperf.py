import json
import numpy as np

uplink_trace = {}
downlink_trace = {}

id_list = []
common_list = []

for i in range(5):
    with open('measurement' + str(i + 1)) as f:
        data = json.load(f)

    for measurment in range(len(data)):
        measure_dict = data[measurment]
        if measure_dict['type'] == 'tcpthroughput':
            parameters = measure_dict['parameters']
            values = measure_dict['values']
            if 'tcp_speed_results' in values:
                if parameters['dir_up'] == True:
                    if values['tcp_speed_results'] != '[]':
                        if measure_dict['id'] in id_list:
                            print(measure_dict['id'])
                        else:
                            id_list.append(measure_dict['id'])
for i in range(5):
    with open('measurement' + str(i + 1)) as f:
        data = json.load(f)

    for measurment in range(len(data)):
        measure_dict = data[measurment]
        if measure_dict['type'] == 'tcpthroughput':
            parameters = measure_dict['parameters']
            values = measure_dict['values']
            if 'tcp_speed_results' in values:
                if parameters['dir_up'] == False:
                    if values['tcp_speed_results'] != '[]':
                        if (measure_dict['id'] in id_list) & (measure_dict['id'] not in common_list):
                            common_list.append(measure_dict['id'])


for device_id in common_list:
    uplink_trace[device_id] = []
    downlink_trace[device_id] = []

for i in range(5):
    with open('measurement' + str(i + 1)) as f:
        data = json.load(f)

    for measurment in range(len(data)):
        measure_dict = data[measurment]
        if (measure_dict['type'] == 'tcpthroughput') & (measure_dict['id'] in common_list):
            parameters = measure_dict['parameters']
            values = measure_dict['values']
            if 'tcp_speed_results' in values:
                if parameters['dir_up'] == True:
                    if values['tcp_speed_results'] != '[]':
                        uplink_trace[measure_dict['id']].extend(json.loads(values['tcp_speed_results']))
                else:
                    if values['tcp_speed_results'] != '[]':
                        downlink_trace[measure_dict['id']].extend(json.loads(values['tcp_speed_results']))

