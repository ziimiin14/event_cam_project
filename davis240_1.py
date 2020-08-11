"""DAVIS240 test example.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""
from __future__ import print_function

import numpy as np
import cv2
from pyaer import libcaer
from pyaer.davis import DAVIS

device = DAVIS(noise_filter=True)

print ("Device ID:", device.device_id)
if device.device_is_master:
    print ("Device is master.")
else:
    print ("Device is slave.")
print ("Device Serial Number:", device.device_serial_number)
print ("Device String:", device.device_string)
print ("Device USB bus Number:", device.device_usb_bus_number)
print ("Device USB device address:", device.device_usb_device_address)
print ("Device size X:", device.dvs_size_X)
print ("Device size Y:", device.dvs_size_Y)
print ("Logic Version:", device.logic_version)
print ("Background Activity Filter:",
       device.dvs_has_background_activity_filter)

device.start_data_stream()
device.set_bias_from_json("davis240c_config.json")

clip_value = 3
histrange = [(0, v) for v in (180, 240)]

size = (240,180)
#  num_packet_before_disable = 1000
img_array = []
img2 = np.zeros((180,240,3))

def get_event(device):
    data = device.get_event('events')
    return data

#pol_Evt = np.array([[0,0,0,0,0]])
pol_Evt = [[0,0,0,0,0]]
time= np.array([[0,]],dtype=np.float32)
imu = [[0,0,0,0,0,0,0,0]]

while True:
    try:
        data1 = get_event(device)
        
        (pol_events_1, num_pol_event_1,
        special_events_1, num_special_event_1,
        frames_ts_1, frames_1, imu_events_1,
        num_imu_event_1) = data1

        #print(pol_events_1[:,0])
        #Update dictionary
        #pol_Evt = np.insert(pol_Evt,pol_Evt.shape[0],pol_events_1,axis=0)
        pol_Evt.append(pol_events_1)
        imu.append(imu_events_1)

    except KeyboardInterrupt:
        device.shutdown()
        break

#out.release()
cv2.destroyAllWindows()

pol_Evt.pop(0)
pol_Evt = np.concatenate(np.array(pol_Evt),axis=0)
imu.pop(0)
imu = np.concatenate(np.array(imu),axis=0)
print('Done concatenate')
time_temp = pol_Evt[:,0].reshape(pol_Evt.shape[0],1)
time = np.append(time,time_temp,axis=0)
time = time[1:,:].reshape(-1,1)
print('Done time assign')
#time = np.insert(time,time.shape[0],time_temp,axis=1)
pol_Evt = pol_Evt[:,1:]
pol_Evt = pol_Evt.astype('uint8')
print('Done event assign')

np.savez_compressed('polEvents_2.npz',pol_Evt)
np.savez_compressed('time_2.npz',time)
np.savez_compressed('imu_2.npz',imu)
