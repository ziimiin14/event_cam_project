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
# set new bias after data streaming
device.set_bias_from_json("davis240c_config.json")

clip_value = 3
histrange = [(0, v) for v in (180, 240)]

size = (240,180)
#  num_packet_before_disable = 1000
img_array = []
img2 = np.zeros((180,240,3))

def get_event_hist(device):
    data = device.get_event("events_hist")

    return data



def get_event(device):
    data = device.get_event('events')


out= cv2.VideoWriter('output007.avi',cv2.VideoWriter_fourcc(*'MPEG'),25,size)

while True:
    
    data = get_event_hist(device)
    if data is not None:
        (pol_events, num_pol_event,
         special_events, num_special_event,
         frames_ts, frames, imu_events,
         num_imu_event) = data
        if frames.shape[0] != 0:
            cv2.imshow("frame", frames[0])

        print ("Number of events:", num_pol_event, "Number of Frames:",
               frames.shape, "Exposure:",
               device.get_config(
                   libcaer.DAVIS_CONFIG_APS,
                   libcaer.DAVIS_CONFIG_APS_EXPOSURE))

        if num_pol_event != 0:
            img = pol_events[..., 1]-pol_events[..., 0]
            #img = np.clip(img, -clip_value, clip_value)
            #img = img+clip_value
            #img = img/float(clip_value*2)
            #img = np.array(img,dtype=np.float32)
            img = np.array(img,dtype = np.uint8)
            #print(img.max())
            cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('frame', 960,720)
            cv2.imshow("frame", img)
            out.write(img)
            #img = np.array(img,dtype = np.uint8)
            #img2[:,:,0] = img
            #img2[:,:,1] = img
            #img2[:,:,2] = img
            #img2 = np.array(img2,dtype = np.uint8)
            #img_array.append(img2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break


device.shutdown()
out.release()
cv2.destroyAllWindows()

#out = cv2.VideoWriter('project1.mp4',cv2.VideoWriter_fourcc(*'XVID'),15,size)

#for i in range(len(img_array)):
#    out.write(img_array[i])


