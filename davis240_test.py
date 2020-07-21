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
    return data

filePath = 'video_file/test2.avi' # or rotation_rpm
out= cv2.VideoWriter(filePath,cv2.VideoWriter_fourcc(*'MPEG'),50,size,0)
dict_temp = {}
i = 0

while True:
    try:
        data = get_event_hist(device)
        data1 = get_event(device)
        if data is not None:
            (pol_events, num_pol_event,
             special_events, num_special_event,
             frames_ts, frames, imu_events,
             num_imu_event) = data
            (pol_events_1, num_pol_event_1,
             special_events_1, num_special_event_1,
             frames_ts_1, frames_1, imu_events_1,
             num_imu_event_1) = data1


            #Update dictionary
            dict_temp.update({i:pol_events_1})
            i = i+1


            #if frames.shape[0] != 0:
            #    cv2.imshow("frame", frames[0])

            #print ("Number of events:", num_pol_event, "Number of Frames:",
            #       frames.shape, "Exposure:",
            #       device.get_config(
            #           libcaer.DAVIS_CONFIG_APS,
            #           libcaer.DAVIS_CONFIG_APS_EXPOSURE))

            if num_pol_event != 0:
                img = pol_events[..., 1]-pol_events[..., 0]
                img = img.astype('uint8')

                #img = np.clip(img, -clip_value, clip_value)
                #img = img+clip_value
                #img = img/float(clip_value*2)
                

                # Show the frame
                #cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
                #cv2.resizeWindow('frame', 960,720)
                #cv2.imshow("frame", img)

                #Convert img from 1 channel into 3 channels
                #img2[:,:,0] = img
                #img2[:,:,1] = img
                #img2[:,:,2] = img
                #img2 = np.array(img2,dtype=np.uint8)

		#Output img into .avi file
                out.write(img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            pass

    except KeyboardInterrupt:
        device.shutdown()
        break

out.release()
cv2.destroyAllWindows()


#Save dict_temp into  file
np.savez_compressed('../test2.npz',dict_temp)



#out = cv2.VideoWriter('project1.mp4',cv2.VideoWriter_fourcc(*'XVID'),15,size)

#for i in range(len(img_array)):
#    out.write(img_array[i])

#out.release()
