########################################################################
#
# Copyright (c) 2020, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Multi cameras sample showing how to open multiple ZED in one program
"""

import pyzed.sl as sl
import os
import cv2
import numpy as np
import threading
import time
import signal

zed_list = []
left_list = []
depth_list = []
timestamp_list = []
thread_list = []
stop_signal = False


def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()


def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        #if err == sl.ERROR_CODE.SUCCESS:
            #zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            #zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.DEPTH)
            #timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.001) #1ms
    zed_list[index].close()


def main():
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global timestamp_list
    global thread_list
    signal.signal(signal.SIGINT, signal_handler)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD2K # either HD720 at 60fps or HD1080 at 30
    init.camera_fps = 10  # The framerate is lowered to avoid any USB3 bandwidth issues  60 fps only at HD720


    #List and open cameras
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    rec_timestamp = str(int(time.time()))
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        timestamp_list.append(0)
        last_ts_list.append(0)
        
        #status = zed_list[index].open(init)
        path_create = rf'/home/student/Videos/Rec_1/Rec'
        os.makedirs(path_create, exist_ok=True)
        output_path = os.path.join(rf'/home/student/Videos/Rec_1/Rec', f"rec_{index}_{rec_timestamp}.svo")
        status = zed_list[index].open(init)
        params = sl.RecordingParameters(video_filename=output_path, compression_mode=sl.SVO_COMPRESSION_MODE.LOSSLESS)
        err = zed_list[index].enable_recording(params)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index = index +1
    
    #Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()

    key = ''
    while key != "q":  # for 'q' key
        key = input("Enter q to stop.")

    #Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()

    print("\nFINISH")

if __name__ == "__main__":
    main()
