import numpy as np
import cv2
import ffmpeg
import time

output_file = 'colorbars.mov'
fps = 25
duration_sec = 10 

White_100 = (4095,4095,4095)
Yellow_100 = (4095,4095,0)
Cyan_100 = (0,4095,4095)
Green_100 = (0,4095,0)
Magenta_100 = (4095,0,4095)
Red_100 = (4095,0,0)
Blue_100 = (0,0,4095)
White_58 = (2375,2375,2375)
Yellow_58 = (2375,2375,0)
Cyan_58 = (0,2375,2375)
Green_58 = (0,2375,0)
Magenta_58 = (2375,0,2375)
Red_58 = (2375,0,0)
Blue_58 = (0,0,2375)
Grey_40 = (1638,1638,1638)
Step_0 = (0,0,0)
Step_10 = (410,410,410)
Step_20 = (819,819,819)
Step_30 = (1229,1229,1229)
Step_40 = (1638,1638,1638)
Step_50 = (2048,2048,2048)
Step_60 = (2457,2457,2457)
Step_70 = (2867,2867,2867)
Step_80 = (3276,3276,3276)
Step_90 = (3686,3686,3686)
Step_100 = (4095,4095,4095)
BT709_58_Yellow = (2356,2370,1480)
BT709_58_Cyan = (1964,2345,2368)
BT709_58_Green = (1915,2339,1420)
BT709_58_Magenta = (2206,1389,2336)
BT709_58_Red = (2178,1337,900)
BT709_58_Blue = (1184,805,2328)
Black_0 = (0,0,0)
Black_2 = (82,82,82)
Black_4 = (164,164,164)

bar_sz_a = 3840
bar_sz_b = 2160
bar_sz_c = 480
bar_sz_d = 412
bar_sz_e = 408
bar_sz_f = 272
bar_sz_g = 140
bar_sz_h = 136
bar_sz_i = 476
bar_sz_j = 876
bar_sz_k = 564

bar_grad_a = 3360
bar_grad_b = 1101
bar_grad_c = 2047
bar_grad_d = 212
ramp_low = 0.0
ramp_high = 4095.0


# Create a black image
imgbuffer = np.zeros((bar_sz_b, bar_sz_a, 3), np.uint16)

cv2.rectangle(imgbuffer,(0,0),(bar_sz_c,bar_sz_b//2+bar_sz_b//12),Grey_40,-1)

# 100% color bars
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*0,0),(bar_sz_c+bar_sz_d*1,bar_sz_b//12),White_100,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*1,0),(bar_sz_c+bar_sz_d*2,bar_sz_b//12),Yellow_100,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*2,0),(bar_sz_c+bar_sz_d*3,bar_sz_b//12),Cyan_100,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*3,0),(bar_sz_c+bar_sz_d*3+bar_sz_e,bar_sz_b//12),Green_100,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*3+bar_sz_e,0),(bar_sz_c+bar_sz_d*4+bar_sz_e,bar_sz_b//12),Magenta_100,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*4+bar_sz_e,0),(bar_sz_c+bar_sz_d*5+bar_sz_e,bar_sz_b//12),Red_100,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*5+bar_sz_e,0),(bar_sz_c+bar_sz_d*6+bar_sz_e,bar_sz_b//12),Blue_100,-1)

# 75% color bars
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*0,bar_sz_b//12),(bar_sz_c+bar_sz_d*1,bar_sz_b//2+bar_sz_b//12),White_58,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*1,bar_sz_b//12),(bar_sz_c+bar_sz_d*2,bar_sz_b//2+bar_sz_b//12),Yellow_58,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*2,bar_sz_b//12),(bar_sz_c+bar_sz_d*3,bar_sz_b//2+bar_sz_b//12),Cyan_58,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*3,bar_sz_b//12),(bar_sz_c+bar_sz_d*3+bar_sz_e,bar_sz_b//2+bar_sz_b//12),Green_58,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*3+bar_sz_e,bar_sz_b//12),(bar_sz_c+bar_sz_d*4+bar_sz_e,bar_sz_b//2+bar_sz_b//12),Magenta_58,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*4+bar_sz_e,bar_sz_b//12),(bar_sz_c+bar_sz_d*5+bar_sz_e,bar_sz_b//2+bar_sz_b//12),Red_58,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*5+bar_sz_e,bar_sz_b//12),(bar_sz_c+bar_sz_d*6+bar_sz_e,bar_sz_b//2+bar_sz_b//12),Blue_58,-1)

cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d*6+bar_sz_e,0),(bar_sz_a,bar_sz_b//2+bar_sz_b//12),Grey_40,-1)

cv2.rectangle(imgbuffer,(0,bar_sz_b//2+bar_sz_b//12),(bar_sz_c,bar_sz_b//2+bar_sz_b//12*2),White_58,-1)

# Stairs
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*0,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*1,bar_sz_b//2+bar_sz_b//12*2),Step_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*1,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*2,bar_sz_b//2+bar_sz_b//12*2),Step_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*3,bar_sz_b//2+bar_sz_b//12*2),Step_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*3,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*4,bar_sz_b//2+bar_sz_b//12*2),Step_10,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*4,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*5,bar_sz_b//2+bar_sz_b//12*2),Step_20,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*5,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*6,bar_sz_b//2+bar_sz_b//12*2),Step_30,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*6              , bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*6+bar_sz_e//2*1, bar_sz_b//2+bar_sz_b//12*2),Step_40,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*6+bar_sz_e//2*1, bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*6+bar_sz_e//2*2, bar_sz_b//2+bar_sz_b//12*2),Step_50,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*6+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*7+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12*2),Step_60,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*7+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*8+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12*2),Step_70,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*8+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*9+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12*2),Step_80,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*9+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*10+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12*2),Step_90,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*10+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*11+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12*2),Step_100,-1)
cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*11+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_c+bar_sz_d//2*12+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12*2),Step_100,-1)

cv2.rectangle(imgbuffer,(bar_sz_c+bar_sz_d//2*12+bar_sz_e//2*2,bar_sz_b//2+bar_sz_b//12),(bar_sz_a,bar_sz_b//2+bar_sz_b//12*2),White_58,-1)

cv2.rectangle(imgbuffer,(0,bar_sz_b//2+bar_sz_b//12*2),(bar_sz_c,bar_sz_b//2+bar_sz_b//12*3),Black_0,-1)

# Ramp
grad_s = np.full(bar_grad_b, ramp_low)
grad_e = np.full(bar_grad_d, ramp_high)
grad = np.linspace(ramp_low, ramp_high, bar_grad_c)
gradient = np.concatenate((grad_s, grad, grad_e))
ramp_row = np.stack([gradient, gradient, gradient], axis=1)
ramp_rect = np.tile(ramp_row, (bar_sz_b//12, 1, 1)).astype(np.uint16) # Tile the gradient row vertically to fill the entire rectangle height
x1, y1 = (bar_sz_c,bar_sz_b//2+bar_sz_b//12*2)
x2, y2 = (bar_sz_a,bar_sz_b//2+bar_sz_b//12*3)
imgbuffer[y1:y2, x1:x2] = ramp_rect

# BT.709 bars + black signals
cv2.rectangle(imgbuffer,(0            ,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*1,bar_sz_b),BT709_58_Yellow,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*1,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*2,bar_sz_b),BT709_58_Cyan,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*2,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3,bar_sz_b),BT709_58_Green,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f,bar_sz_b),Black_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g,bar_sz_b),Black_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g+bar_sz_h,bar_sz_b),Black_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g+bar_sz_h,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g*2+bar_sz_h,bar_sz_b),Black_2,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g*2+bar_sz_h,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g*2+bar_sz_h*2,bar_sz_b),Black_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g*2+bar_sz_h*2,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2,bar_sz_b),Black_4,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i,bar_sz_b),Black_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j,bar_sz_b),White_58,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j+bar_sz_k,bar_sz_b),Black_0,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*3+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j+bar_sz_k,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*4+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j+bar_sz_k,bar_sz_b),BT709_58_Magenta,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*4+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j+bar_sz_k,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*5+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j+bar_sz_k,bar_sz_b),BT709_58_Red,-1)
cv2.rectangle(imgbuffer,(bar_sz_c//3*5+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j+bar_sz_k,bar_sz_b//2+bar_sz_b//12*3),(bar_sz_c//3*6+bar_sz_f+bar_sz_g*3+bar_sz_h*2+bar_sz_i+bar_sz_j+bar_sz_k,bar_sz_b),BT709_58_Blue,-1)

# Calculate timecode
def ts_to_tc(timestamp_ns, fps):
    total_seconds = timestamp_ns / 1000000000 # Convert nanoseconds to seconds
    seconds_in_day = total_seconds % 86400 # Get seconds within the current day (modulo 86400 seconds per day)
    hours = int(seconds_in_day // 3600)  # Will be 0-23
    remaining = seconds_in_day % 3600
    minutes = int(remaining // 60)
    seconds = remaining % 60
    # Calculate frame number from fractional seconds
    whole_seconds = int(seconds)
    fractional_seconds = seconds - int(seconds)
    frames = int(fractional_seconds * fps)
    # Format as timecode
    tc = f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}:{frames:02d}"
    return tc

# no need to regenerate static image every time
imgbuffer = imgbuffer.astype(np.float64)
imgbuffer = (imgbuffer - np.min(imgbuffer))/np.ptp(imgbuffer)
imgbuffer = (imgbuffer * 65535.0).astype(np.uint16) # back to 16-bit

counter = 0

# imgbuffer = cv2.cvtColor(imgbuffer, cv2.COLOR_RGB2BGR)
# cv2.imwrite(output_folder + f'colorbars{counter:03d}.tiff', imgbuffer)

start_timestamp = time.time_ns() # same as [FrameWallClock]
tc = ts_to_tc(start_timestamp, fps)

process = (
    ffmpeg
        .input('pipe:', format = 'rawvideo', pix_fmt = 'rgb48', s = '{}x{}'.format(bar_sz_a, bar_sz_b))
        .output(output_file, pix_fmt = 'yuv422p10le', vcodec = 'prores_ks', vprofile = '3', r = fps, timecode = tc)
        .overwrite_output()
        .run_async(pipe_stdin = True)
)

framedurr = 1000000000 // fps
for x in range(0, duration_sec * fps):
    frame = cv2.putText(imgbuffer.copy(), ts_to_tc(start_timestamp + x*framedurr, fps), (bar_sz_d//2,bar_sz_b//2), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 1)
    process.stdin.write(frame.tobytes())

# Finish encoding and save the file
process.stdin.close()
process.wait()