import numpy as np
from fractions import Fraction
import json
from pathlib import Path
import cv2 as cv
import matplotlib.pyplot as plt


input_video_file = "../CinePi_Video/lightbulbHDR_02.raw"
input_metadata_file = input_video_file.replace('.raw', '.txt')

height = 2180
stride = 7744
width = 3872
out_height = 3840
out_width = 2160
imgsize = height*stride
max_denominator = 1000000

def ProcessFrame(counter, metadata, chunk, fps):
    print(f"Processing frame: {counter}\n")

    imgbuffer = np.frombuffer(chunk, dtype=np.uint16).copy()
    imgbuffer = np.reshape(imgbuffer, (height, width))
    # Crop to final dimension, disregard blank pixels to fill the stride - img[y:y+h, x:x+w]
    imgbuffer = imgbuffer[0:out_width, 0:out_height]
    # imgbuffer = imgbuffer.astype(np.float64)

    # Black level subtraction
    # After substracting black levels, some noise might go below zero. This needs to be clipped.
    imgbuffer = np.clip(imgbuffer, metadata["SensorBlackLevels"][0], None)
    imgbuffer[0::2, 0::2] -= metadata["SensorBlackLevels"][0] # Red
    imgbuffer[0::2, 1::2] -= metadata["SensorBlackLevels"][1] # Green1
    imgbuffer[1::2, 0::2] -= metadata["SensorBlackLevels"][2] # Green2
    imgbuffer[1::2, 1::2] -= metadata["SensorBlackLevels"][3] # Blue

    a_gain = metadata["AnalogueGain"]
    d_gain = metadata["DigitalGain"]
    iso = int(a_gain * d_gain * 100)

    # save graph to image file
    hist = cv.calcHist([imgbuffer], [0], None, [65565], [0, 65535])

    fig = plt.figure(figsize = (12, 12))
    fig.patch.set_alpha(0.0)
    ax = plt.gca()
    ax.patch.set_alpha(0.0)
    ax.spines['bottom'].set_color('gray')
    ax.spines['left'].set_color('gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='gray')
    plt.plot(hist, linewidth = 0.5, color = 'white')
    plt.ylim(0, 1000)
    plt.xlim(0, 65536 - metadata["SensorBlackLevels"][0])
    plt.grid(True, alpha = 0.3, linewidth = 0.1, color = 'gray')
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'hist_{counter:05d}.png', transparent = True, dpi = 150)
    plt.close()

def main():
    counter = 0

    # Load metadata file so we can get WB and CCM values
    with open(input_metadata_file) as metadata_file:
        metadata = json.load(metadata_file)

    timestamps = np.array([item['FrameWallClock'] for item in metadata if 'FrameWallClock' in item])
    durations = np.diff(timestamps)
    frametime = np.mean(durations)
    frametime /= 1000000
    fps = 1 / frametime

    with open(input_video_file, "rb") as rawfile:
        for chunk in iter(lambda: rawfile.read(imgsize), b''):
            ProcessFrame(counter, metadata[counter], chunk, fps)
            counter += 1

    # print(frametime)
    # print(np.std(durations, ddof=0))
    print(f"Calculated fps: {fps:.2f}")


if __name__ == "__main__":
    main()