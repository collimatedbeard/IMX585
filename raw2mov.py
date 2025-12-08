import numpy as np
# import cv2
import json
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from colour_demosaicing import *
import ffmpeg

input_video_file = "../CinePi_Video/test.raw"
input_metadata_file = input_video_file.replace('.raw', '.txt')
output_file = input_video_file.replace('.raw', '.mov')
height = 2180
stride = 7744
width = 3872
imgsize = height*stride
out_height = 3840
out_width = 2160
nr_threads = 48 # Tweak for your CPU and ammount of RAM

def ProcessFrame(counter, metadata, chunk):
    print(f"Processing frame: {counter}\n")
    imgbuffer = np.frombuffer(chunk, dtype=np.uint16).copy()
    imgbuffer = np.reshape(imgbuffer, (height, width))
    # Crop to final dimension, disregard blank pixels to fill the stride - img[y:y+h, x:x+w]
    imgbuffer = imgbuffer[0:out_width, 0:out_height]
    imgbuffer = imgbuffer.astype(np.float64)

    # Black level subtraction
    imgbuffer[0::2, 0::2] -= metadata["SensorBlackLevels"][0] # Red
    imgbuffer[0::2, 1::2] -= metadata["SensorBlackLevels"][1] # Green1
    imgbuffer[1::2, 0::2] -= metadata["SensorBlackLevels"][2] # Green2
    imgbuffer[1::2, 1::2] -= metadata["SensorBlackLevels"][3] # Blue

    # White balance
    imgbuffer[0::2, 0::2] *= metadata["ColourGains"][0]  # Red multiplier
    imgbuffer[1::2, 1::2] *= metadata["ColourGains"][1]  # Blue multiplier

    # After substracting black levels, some noise might go below zero. This needs to be clipped.
    imgbuffer = np.clip(imgbuffer, 0, None)
    # Scale values to range [0, 1]
    imgbuffer = imgbuffer / 65535.0
    # imgbuffer = (imgbuffer - np.min(imgbuffer))/np.ptp(imgbuffer)

    # DeBayer the image
    rgb_buffer = demosaicing_CFA_Bayer_DDFAPD(imgbuffer, 'RGGB', refining_step=True)

    rgb_buffer = np.clip(rgb_buffer, 0, 1)
    # rgb_buffer = (rgb_buffer - np.min(rgb_buffer))/np.ptp(rgb_buffer) # Scale values to range [0, 1]
    rgb_buffer = (rgb_buffer * 65535.0).astype(np.uint16) # back to 16-bit

    return rgb_buffer

def read_and_process_streaming_with_counter(input_video_file, fps, metadata, chunksize, max_workers=4, buffer_size=100):

    process = (
        ffmpeg
            .input('pipe:', format = 'rawvideo', pix_fmt = 'rgb48', s = '{}x{}'.format(out_height, out_width))
            .output(output_file, pix_fmt = 'yuv422p10le', vcodec = 'prores_ks', vprofile = '3', r = fps)
            .overwrite_output()
            .run_async(pipe_stdin = True)
    )
    counter = 0

    with open(input_video_file, "rb") as rawfile:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = deque()
            
            # Read and submit chunks
            for chunk in iter(lambda: rawfile.read(chunksize), b''):
                # Submit chunk for processing with counter
                future = executor.submit(ProcessFrame, counter, metadata[counter], chunk)
                futures.append((future, counter))
                counter += 1
                
                # Limit buffer size to control memory usage
                if len(futures) >= buffer_size:
                    # Wait for oldest future to complete
                    completed_future, chunk_counter = futures.popleft()
                    try:
                        result = completed_future.result()
                        # Push the frame to FFMpeg
                        process.stdin.write(result.tobytes())
                        print(f"Completed processing frame {chunk_counter}\n")
                        # Handle result
                    except Exception as exc:
                        print(f'Frame {chunk_counter} processing failed: {exc}\n')
            
            # Process remaining futures
            while futures:
                completed_future, chunk_counter = futures.popleft()
                try:
                    result = completed_future.result()
                    # Push the frame to FFMpeg
                    process.stdin.write(result.tobytes())
                    print(f"Completed processing frame {chunk_counter}\n")
                except Exception as exc:
                    print(f'Frame {chunk_counter} processing failed: {exc}\n')

        # Finish encoding and save the file
        process.stdin.close()
        process.wait()


def main():
    # Load metadata file so we can get WB and CCM values
    with open(input_metadata_file) as metadata_file:
        metadata = json.load(metadata_file)

    timestamps = np.array([item['FrameWallClock'] for item in metadata if 'FrameWallClock' in item])
    durations = np.diff(timestamps)
    frametime = np.mean(durations)
    frametime /= 1000000
    fps = 1 / frametime

    # run in parallel
    read_and_process_streaming_with_counter(input_video_file, fps, metadata, imgsize, nr_threads, 100)
    # print(frametime)
    # print(np.std(durations, ddof=0))
    print(f"Calculated fps: {fps:.2f}")

if __name__ == "__main__":
    main()
