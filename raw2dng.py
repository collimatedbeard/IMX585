import numpy as np
from fractions import Fraction
import json
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
from pathlib import Path

input_video_file = "../CinePi_Video/test.raw"
input_metadata_file = input_video_file.replace('.raw', '.txt')

height = 2180
stride = 7744
width = 3872
imgsize = height*stride
max_denominator = 1000000

def ProcessFrame(counter, metadata, chunk, fps):
    print(f"Processing frame: {counter}\n")
    imgbuffer = np.frombuffer(chunk, dtype=np.uint16).copy()
    imgbuffer = np.reshape(imgbuffer, (height, width))
    h, w = imgbuffer.shape
    numPixels = w*h
    rawImage = np.reshape(imgbuffer, (1, numPixels))

    a_gain = metadata["AnalogueGain"]
    d_gain = metadata["DigitalGain"]
    iso = int(a_gain * d_gain * 100)

    # CCM
    correction_matrixT = np.array([[  1.844691, -0.496813, -0.371959 ], [ -0.233085,  1.459965, -0.247254 ], [  0.121883, -0.103817,  0.355234 ]])
    ccmT = np.ravel(correction_matrixT)
    ccmT = [Fraction(ccmT[i]).limit_denominator(100000).as_integer_ratio() for i in range(len(ccmT))]
    correction_matrixD = np.array([[  1.196056, -0.231126, -0.115278 ], [ -0.324590,  1.370661, -0.061927 ], [  0.067607,  0.149448,  0.569398 ]])
    ccmD = np.ravel(correction_matrixD)
    ccmD = [Fraction(ccmD[i]).limit_denominator(100000).as_integer_ratio() for i in range(len(ccmD))]
    # WB
    red_gain = metadata["ColourGains"][0]
    blue_gain = metadata["ColourGains"][1]
    asn = [Fraction(red_gain).limit_denominator(max_denominator).as_integer_ratio()  # red channel
        , Fraction(1).limit_denominator(max_denominator).as_integer_ratio()          # green is always 1
        , Fraction(blue_gain).limit_denominator(max_denominator).as_integer_ratio()] # blue channel
    # DNG tags
    t = DNGTags()
    t.set(Tag.ImageWidth, w)
    t.set(Tag.ImageLength, h)
    # crop here instead image data
    t.set(Tag.DefaultCropOrigin, [0, 20])
    t.set(Tag.DefaultCropSize, [3840, 2160])
    t.set(Tag.PlanarConfiguration, 1) # pcInterleaved = 1, pcPlanar	= 2
    t.set(Tag.Orientation, Orientation.Horizontal)
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, 16)
    t.set(Tag.CFARepeatPatternDim, [2,2])
    t.set(Tag.CFAPattern, CFAPattern.RGGB)
    t.set(Tag.BlackLevel, metadata["SensorBlackLevels"][0]) # Do not put all four values for Black Levels or Resolve will consider files invalid
    t.set(Tag.WhiteLevel, ((1 << 16) -1) )
    t.set(Tag.ColorMatrix1, ccmT)
    t.set(Tag.ColorMatrix2, ccmD)
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Tungsten_Incandescent)
    t.set(Tag.CalibrationIlluminant2, CalibrationIlluminant.D65)
    t.set(Tag.AsShotNeutral, asn)
    # ISO
    t.set(Tag.SensitivityType, 3)
    t.set(Tag.PhotographicSensitivity, iso)
    # t.set(Tag.BaselineExposure, [[-150,100]])
    t.set(Tag.Make, "Sony")
    t.set(Tag.Model, "IMX585")
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_1)
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)
    # CinemaDNG tags
    t.set(Tag.FrameRate, [Fraction(fps).limit_denominator(max_denominator).as_integer_ratio()])

    # save to dng file
    r = RAW2DNG()
    r.options(t, path='')
    r.convert(rawImage, filename=f"{Path(input_video_file).stem}_{counter:05d}")


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