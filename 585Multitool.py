import dearpygui.dearpygui as dpg
import numpy as np
import cv2
import os
import random
import glob
import json
from fabric import Connection

dpg.create_context()

height = 2180
stride = 7744
resize_fac = 1
black_level = 3200
frameNrForPreview =  7 # Warning: if you are changing this value, make sure there are enough frames in recording

dataraw = np.zeros([height*stride], dtype=np.uint16)
x_data = np.zeros([65535], dtype=np.float32)
hist = np.zeros([65535], dtype=np.float32)

def takePhotoBtn_cb():
    dpg.set_value("tbLog", "Running...\n")
    execCommand(buildCmdLines(singleFrame=True))

def startVideoBtn_cb():
    dpg.set_value("tbLog", "Please refrain from clicking this button.\n")
    # execCommand(buildCmdLines(singleFrame=False))

def dncBtn_cb():
    raw_files = glob.glob("*.raw")
    if raw_files:
        random_file = random.choice(raw_files)
        # print(random_file)
        with open(random_file, "rb") as rawfile:
            global dataraw
            dataraw = np.fromfile(rawfile, np.uint16)
            processImage()

def setWBPickBtn_cb():
    pass

def setWBGrayWrldBtn_cb():
    pass

def setWBFileBtn_cb():
    with open(f'test.txt', "rb") as metadatafile:
        metadata = json.load(metadatafile) 
        dpg.set_value("inGainR", metadata[frameNrForPreview]["ColourGains"][0])
        dpg.set_value("inGainB", metadata[frameNrForPreview]["ColourGains"][1])
        correction_matrix = np.asarray(metadata[frameNrForPreview]["ColourCorrectionMatrix"])
        correction_matrix = np.reshape(correction_matrix, (3, 3))
        for i in range(3):
            for j in range(3):
                dpg.set_value(f"ccmC{i}R{j}", correction_matrix[i,j])

def readMetadataExpBtn_cb():
    with open(f'test.txt', "rb") as metadatafile:
        metadata = json.load(metadatafile) 
        dpg.set_value("inExposure", metadata[frameNrForPreview]["ExposureTime"])

        
def execCommand(cmdString):
    c = Connection('cinepi')
    for cmd in cmdString:
        result = c.run(cmd, hide=True, warn=True, pty=True)
        dpg.set_value("tbLog", dpg.get_value("tbLog")
                            + f'\n{result.command} {("FAIL", "OK")[result.ok]}\n'
                            + result.stdout)
    c.get(f'/tmp/test_{frameNrForPreview:05d}.raw', preserve_mode=False)
    c.get(f'/tmp/test.txt', preserve_mode=False)
    with open(f'test_{frameNrForPreview:05d}.raw', "rb") as rawfile:
        global dataraw
        dataraw = np.fromfile(rawfile, np.uint16)
        processImage()


def buildCmdLines(singleFrame = True):
    cmds = []
    if dpg.get_value("inExposure") == 0:
        exposure = "" # Auto
    else:
        exposure = f"--shutter {dpg.get_value("inExposure")}"
    
    fps = dpg.get_value("inFPS")
    if dpg.get_value("cbSinleFile"):
        singlStr = "--segment 1"
    else:
        singlStr = ""
    if dpg.get_value("cSensMode") == "10 bit":
        bittness = 10
        hdr = False
    elif dpg.get_value("cSensMode") == "12 bit":
        bittness = 12
        hdr = False
    elif dpg.get_value("cSensMode") == "12 bit HDR":
        bittness = 12
        hdr = True
    elif dpg.get_value("cSensMode") == "16 bit HDR":
        bittness = 16
        hdr = True
    
    sensGain = dpg.get_value("inGainSens")

    if dpg.get_value("inWBSetting") == "Fixed 1,1":
        awbgains = "--awbgain 1.0,1.0"
    elif dpg.get_value("inWBSetting") == "WB from below":
        r = dpg.get_value("inGainR")   # red channel
        b = dpg.get_value("inGainB")   # blue channel
        awbgains = f"--awbgain {r:1.6f},{b:1.6f}"
    else:
        awbgains = "" # Auto

    if hdr:
        lg = dpg.get_value("inLG")
        hg = dpg.get_value("inHG")
        if dpg.get_value("inBlending") == "HG 3/4,   LG 1/4":
            blending = 1
        elif dpg.get_value("inBlending") == "HG 7/8,   LG 1/8":
            blending = 3
        elif dpg.get_value("inBlending") == "HG 15/16, LG 1/16":
            blending = 4
        elif dpg.get_value("inBlending") == "HG 1/16,  LG 15/16":
            blending = 6
        elif dpg.get_value("inBlending") == "HG 1/8,   LG 7/8":
            blending = 7
        elif dpg.get_value("inBlending") == "HG 1/4,   LG 3/4":
            blending = 8
        else:
            blending = 0
        if dpg.get_value("inGainExp") == "6 db":
            gain_add = 1
        elif dpg.get_value("inGainExp") == "12 db":
            gain_add = 2
        elif dpg.get_value("inGainExp") == "18 db":
            gain_add = 3
        elif dpg.get_value("inGainExp") == "24 db":
            gain_add = 4
        elif dpg.get_value("inGainExp") == "29.1 db":
            gain_add = 5
        else:
            gain_add = 0
        cmds.append(f'v4l2-ctl -d /dev/v4l-subdev2 --set-ctrl=wide_dynamic_range=1')
        cmds.append(f'v4l2-ctl -d /dev/v4l-subdev2 --subset hdr_data_selection_threshold,0,1  --set-ctrl hdr_data_selection_threshold={lg}')
        cmds.append(f'v4l2-ctl -d /dev/v4l-subdev2 --subset hdr_data_selection_threshold,1,1  --set-ctrl hdr_data_selection_threshold={hg}')
        cmds.append(f'v4l2-ctl -d /dev/v4l-subdev2 --set-ctrl=hdr_data_blending_mode={blending}')
        cmds.append(f'v4l2-ctl -d /dev/v4l-subdev2 --set-ctrl=hdr_gain_adder_db={gain_add}')
    else:
        cmds.append(f'v4l2-ctl -d /dev/v4l-subdev2 --set-ctrl=wide_dynamic_range=0')

    # show V4L settings at the end
    cmds.append(f'v4l2-ctl -d /dev/v4l-subdev2 -l') 

    if singleFrame:
        cmds.append(f'rpicam-raw -n --segment 1 -o /tmp/test_%05d.raw -f -t 3000 --mode "3856:2180:{bittness}:U" --framerate 5 --denoise off -v 3 --gain {sensGain:.1f} {awbgains} {exposure} --metadata /tmp/test.txt')
    else: # Video settings
        cmds.append(f'rpicam-raw -n {singlStr} -o /mnt/test_%05d.raw -f -t 3000 --mode "3856:2180:{bittness}:U" --framerate {fps} --denoise off -v 3 --gain {sensGain:.1f} {awbgains} {exposure} --metadata /mnt/test.txt')
    # cmds.append('ls -l /tmp')
    return cmds

def stretchCB_cb():
    if dpg.get_value("inStrTo") == "to 16-bit max":
        dpg.set_axis_limits("x_axis", 0, 65536)
    elif dpg.get_value("inStrTo") == "to 12-bit max":
        dpg.set_axis_limits("x_axis", 0, 4096)
    processImage()

def processImage():
    global dataraw

    black_level = dpg.get_value("inBlackLvl")

    if "HDR" in dpg.get_value("cSensMode"):
        setLG = (black_level, 0)[dpg.get_value("cbBlackLvl")] + dpg.get_value("inLG")
        setHG = (black_level, 0)[dpg.get_value("cbBlackLvl")] + dpg.get_value("inHG")
        # if dpg.get_value("cb12Bit"):
        #     setLG = setLG >> 4
        #     setHG = setHG >> 4
    else:
        setHG = 0
        setLG = 0

    if dpg.get_value("cbBlackLvl"):
        dataimg = np.clip(dataraw, black_level, None).copy()
        dataimg -= black_level
    else:
        dataimg = dataraw.copy()

    if dpg.get_value("cb12Bit"):
        dataimg = np.right_shift(dataimg, np.uint16(4)) # back to 12 bit

    width = dataimg.size // height
    dataimg = np.reshape(dataimg, (height, width))

    # Crop to final dimension, disregard blank pixels used to fill the stride - img[y:y+h, x:x+w]
    dataimg = dataimg[20:2160, 0:3840]

    # Filter if appropriate RB was selected
    if dpg.get_value("inShowParts") == "Below LG":
        dataimg[dataimg > setHG] = 0
    elif dpg.get_value("inShowParts") == "Between":
        dataimg[(dataimg >= setLG) | (dataimg <= setHG)] = 0
    elif dpg.get_value("inShowParts") == "Above HG":
        dataimg[dataimg < setLG] = 0
    
    # Create pic texture
    if dpg.get_value("cbDebayer"):
        imgbuffer = cv2.cvtColor(dataimg, cv2.COLOR_BAYER_RGGB2RGB)
    else:
        imgbuffer = cv2.cvtColor(dataimg, cv2.COLOR_GRAY2RGB)
    imgbuffer = cv2.resize(imgbuffer, [3840//resize_fac, 2160//resize_fac])

    if dpg.get_value("inStrFrom") == "from 0":
        minval = 0
    else: # auto
        minval = np.min(imgbuffer)
    if dpg.get_value("inStrTo") == "to 16-bit max":
        maxval = 0xffff
    elif dpg.get_value("inStrTo") == "to 12-bit max":
        maxval = 0x0fff
    else: # auto
        maxval = np.max(imgbuffer)

    imgbuffer = np.clip(imgbuffer, minval, maxval)
    imgbuffer = (imgbuffer - minval) / maxval # Scale values to 0..1 which is required for DerPyGUI texture

    if dpg.get_value("cbProcessColor"):
        # WB
        imgbuffer[:,:, 0] *= dpg.get_value("inGainR") # red channel
        imgbuffer[:,:, 2] *= dpg.get_value("inGainB") # blue channel
        # imgbuffer = imgbuffer / np.ptp(imgbuffer) # scale again
        # CCM
        correction_matrix = np.array([ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for i in range(3):
            for j in range(3):
                correction_matrix[i,j] = dpg.get_value(f"ccmC{i}R{j}")
        xyzbuffer = cv2.cvtColor(imgbuffer.astype('float32'), cv2.COLOR_RGB2XYZ)
        h, w, ch = xyzbuffer.shape # save image dimensions
        xyz_flat = xyzbuffer.reshape(-1, 3) # we need "flat" array for the next operation
        corrected_xyz_flat = np.dot(xyz_flat, correction_matrix.T) # apply color correction matrix
        corrected_xyz = corrected_xyz_flat.reshape(h, w, ch).astype('float32') # back to image shape and it needs to be float32 (output is float64)
        imgbuffer = cv2.cvtColor(corrected_xyz, cv2.COLOR_XYZ2RGB)
        # imgbuffer = imgbuffer / np.ptp(imgbuffer) # scale again
        imgbuffer = np.clip(imgbuffer, 0, 1) # Clip values back to valid range 0..1
        
    texture_data = imgbuffer.ravel()
    raw_data = np.array(texture_data, dtype=np.float32)
    dpg.set_value("texture_tag", raw_data)

    # compute and plot the image histograms
    hist = cv2.calcHist([dataimg], [0], None, [65536], [0, 65535])
    # Create x-axis data
    x_data = np.arange(65535, dtype=np.float32)
    dpg.set_value('hist_data', [x_data, hist])
    dpg.set_axis_limits_auto("x_axis")
    dpg.set_axis_limits_auto("y_axis")
    dpg.set_value('drgLG', setLG)
    dpg.set_value('drgHG', setHG)



with dpg.theme(tag="__window_nopad"):
    with dpg.theme_component(dpg.mvAll): # remove padding
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, category=dpg.mvThemeCat_Core)
with dpg.theme(tag="__window_noborder"):
    with dpg.theme_component(dpg.mvAll): # remove padding + border
        dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, category=dpg.mvThemeCat_Core)
        dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0, category=dpg.mvThemeCat_Core)
    with dpg.theme(tag="__button_theme_photo"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (200, 0, 200))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 10)
    with dpg.theme(tag="__button_theme_video"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 0, 0))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 10)
    with dpg.theme(tag="__button_theme_exit"):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (100, 100, 100))
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 10, 10)
    with dpg.theme(tag="__series_theme_shaded"):
        with dpg.theme_component(0):
            dpg.add_theme_style(dpg.mvPlotStyleVar_FillAlpha, category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Line, (200, 200, 200), category=dpg.mvThemeCat_Plots)
            dpg.add_theme_color(dpg.mvPlotCol_Fill, (150, 150, 150, 100), category=dpg.mvThemeCat_Plots)

dpg.configure_app(init_file="585mtool.ini")

with dpg.window(label="Image", tag="Primary Window"):
    raw_data = np.zeros([(2160//resize_fac) * (3840//resize_fac) * 3], dtype=np.float32)
    with dpg.texture_registry(show = False):
        dpg.add_raw_texture(width=3840//resize_fac, height=2160//resize_fac, default_value=raw_data, tag="texture_tag", format=dpg.mvFormat_Float_rgb)
    dpg.set_value("texture_tag", raw_data)
    dpg.add_image("texture_tag", tag="image_tag")


with dpg.window(label = "Controls", height = 700, width = 400, tag="Controls"):
    with dpg.menu_bar():
        with dpg.menu(label="File"):
            dpg.add_menu_item(label="Save screen")
            dpg.add_menu_item(label="Export DNG")
            dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())

        with dpg.menu(label="Settings"):
            dpg.add_menu_item(label="Save window position", callback=lambda: dpg.save_init_file("585mtool.ini"))

        with dpg.menu(label="Help"):
            dpg.add_menu_item(label="About")

    with dpg.table(header_row=False):
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()
        dpg.add_table_column()
        with dpg.table_row():
            dpg.add_button(label="Take\nstill", width=-1, callback=takePhotoBtn_cb)
            dpg.bind_item_theme(dpg.last_item(), "__button_theme_photo")
            dpg.add_button(label="Start\nvideo", width=-1, callback=startVideoBtn_cb)
            dpg.bind_item_theme(dpg.last_item(), "__button_theme_video")
            dpg.add_button(label="Don't\nclick", width=-1, callback=dncBtn_cb)
            dpg.bind_item_theme(dpg.last_item(), "__button_theme_exit")
            dpg.add_button(label="\nExit", width=-1, callback=lambda: dpg.stop_dearpygui())
            dpg.bind_item_theme(dpg.last_item(), "__button_theme_exit")     

    dpg.add_separator(label="Sensor settings")
    dpg.add_combo(("10 bit", "12 bit", "12 bit HDR", "16 bit HDR"), default_value="12 bit", tag="cSensMode", label="Sensor Mode", width=100)
    with dpg.group(horizontal=True):
        dpg.add_combo(("0 db", "6 db", "12 db", "18 db", "24 db", "29.1 db"), label="Exp gain", tag="inGainExp", default_value="0 db", fit_width=True)
        dpg.add_input_double(label="Sensor gain", tag="inGainSens", default_value=1.0, format="%.1f", width=100)
    dpg.add_slider_int(label = "Low Gain (LG)", tag="inLG", default_value=3000, min_value=0, max_value=0x0fff, format="%d", width=200)
    dpg.add_slider_int(label = "High Gain (HG)", tag="inHG", default_value=2000, min_value=0, max_value=0x0fff, format="%d", width=200)
    dpg.add_combo((
        "HG 1/2,   LG 1/2",
        "HG 3/4,   LG 1/4",
        "HG 7/8,   LG 1/8",
        "HG 15/16, LG 1/16",
        # "HG 1/2,(2)LG 1/2",
        "HG 1/16,  LG 15/16",
        "HG 1/8,   LG 7/8",
        "HG 1/4,   LG 3/4"), label="Blending", tag="inBlending", default_value="HG 1/2,   LG 1/2", width=150)
    with dpg.group(horizontal=True):
        dpg.add_slider_int(tag="inExposure", label = "Exposure time", min_value=0, default_value=5000, max_value=100000, format="%d us", width=200)
        dpg.add_button(label="M", callback=readMetadataExpBtn_cb)
    with dpg.group(horizontal=True):
        dpg.add_text("WB gains:")
        dpg.add_radio_button(("Auto", "Fixed 1,1", "WB from below"), tag="inWBSetting", default_value="Auto", horizontal=True)


    dpg.add_separator(label="Video options")
    with dpg.group(horizontal=True):
        dpg.add_input_int(tag="inFPS", label = "FPS", default_value=25, min_value=1, max_value=50, width=70)
        # f"{tS//3600:02d}:{tS%3600//60:02d}:{tS%60:02d}"
        dpg.add_drag_int(tag="inVideoLength", label = "Length", min_value=1, default_value=10, max_value=600, format="%02d sec", width=100)
    dpg.add_checkbox(label="Single file", tag="cbSinleFile")

    dpg.add_separator(label="Raw manipulation")
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label="Black Level", tag="cbBlackLvl", callback=lambda: processImage())
        dpg.add_input_int(label="Value", tag="inBlackLvl", default_value=3200, max_value=4000, width=150)
    with dpg.group(horizontal=True):
        dpg.add_checkbox(label="De-Bayer", tag="cbDebayer", default_value=True, callback=lambda: processImage())
        dpg.add_checkbox(label="12-bit un-shift", tag="cb12Bit", callback=lambda: processImage())
    with dpg.group(horizontal=True):
        dpg.add_text("Stretch bottom ")
        dpg.add_combo(("from 0", "auto"), default_value="from 0", fit_width=True, tag="inStrFrom", callback=lambda: stretchCB_cb())
        dpg.add_text(", top ")
        dpg.add_combo(("to 16-bit max", "to 12-bit max", "auto"), default_value="auto", fit_width=True, tag="inStrTo", callback=lambda: stretchCB_cb())
    with dpg.group(horizontal=True):
        dpg.add_text("Show:")
        dpg.add_radio_button(("All", "Below LG", "Between", "Above HG"), tag="inShowParts", default_value="All", horizontal=True, callback=lambda: processImage())
    dpg.add_separator(label="Color")
    dpg.add_checkbox(label="Fine color processing", tag="cbProcessColor", callback=lambda: processImage())
    with dpg.group(horizontal=True):
        with dpg.group(horizontal=False):
            dpg.add_text("WB:")
            dpg.add_button(label="R", callback=lambda: [dpg.set_value(f"inGain{i}", 1.0) for i in ["R","G","B"]]) # Reset values
        with dpg.theme(tag="__slider_theme_R"):
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (100, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,     (150, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  (250, 0, 0))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (200, 0, 0))
        dpg.add_slider_float(tag="inGainR", default_value=1, vertical=True, max_value=3.0, height=70)
        dpg.bind_item_theme(dpg.last_item(), "__slider_theme_R")
        with dpg.theme(tag="__slider_theme_G"):
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (0, 100, 0))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,     (0, 150, 0))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  (0, 250, 0))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (0, 200, 0))
        dpg.add_slider_float(tag="inGainG", default_value=1, vertical=True, max_value=3.0, height=70)
        dpg.bind_item_theme(dpg.last_item(), "__slider_theme_G")
        with dpg.theme(tag="__slider_theme_B"):
            with dpg.theme_component(0):
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        (0, 0, 100))
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,     (0, 0, 150))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  (0, 0, 250))
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (0, 0, 200))
        dpg.add_slider_float(tag="inGainB", default_value=1, vertical=True, max_value=3.0, height=70)
        dpg.bind_item_theme(dpg.last_item(), "__slider_theme_B")
        with dpg.group(horizontal=False):
            dpg.add_button(label="Pick WB point from image", callback=setWBPickBtn_cb)
            dpg.add_button(label="Calculate from grayworld", callback=setWBGrayWrldBtn_cb)
            dpg.add_button(label="Read from metadata file", callback=setWBFileBtn_cb)
    with dpg.group(horizontal=True):
        with dpg.group(horizontal=False):
            dpg.add_text("CCM:")
            dpg.add_button(label="R", callback=lambda: [dpg.set_value(f"ccmC{i}R{j}", 1.0 if i==j else 0.0) for j in range(3) for i in range(3)]) # Reset values
        with dpg.table(header_row=False, row_background=False, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
            dpg.add_table_column()
            dpg.add_table_column()
            dpg.add_table_column()
            for c in range(3):
                with dpg.table_row():
                    for r in range(3):
                        dpg.add_drag_float(tag=f"ccmC{c}R{r}", width=-1, default_value=1.0 if c==r else 0.0, format="%.6f")


with dpg.window(label="Histogram", height=400, width=750, no_scrollbar=True, tag="Histogram"):
    with dpg.plot(label="Image Histogram", height=-1, width=-1, tag="histogram_plot"):
        # Configure plot axes
        dpg.add_plot_axis(dpg.mvXAxis, label="Pixel Value", tag="x_axis")
        dpg.set_axis_ticks(dpg.last_item(), (("0", 0), ("8", 255), ("10", 1023), ("11", 2047), ("12", 4095), ("13", 8191), ("14", 16383), ("15", 32767), ("16", 65565)))
        dpg.add_plot_axis(dpg.mvYAxis, label="Frequency", tag="y_axis")
        dpg.add_drag_line(label="Low Gain", tag="drgLG", color=[200, 0, 200, 255], default_value=0, no_fit=True)
        dpg.add_drag_line(label="High Gain", tag="drgHG", color=[200, 50, 150, 255], default_value=0, no_fit=True)
        dpg.set_axis_limits("x_axis", 0, 65535)
        dpg.set_axis_limits("y_axis", 0, 10000)
        dpg.add_line_series(x_data, hist, label="Histogram", tag="hist_data", parent="x_axis", shaded=True)
        dpg.bind_item_theme("hist_data", "__series_theme_shaded")

with dpg.window(label = "Log", height=500, width=750, tag="Log"):
    dpg.add_input_text(multiline=True, height=-1, width=-1, tag="tbLog")


dpg.bind_item_theme("Histogram", "__window_nopad")
dpg.bind_item_theme("Primary Window", "__window_noborder")

dpg.create_viewport(title='IMX585 Multitool', width = 1920, height = 1180)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.toggle_viewport_fullscreen()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()