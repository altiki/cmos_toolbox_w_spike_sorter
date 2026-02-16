import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import datetime


def selectElements(array, selection_threshold: int = 40, n_sample: int = 1024):
    # Parameters for electrode selection functions
    voltage_map = array
    width = array.shape[1]
    height = array.shape[0]
    scale_factor = 7
    selection_threshold = selection_threshold
    selection_threshold_default = selection_threshold
    n_sample = n_sample
    color_map = "hot"

    line_thickness = 1
    dot_radius = 2
    alpha = 0.3
    fontSize = 0.5
    windowSize = [height * scale_factor, width * scale_factor]

    # interactive functionality
    clickRegistered = np.zeros(1)==1
    stimElectrodesPixel = []
    stimElectrodes = []
    stimElectrodesAdd = []
    stimElectrodeAddHist = []
    stimElectrodesHist = []
    selected_pixels_add = []
    selected_pixels_remove = []
    selected_electrodes = []
    selected_electrodes_hist = []
    add_remove_hist = []
    selection_threshold_hist = []

    def pixel_to_electrode(selected_pixels, voltage_map, selection_threshold):
        # determine the electrodes corresponding to the marked pixels
        selected_electrodes = []
        selected_electrodes_vertices = np.asarray([[round(coord / scale_factor) for coord in selected_pixel]
                                        for selected_pixel in selected_pixels])
        x_vals = selected_electrodes_vertices[:,0]
        y_vals = selected_electrodes_vertices[:,1]

        x_arange = np.arange(min(x_vals), max(x_vals))
        y_arange = np.arange(min(y_vals), max(y_vals))
        for x in x_arange:
            for y in y_arange:
                if voltage_map[y][x] > selection_threshold:
                    in_poly = pixel_in_selection(x, y, selected_electrodes_vertices)
                    if in_poly:
                        selected_electrodes.append([y, x])
        return selected_electrodes

    def pixel_in_selection(x, y, selected_electrodes_vertices):
        # check if pixel is in selection using the even-odd rule (https://en.wikipedia.org/wiki/Even-odd_rule)
        result = False
        j = len(selected_electrodes_vertices) - 1
        for i in range(len(selected_electrodes_vertices)):
            if (x == selected_electrodes_vertices[i][0]) and (y == selected_electrodes_vertices[i][1]):
                return True
            if ((selected_electrodes_vertices[i][1] > y) != (selected_electrodes_vertices[j][1] > y)):
                slope = (x - selected_electrodes_vertices[i][0]) * (
                            selected_electrodes_vertices[j][1] - selected_electrodes_vertices[i][1]) - (
                                    selected_electrodes_vertices[j][0] - selected_electrodes_vertices[i][0]) * (
                                    y - selected_electrodes_vertices[i][1])
                if slope == 0:
                    return True
                elif (slope < 0) != (selected_electrodes_vertices[j][1] < selected_electrodes_vertices[i][1]):
                    result = not result
            j = i
        return result

    def plot_electrodes(selected_electrodes, map, dot_radius, color=[0, 0, 255, 1], plotRectangleBool = False):
        # plot electrodes
        selected_electrodes_plot = [
            tuple(coords * scale_factor for coords in selectred_electrode[::-1]) for selectred_electrode in
            selected_electrodes]
        for selected_electrode_plot in selected_electrodes_plot:
            if plotRectangleBool:
                cv2.rectangle(map, (selected_electrode_plot[0]-dot_radius,selected_electrode_plot[1]-dot_radius),
                              (selected_electrode_plot[0]+dot_radius,selected_electrode_plot[1]+dot_radius),
                              color=color,thickness=-1)
            else:
                cv2.circle(map, selected_electrode_plot, radius=dot_radius, color=color,
                           thickness=-1)
        return map

    def setFlag():
        clickRegistered[0] = True

    def selection_pixels(event, x, y, flags, param):
        # add pixels to selection polygon
        if event == cv2.EVENT_LBUTTONDOWN:
            selected_pixels_add.append((x, y))
        if event == cv2.EVENT_RBUTTONDOWN:
            selected_pixels_remove.append((x, y))
        if event == cv2.EVENT_MBUTTONDOWN:
            stimElectrodesPixel.append((x,y))
            print("Clicked electrode for stimulation: " + str((round(y/scale_factor)*width+round(x/scale_factor))))
        setFlag()
    im = plt.imshow(voltage_map,
                    cmap=color_map)  # sometimes a specific colormap breaks a voltage map, use a different one if that happen
    plt.close()
    voltage_map_rgb = np.array(im.cmap(im.norm(im.get_array()))[:, :, 0:3])
    voltage_map_rgb = cv2.resize(voltage_map_rgb, (
    scale_factor * voltage_map_rgb.shape[1], scale_factor * voltage_map_rgb.shape[0]))
    initial_map = voltage_map_rgb.copy()
    prev_map = voltage_map_rgb.copy()
    cv2.namedWindow("voltage_map")
    cv2.setMouseCallback("voltage_map", selection_pixels)
    selected_electrodes_boolean = np.zeros((height, width))

    def updatePlot(ogMap):
        newMap = plot_electrodes(selected_electrodes, ogMap, dot_radius)
        return plot_electrodes(stimElectrodes, newMap, dot_radius+1, color=[0, 255,255, 1])
    while True:
        key = cv2.waitKey(1) & 0xFF

        if clickRegistered:

            clickRegistered[0] = False

            if selected_pixels_add:
                voltage_map_rgb_add = initial_map.copy()
                cv2.fillPoly(voltage_map_rgb_add, [np.array(selected_pixels_add)], color=[0, 0, 255, 1])
                voltage_map_rgb_add = cv2.addWeighted(voltage_map_rgb_add, alpha, copy.deepcopy(prev_map), 1 - alpha, 0)

            if selected_pixels_remove:
                voltage_map_rgb_remove = initial_map.copy()
                cv2.fillPoly(voltage_map_rgb_remove, [np.array(selected_pixels_remove)], color=[0, 255, 0, 1])
                voltage_map_rgb_remove = cv2.addWeighted(voltage_map_rgb_remove, alpha, copy.deepcopy(prev_map),
                                                         1 - alpha, 0)
            if stimElectrodesPixel:
                stimSelects = np.asarray([[round(coord / scale_factor) for coord in selected_pixel[::-1]]
                                          for selected_pixel in stimElectrodesPixel])
                stimSelects = np.atleast_2d(stimSelects)
                stimSelects = stimSelects[:,:1] * width + stimSelects[:,1:]
                stimSelects = np.unique(stimSelects, return_counts=True)
                stimElectrodesAdd = []
                for index, counts in enumerate(stimSelects[1]):
                    if counts % 2 == 1:
                        stimElectrodesAdd.append([stimSelects[0][index] // width, stimSelects[0][index] % width])
                voltage_map_rgb_stimEl = plot_electrodes(stimElectrodesAdd, initial_map.copy(), dot_radius+1, color=[0, 255, 255, 1],plotRectangleBool=True)
                voltage_map_rgb_stimEl = cv2.addWeighted(voltage_map_rgb_stimEl, alpha, copy.deepcopy(prev_map),
                                                         1 - alpha, 0)

            if selected_pixels_add and selected_pixels_remove and stimElectrodesPixel:
                voltage_map_rgb_add = cv2.addWeighted(voltage_map_rgb_add, 0.5, voltage_map_rgb_remove, 0.5, 0)
                voltage_map_rgb = cv2.addWeighted(voltage_map_rgb_add, 0.5, voltage_map_rgb_stimEl, 0.5, 0)
            elif selected_pixels_add and selected_pixels_remove:
                voltage_map_rgb = cv2.addWeighted(voltage_map_rgb_add, 0.5, voltage_map_rgb_remove, 0.5, 0)
            elif selected_pixels_remove and stimElectrodesPixel:
                voltage_map_rgb = cv2.addWeighted(voltage_map_rgb_stimEl, 0.5, voltage_map_rgb_remove, 0.5, 0)
            elif selected_pixels_add and stimElectrodesPixel:
                voltage_map_rgb = cv2.addWeighted(voltage_map_rgb_stimEl, 0.5, voltage_map_rgb_add, 0.5, 0)
            elif selected_pixels_add:
                voltage_map_rgb = voltage_map_rgb_add
                voltage_map_rgb_add = copy.deepcopy(initial_map)
            elif stimElectrodesPixel:
                voltage_map_rgb = voltage_map_rgb_stimEl
                voltage_map_rgb_stimEl = copy.deepcopy(initial_map)
            elif selected_pixels_remove:
                voltage_map_rgb = voltage_map_rgb_remove
                voltage_map_rgb_remove = copy.deepcopy(initial_map)
            else:
                voltage_map_rgb_add = copy.deepcopy(initial_map)
                voltage_map_rgb_remove = copy.deepcopy(initial_map)
                voltage_map_rgb_stimEl = copy.deepcopy(initial_map)

            updatePlot(voltage_map_rgb)


        if key == ord("a"):
            # add or subtract electrodes based on drawn polygons
            if selected_pixels_add or selected_pixels_remove or stimElectrodesAdd:
                if selected_pixels_add:
                    # electrodes in addition polygon
                    selected_electrodes_add = pixel_to_electrode(selected_pixels_add, voltage_map,
                                                                 selection_threshold)
                    selected_electrodes += selected_electrodes_add
                else:
                    selected_electrodes_add = []
                if selected_pixels_remove:
                    # electrodes in subtraction polygon
                    selected_electrodes_remove = pixel_to_electrode(selected_pixels_remove, voltage_map,
                                                                    selection_threshold)
                    selected_electrodes = [i for i in selected_electrodes if i not in selected_electrodes_remove]
                else:
                    selected_electrodes_remove = []

                add_remove_hist.append(selected_electrodes_add)
                add_remove_hist.append(selected_electrodes_remove)
                stimElectrodes += stimElectrodesAdd
                stimElectrodesHist.append(copy.deepcopy(stimElectrodes))
                stimElectrodeAddHist.append(copy.deepcopy(stimElectrodesAdd))
                selected_electrodes_hist.append(copy.deepcopy(selected_electrodes))
                selection_threshold_hist.append(0)
                selected_electrodes_indices = np.atleast_2d(selected_electrodes)[:,:1] * width + np.atleast_2d(selected_electrodes)[:,1:]
                selected_electrodes_indices = np.sort(selected_electrodes_indices)
                selected_electrodes_boolean = np.zeros((height, width))
                for index in selected_electrodes:
                    selected_electrodes_boolean[index[0], index[1]] = 1
                for nr, index in enumerate(stimElectrodes):
                    selected_electrodes_boolean[index[0], index[1]] = nr + 2
                stimElectrodesPixel = []
                selected_pixels_add = []
                selected_pixels_remove = []
                voltage_map_rgb_add = []
                voltage_map_rgb_remove = []
                stimElectrodesAdd = []
                print('electrodes selected: ', len(selected_electrodes), '\n')
                voltage_map_rgb = updatePlot(initial_map.copy())
            else:
                print('currently no selection')

        if key == ord("e"):
            from maxlabCommands.Array import arrayInterface
            from data.utils.h5pyUtils import getMetaInfo,loadh5pySpikes
            # add or subtract electrodes based on drawn polygons
            # if selected_pixels_add or selected_pixels_remove or stimElectrodesAdd:
            if selected_pixels_add or selected_pixels_remove:
                selected_electrodes_pre = []
                if selected_pixels_add:
                    # electrodes in addition polygon
                    selected_electrodes_add_pre = pixel_to_electrode(selected_pixels_add, voltage_map,
                                                                 selection_threshold)
                    selected_electrodes_pre += selected_electrodes_add_pre
                else:
                    selected_electrodes_add_pre = []
                if selected_pixels_remove:
                    # electrodes in subtraction polygon
                    selected_electrodes_remove_pre = pixel_to_electrode(selected_pixels_remove, voltage_map,
                                                                    selection_threshold)
                    selected_electrodes_pre = [i for i in selected_electrodes_pre if i not in selected_electrodes_remove_pre]
                else:
                    selected_electrodes_remove_pre = []

                print("Start spontaneous recording")
                filename = f"temp{datetime.datetime.today().strftime('%Y_%m_%d__%H_%M_%S')}"
                saveDirectory = "/home/mw3/Documents/Joel"
                selected_electrodes_pre = [i[0]*width+i[1] for i in selected_electrodes_pre]
                maxlabInterface = arrayInterface()
                maxlabInterface.routeElectrodes(selected_electrodes_pre)
                maxlabInterface.spontaneousRecordAndSave(duration=30,filename=filename,saveDirectory=saveDirectory)
                print("Finished recording")
                filenameStimulus = os.path.join(saveDirectory,filename+".raw.h5")
                electrodeChannelMapping = getMetaInfo(filenameStimulus)[0]
                # Get wanted subselection of recording
                channelIndices = np.intersect1d(electrodeChannelMapping[0], selected_electrodes_pre, return_indices=True)[1]
                # Get spikedata of wanted electrodes
                # spikeData: [spikeTime, spikeAmplitude, spikeChannel]
                spikeData = loadh5pySpikes([filenameStimulus], electrodeChannelMapping[1][channelIndices])
                # aboveShreshIndice = np.where(np.abs(spikeData[1]) > 50)[0]
                # channel = spikeData[2, aboveShreshIndice]
                channel = spikeData[2]
                counts = np.array(np.unique(channel, return_counts=True))
                if len(counts) == 0:
                    bestChannel = (electrodeChannelMapping[1][channelIndices])[0]
                    print("Most active electrode not found")
                else:
                    counts = counts[:, counts[1].argsort()[::-1]]
                    try:
                        bestChannel = counts[0, 0]
                        print(f"Counted {counts[1,0]} spikes on best, counted {counts[1]} overall")
                    except:
                        bestChannel = (electrodeChannelMapping[1][channelIndices])[0]
                        print("Most active electrode not found")
                bestElectrode = electrodeChannelMapping[0, np.where(electrodeChannelMapping[1] == bestChannel)[0]][0]
                #bestElectrode = np.random.choice(selected_electrodes_pre,1)[0]
                print("Finished auto stimulation electrode selection")
                print(f"Selected {bestElectrode}")
                selected_pixels_add = []
                selected_pixels_remove = []

                clickRegistered[0] = True
                stimElectrodesPixel.append((scale_factor*(bestElectrode % width),scale_factor*(bestElectrode // width)))
            else:
                print('currently no selection')

        if key == ord("s"):
            # randomly sample n electrodes from selection
            selection_threshold_hist.append(0)
            electrodesToRoute = np.sort(np.unique(np.atleast_2d(selected_electrodes)[:,:1] * width + np.atleast_2d(selected_electrodes)[:,1:]))
            stimElectrodesToRoute = np.sort(np.unique(np.atleast_2d(stimElectrodes)[:,:1] * width + np.atleast_2d(stimElectrodes)[:,1:]))
            onlyRecordElectrodes = np.asarray([x for x in electrodesToRoute if not np.isin(x, stimElectrodesToRoute).any()])
            stimElectrodesToRoute = np.transpose([(stimElectrodesToRoute // width).tolist(),(stimElectrodesToRoute % width).tolist()])
            onlyRecordElectrodes = np.transpose([(onlyRecordElectrodes // width).tolist(), (onlyRecordElectrodes % width).tolist()])
            #nrOfElectrodes
            if add_remove_hist and len(onlyRecordElectrodes)  > n_sample - len(stimElectrodesToRoute):
                print('sampling ' + str(n_sample) + ' electrodes')
                selected_electrodes = np.zeros([n_sample,2],dtype=int)
                selected_electrodes[:len(stimElectrodesToRoute)] = np.asarray(stimElectrodesToRoute)
                selected_electrodes[len(stimElectrodesToRoute):] = onlyRecordElectrodes[np.random.choice(len(onlyRecordElectrodes),
                                                                   size=n_sample - len(stimElectrodesToRoute),
                                                                   replace=False).astype(int)]
                selected_electrodes_hist.append(copy.deepcopy(selected_electrodes))
                stimElectrodesHist.append(copy.deepcopy(stimElectrodes))
                stimElectrodeAddHist.append(copy.deepcopy(stimElectrodes))
                add_remove_hist.append(copy.deepcopy(selected_electrodes))
                add_remove_hist.append([])
                selected_electrodes_indices = np.atleast_2d(selected_electrodes)[:,:1] * width + np.atleast_2d(selected_electrodes)[:,1:]
                selected_electrodes_indices = np.sort(selected_electrodes_indices)
                selected_electrodes_boolean = np.zeros((height, width))
                for index in selected_electrodes:
                    selected_electrodes_boolean[index[0], index[1]] = 1
                for nr, index in enumerate(stimElectrodes):
                    selected_electrodes_boolean[index[0], index[1]] = nr + 2

                print('electrodes removed:     ' + str(
                    len(selected_electrodes_hist[-2]) - len(selected_electrodes_hist[-1])))
                voltage_map_rgb = updatePlot(initial_map.copy())
            else:
                print(' ')
            print('electrodes selected: ', len(selected_electrodes), '\n')
            print('stim electrodes selected: ', len(stimElectrodes), '\n')

        if key == ord("z"):
            # revert selection one step
            if len(selected_electrodes_hist) >= 2:
                selected_electrodes = copy.deepcopy(selected_electrodes_hist[-2])
                if len(selected_electrodes_hist[-1]) >= len(selected_electrodes_hist[-2]):
                    print('electrodes unadded:   ' + str(
                        len(selected_electrodes_hist[-1]) - len(selected_electrodes_hist[-2])))
                elif len(selected_electrodes_hist[-2]) >= len(selected_electrodes_hist[-1]):
                    print('electrodes readded:   ' + str(
                        len(selected_electrodes_hist[-2]) - len(selected_electrodes_hist[-1])))
            elif len(selected_electrodes_hist) == 1:
                selected_electrodes = []
                print('electrodes unadded:   ' + str(len(selected_electrodes_hist[-1])))
            elif not selected_electrodes_hist:
                print('')

            if len(stimElectrodesHist) >= 2:
                stimElectrodes = copy.deepcopy(stimElectrodesHist[-2])
                if len(stimElectrodesHist[-1]) >= len(stimElectrodesHist[-2]):
                    print('stim electrodes unadded:   ' + str(
                        len(stimElectrodesHist[-1]) - len(stimElectrodesHist[-2])))
                elif len(stimElectrodesHist[-2]) >= len(stimElectrodesHist[-1]):
                    print('stim electrodes readded:   ' + str(
                        len(stimElectrodesHist[-2]) - len(stimElectrodesHist[-1])))
            elif len(stimElectrodesHist) == 1:
                stimElectrodes = []
                print('stim electrodes unadded:   ' + str(len(stimElectrodesHist[-1])))
            elif not stimElectrodesHist:
                print('')
            selected_electrodes_indices = np.atleast_2d(selected_electrodes)[:,:1] * width + np.atleast_2d(selected_electrodes)[:,1:]
            selected_electrodes_indices = np.sort(selected_electrodes_indices)
            selected_electrodes_boolean = np.zeros((height, width))
            for index in selected_electrodes:
                selected_electrodes_boolean[index[0], index[1]] = 1
            for nr, index in enumerate(stimElectrodes):
                selected_electrodes_boolean[index[0], index[1]] = nr + 2
            stimElectrodesHist = stimElectrodesHist[:-1 or None]
            stimElectrodeAddHist = stimElectrodeAddHist[:-1 or None]
            selected_electrodes_hist = selected_electrodes_hist[:-1 or None]
            add_remove_hist = add_remove_hist[:-2 or None]
            print('electrodes selected: ', len(selected_electrodes), '\n')
            voltage_map_rgb = updatePlot(initial_map.copy())

        if key == ord("r"):
            # reset selection
            selected_pixels_add = []
            selected_pixels_remove = []
            selected_electrodes = []
            stimElectrodes = []
            selected_electrodes_indices = []
            selection_threshold = selection_threshold_default
            selected_electrodes_hist = []
            selection_threshold_hist = []
            add_remove_hist = []
            stimElectrodesHist = []
            stimElectrodeAddHist = []
            voltage_map_rgb_add = copy.deepcopy(initial_map)
            voltage_map_rgb_remove = copy.deepcopy(initial_map)
            prev_map = copy.deepcopy(initial_map)
            print('selection resetted')
            voltage_map_rgb = updatePlot(initial_map.copy())


        if key == ord("c"):
            singleNetworks = []
            for nr,network in enumerate(add_remove_hist[::2]):
                noNetwork = np.zeros((height,width))
                selected_electrodes_remove = add_remove_hist[nr*2+1::2]
                network = np.asarray([i for i in network if i not in selected_electrodes_remove])
                stim = stimElectrodeAddHist[nr]
                for index in network:
                    noNetwork[index[0], index[1]] = 1
                for k, index in enumerate(stim):
                    noNetwork[index[0], index[1]] = k + 2
                singleNetworks.append(noNetwork)
            # end script
            cv2.destroyAllWindows()
            return selected_electrodes_boolean, singleNetworks
        cv2.imshow('voltage_map', voltage_map_rgb)
        cv2.putText(voltage_map_rgb,
                    'C: close | A: confirm selection | E: Pick Stim Electrode | R: reset selection | MMB Click: stim electrode selection |S: sample ' + str(
                        n_sample) + ' electrodes | Z: revert action | L/R Click: add join/disjoin vertex',
                    (0, int(round(0.995 * windowSize[0]))), cv2.FONT_HERSHEY_TRIPLEX, fontSize,
                    color=(255, 255, 255))


def getElectrodeListsWithSelection(array: np.ndarray, savePath: str, filename: str = "selection",
                                   loadFileBool: bool = False, selection_threshold: int = 40, nNetworks: int = 0,
                                   n_sample: int = 1024, multiInputBool: bool = False):
    if loadFileBool:
        networks = []
        total = np.load(os.path.join(savePath,filename+".npy"))
        for i in range(nNetworks):
            networks.append(np.load(os.path.join(savePath, filename + "_N{}.npy".format(i))))
    else:
        total, networks = selectElements(array,selection_threshold,n_sample)
        np.save(os.path.join(savePath,filename+".npy"),total)
        for i, network in enumerate(networks):
            np.save(os.path.join(savePath, filename + "_N{}.npy".format(i)), network)

    stimElectrodes = []
    for i, network in enumerate(networks):
        temp = []
        for i in range(2, np.max(network).astype(int) + 1):
            temp.append(np.squeeze(np.where(network.flatten() == i)).tolist())
        stimElectrodes.append(temp)
    routing = np.where(total.flatten() > 0)[0]
    if multiInputBool:
        return routing.tolist(), stimElectrodes
    else:
        stimElectrodes = np.where(total.flatten() > 1)[0]
        return routing.tolist(), stimElectrodes.tolist()