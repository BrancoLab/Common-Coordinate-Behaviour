import cv2
import numpy as np
import scipy.misc
from termcolor import colored
from tqdm import tqdm
import glob
import os


def get_background(vidpath, start_frame = 1000, avg_over = 100):
    """ extract background: average over frames of video """

    vid = cv2.VideoCapture(vidpath)

    # initialize the video
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    j = 0

    for i in tqdm(range(num_frames)):

        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()  # get the frame

            if ret:
                # store the current frame in as a numpy array
                background += frame[:, :, 0]
                j+=1


    background = (background / (j)).astype(np.uint8)
    cv2.imshow('background', background)
    cv2.waitKey(10)
    vid.release()

    return background

# =================================================================================
#              CREATE MODEL ARENA FOR COMMON COORDINATE BEHAVIOUR
# =================================================================================
def model_arena(size):
    ''' NOTE: this is the model arena for the Barnes maze with wall
    this function must be customized for any other arena'''
    # initialize model arena
    model_arena = np.zeros((1000,1000)).astype(np.uint8)
    cv2.circle(model_arena, (500,500), 460, 255, -1)

    # add wall - up
    cv2.rectangle(model_arena, (int(500 - 554 / 2), int(500 - 6 / 2)), (int(500 + 554 / 2), int(500 + 6 / 2)), 60, thickness=-1)
    # add wall - down
    cv2.rectangle(model_arena, (int(500 - 504 / 2), int(500 - 8 / 2)), (int(500 + 504 / 2), int(500 + 8 / 2)), 0, thickness=-1)

    # add shelter
    model_arena_shelter = model_arena.copy()
    cv2.rectangle(model_arena_shelter, (int(500 - 50), int(500 + 385 + 25 - 50)), (int(500 + 50), int(500 + 385 + 25 + 50)), (0, 0, 255),thickness=-1)
    alpha = .5
    cv2.addWeighted(model_arena, alpha, model_arena_shelter, 1 - alpha, 0, model_arena)

    # add circular wells along edge
    number_of_circles = 20
    for circle_num in range(number_of_circles):
        x_center = int(500+385*np.sin(2*np.pi/number_of_circles*circle_num))
        y_center = int(500-385*np.cos(2*np.pi/number_of_circles*circle_num))
        cv2.circle(model_arena,(x_center,y_center),25,0,-1)

    model_arena = cv2.resize(model_arena,size)

    # --------------------------------------------------------------------------------------------
    # THESE ARE THE FOUR POINTS USED TO INITIATE REGISTRATION -- CUSTOMIZE FOR YOUR OWN PURPOSES
    # --------------------------------------------------------------------------------------------
    points = np.array(([500,500+460-75],[500-460+75,500],[500,500-460+75],[500+460-75,500]))* [size[0]/1000,size[1]/1000]

    # cv2.imshow('model_arena',model_arena)

    return model_arena, points

# =================================================================================
#              IMAGE REGISTRATION GUI
# =================================================================================
def register_arena(background, fisheye_map_location, x_offset, y_offset):
    """ extract background: first frame of first video of a session
    Allow user to specify ROIs on the background image """

    # create model arena and background
    arena, arena_points = model_arena(background.shape)

    # load the fisheye correction
    try:
        maps = np.load(fisheye_map_location)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2]*0

        background_copy = cv2.copyMakeBorder(background, x_offset, int((map1.shape[0] - background.shape[0]) - x_offset),
                                             y_offset, int((map1.shape[1] - background.shape[1]) - y_offset),cv2.BORDER_CONSTANT, value=0)

        background_copy = cv2.remap(background_copy, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        background_copy = background_copy[x_offset:-int((map1.shape[0] - background.shape[0])- x_offset),
                                          y_offset:-int((map1.shape[1] - background.shape[1]) - y_offset)]
    except:
        background_copy = background.copy()
        fisheye_map_location = ''
        print('fisheye correction not available')

    # initialize clicked points
    blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]
    cv2.namedWindow('registered background')
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)
    use_loaded_transform = False
    make_new_transform_immediately = False
    use_loaded_points = False

    # LOOP OVER TRANSFORM FILES
    file_num = -1;
    transform_files = glob.glob('*transform.npy')
    for file_num, transform_file in enumerate(transform_files[::-1]):

        # USE LOADED TRANSFORM AND SEE IF IT'S GOOD
        loaded_transform = np.load(transform_file)
        M = loaded_transform[0]
        background_data[1] = loaded_transform[1]
        arena_data[1] = loaded_transform[2]

        # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)
        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                       * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        print('Does transform ' + str(file_num+1) + ' / ' + str(len(transform_files)) + ' match this session?')
        print('\'y\' - yes! \'n\' - no. \'q\' - skip examining loaded transforms. \'p\' - update current transform')
        while True:
            cv2.imshow('registered background', overlaid_arenas)
            k = cv2.waitKey(10)
            if  k == ord('n'):
                break
            elif k == ord('y'):
                use_loaded_transform = True
                break
            elif k == ord('q'):
                make_new_transform_immediately = True
                break
            elif k == ord('p'):
                use_loaded_points = True
                break
        if use_loaded_transform:
            update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M]
            break
        elif make_new_transform_immediately or use_loaded_points:
            file_num = len(glob.glob('*transform.npy'))-1
            break

    if not use_loaded_transform:
        if not use_loaded_points:
            print('\nSelect reference points on the experimental background image in the indicated order')

            # initialize clicked point arrays
            background_data = [background_copy, np.array(([], [])).T]
            arena_data = [[], np.array(([], [])).T]

            # add 1-2-3-4 markers to model arena
            for i, point in enumerate(arena_points.astype(np.uint32)):
                arena = cv2.circle(arena, (point[0], point[1]), 3, 255, -1)
                arena = cv2.circle(arena, (point[0], point[1]), 4, 0, 1)
                cv2.putText(arena, str(i+1), tuple(point), 0, .55, 150, thickness=2)

                point = np.reshape(point, (1, 2))
                arena_data[1] = np.concatenate((arena_data[1], point))

            # initialize GUI
            cv2.startWindowThread()
            cv2.namedWindow('background')
            cv2.imshow('background', background_copy)
            cv2.namedWindow('model arena')
            cv2.imshow('model arena', arena)

            # create functions to react to clicked points
            cv2.setMouseCallback('background', select_transform_points, background_data)  # Mouse callback

            while True: # take in clicked points until four points are clicked
                cv2.imshow('background',background_copy)

                number_clicked_points = background_data[1].shape[0]
                if number_clicked_points == len(arena_data[1]):
                    break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        # perform projective transform
        # M = cv2.findHomography(background_data[1], arena_data[1])
        M = cv2.estimateRigidTransform(background_data[1], arena_data[1], False)


        # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
        # registered_background = cv2.warpPerspective(background_copy,M[0],background.shape)
        registered_background = cv2.warpAffine(background_copy, M, background.shape)

        # --------------------------------------------------
        # overlay images
        # --------------------------------------------------
        alpha = .7
        colors = [[150, 0, 150], [0, 255, 0]]
        color_array = make_color_array(colors, background.shape)

        registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                 * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
        arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                       * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

        overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
        cv2.imshow('registered background', overlaid_arenas)

        # --------------------------------------------------
        # initialize GUI for correcting transform
        # --------------------------------------------------
        print('\nLeft click model arena // Right click model background')
        print('Purple within arena and green along the boundary represent the model arena')
        print('Advanced users: use arrow keys and \'wasd\' to fine-tune translation and scale as a final step')
        print('Crème de la crème: use \'tfgh\' to fine-tune shear\n')
        print('y: save and use transform')
        print('r: reset transform (left and right click four points to recommence)')
        update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]

        # create functions to react to additional clicked points
        cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

        # take in clicked points until 'q' is pressed
        initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        M_initial = M
        M_indices = [(0,2),(1,2),(0,0),(1,1),(0,1),(1,0),(2,0),(2,2)]
        # M_indices_meanings = ['x-translate','y-translate','x-scale','y-scale','x->y shear','y->x shear','x perspective','y perspective']
        M_mod_keys = [2424832, 2555904, 2490368, 2621440, ord('w'), ord('a'), ord('s'), ord('d'), ord('f'), ord('t'),
                      ord('g'), ord('h'), ord('j'), ord('i'), ord('k'), ord('l')]
        while True:
            cv2.imshow('registered background',overlaid_arenas)
            cv2.imshow('background', registered_background)
            number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
            update_transform = False
            k = cv2.waitKey(10)
            # If a left and right point are clicked:
            if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
                initial_number_clicked_points = number_clicked_points
                # update transform and overlay images
                try:
                    # M = cv2.findHomography(update_transform_data[1], update_transform_data[2])
                    M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],False) #True ~ full transform
                    update_transform = True
                except:
                    continue
            elif k in M_mod_keys: # if an arrow key if pressed
                if k == 2424832: # left arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] - abs(M_initial[M_indices[0]]) * .005
                elif k == 2555904: # right arrow - x translate
                    M[M_indices[0]] = M[M_indices[0]] + abs(M_initial[M_indices[0]]) * .005
                elif k == 2490368: # up arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] - abs(M_initial[M_indices[1]]) * .005
                elif k == 2621440: # down arrow - y translate
                    M[M_indices[1]] = M[M_indices[1]] + abs(M_initial[M_indices[1]]) * .005
                elif k == ord('a'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] + abs(M_initial[M_indices[2]]) * .005
                elif k == ord('d'): # down arrow - x scale
                    M[M_indices[2]] = M[M_indices[2]] - abs(M_initial[M_indices[2]]) * .005
                elif k == ord('s'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] + abs(M_initial[M_indices[3]]) * .005
                elif k == ord('w'): # down arrow - y scale
                    M[M_indices[3]] = M[M_indices[3]] - abs(M_initial[M_indices[3]]) * .005
                elif k == ord('f'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] - abs(M_initial[M_indices[4]]) * .005
                elif k == ord('h'): # down arrow - x-y shear
                    M[M_indices[4]] = M[M_indices[4]] + abs(M_initial[M_indices[4]]) * .005
                elif k == ord('t'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] - abs(M_initial[M_indices[5]]) * .005
                elif k == ord('g'): # down arrow - y-x shear
                    M[M_indices[5]] = M[M_indices[5]] + abs(M_initial[M_indices[5]]) * .005

                update_transform = True

            elif  k == ord('r'):
                print('Transformation erased')
                update_transform_data[1] = np.array(([],[])).T
                update_transform_data[2] = np.array(([],[])).T
                initial_number_clicked_points = [3,3]
            elif k == ord('q') or k == ord('y'):
                print('Registration completed')
                break

            if update_transform:
                update_transform_data[3] = M
                # registered_background = cv2.warpPerspective(background_copy, M, background.shape)
                registered_background = cv2.warpAffine(background_copy, M, background.shape)
                registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                               * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
                overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
                update_transform_data[0] = overlaid_arenas

        np.save(str(file_num+1)+'_transform',[M, update_transform_data[1], update_transform_data[2], fisheye_map_location])

    cv2.destroyAllWindows()
    return [M, update_transform_data[1], update_transform_data[2], fisheye_map_location]


# mouse callback function I
def select_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[1] = np.concatenate((data[1], clicks))

# mouse callback function II
def additional_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])
        # M_inverse = np.linalg.inv(data[3])
        # M_inverse = cv2.findHomography(data[2][:len(data[1])], data[1])
        # transformed_clicks = np.matmul(M_inverse[0], [x, y, 1])

        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[2] = np.concatenate((data[2], clicks))

def make_color_array(colors, image_size):
    color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                colors[c])
    return color_array



# =================================================================================
#              GENERATE PERI-STIMULUS VIDEO CLIPS and FLIGHT IMAGE
# =================================================================================
def peri_stimulus_video_clip(vidpath = '', videoname = '', savepath = '', start_frame=0., end_frame=100., stim_frame = 0,
                             registration = 0, x_offset = 0, y_offset = 0, dark_threshold = [.55, 950],
                             fps=False, save_clip = False, display_clip = False, counter = True, make_flight_image = True):
    # GET BEAHVIOUR VIDEO - ######################################
    vid = cv2.VideoCapture(vidpath)
    if not fps: fps = vid.get(cv2.CAP_PROP_FPS)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # SETUP VIDEO CLIP SAVING - ######################################
    # file_already_exists = os.path.isfile(os.path.join(savepath,videoname+'.avi'))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # LJPG for lossless, XVID for compressed
    border_size = 20
    if save_clip:
        video_clip = cv2.VideoWriter(os.path.join(savepath,videoname+'.avi'), fourcc, fps, (width+2*border_size*counter, height+2*border_size*counter), counter)
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    pre_stim_color = [255, 120, 120]
    post_stim_color = [120, 120, 255]

    if registration[3]: # setup fisheye correction
        maps = np.load(registration[3])
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2] * 0
    else:
        print(colored('Fisheye correction unavailable', 'green'))
    # RUN SAVING AND ANALYSIS OVER EACH FRAME - ######################################
    while True: #and not file_already_exists:
        ret, frame = vid.read()  # get the frame
        if ret:
            frame_num = vid.get(cv2.CAP_PROP_POS_FRAMES)
            if [registration]:
                # load the fisheye correction
                frame_register = frame[:, :, 0]
                if registration[3]:
                    frame_register = cv2.copyMakeBorder(frame_register, x_offset, int((map1.shape[0] - frame.shape[0]) - x_offset),
                                                         y_offset, int((map1.shape[1] - frame.shape[1]) - y_offset), cv2.BORDER_CONSTANT, value=0)
                    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    frame_register = frame_register[x_offset:-int((map1.shape[0] - frame.shape[0]) - x_offset),
                                      y_offset:-int((map1.shape[1] - frame.shape[1]) - y_offset)]
                frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)

            # MAKE ESCAPE TRAJECTORY IMAGE - #######################################
            if make_flight_image:
                # at stimulus onset, take this frame to lay all the superimposed mice on top of
                if frame_num == stim_frame:
                    flight_image_by_distance = frame[:,:,0].copy()

                # in subsequent frames, see if frame is different enough from previous image to merit joining the image
                elif frame_num > stim_frame and (frame_num - stim_frame) < fps*10:
                    # get the number of pixels that are darker than the flight image
                    difference_from_previous_image = ((frame[:,:,0]+.001) / (flight_image_by_distance+.001))<dark_threshold[0] #.5 original parameter
                    number_of_darker_pixels = np.sum(difference_from_previous_image)

                    # if that number is high enough, add mouse to image
                    if number_of_darker_pixels > dark_threshold[1]: # 850 original parameter
                        # add mouse where pixels are darker
                        flight_image_by_distance[difference_from_previous_image] = frame[difference_from_previous_image,0]

            # SHOW BOUNDARY AND TIME COUNTER - #######################################
            if counter and (display_clip or save_clip):
                # cv2.rectangle(frame, (0, height), (150, height - 60), (150,150,150), -1)
                if frame_num < stim_frame:
                    cur_color = tuple([x * ((frame_num - start_frame) / (stim_frame - start_frame)) for x in pre_stim_color])
                    sign = ''
                else:
                    cur_color = tuple([x * (1 - (frame_num - stim_frame) / (end_frame-stim_frame))  for x in post_stim_color])
                    sign = '+'

                # border and colored rectangle around frame
                frame = cv2.copyMakeBorder(frame, border_size, border_size, border_size, border_size,cv2.BORDER_CONSTANT, value=cur_color)

                # report video details
                cv2.putText(frame, videoname, (20, 40), 0, .55, (180, 180, 180), thickness=1)

                # report time relative to stimulus onset
                frame_time = (frame_num - stim_frame) / fps
                frame_time = str(round(.2*round(frame_time/.2), 1))+ '0'*(abs(frame_time)<10)
                cv2.putText(frame, sign + str(frame_time) + 's', (width-110, height+10), 0, 1,(180,180,180), thickness=2)

            else:
                frame = frame[:,:,0] # or use 2D grayscale image instead

            # SHOW AND SAVE FRAME - #######################################
            if display_clip:
                cv2.imshow('Trial Clip', frame)
            if save_clip:
                video_clip.write(frame)
            if display_clip:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if frame_num >= end_frame:
                break
        else:
            print('Problem with movie playback')
            cv2.waitKey(1000)
            break

    # wrap up
    vid.release()
    if make_flight_image:
        flight_image_by_distance = cv2.copyMakeBorder(flight_image_by_distance, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=0)
        cv2.putText(flight_image_by_distance, videoname, (border_size, border_size-5), 0, .55, (180, 180, 180), thickness=1)
        cv2.imshow('Flight image', flight_image_by_distance)
        cv2.waitKey(10)
        scipy.misc.imsave(os.path.join(savepath, videoname + '.tif'), flight_image_by_distance)
    if save_clip:
        video_clip.release()
    # cv2.destroyAllWindows()


def invert_fisheye_map(registration, inverse_fisheye_map_location):
    '''Go from a normal opencv fisheye map to an inverted one, so coordinates can be transform'''

    if len(registration) == 5:
        pass
    elif os.path.isfile(inverse_fisheye_map_location):
        registration.append(inverse_fisheye_map_location)
    elif len(registration) == 4:  # setup fisheye correction
        print('creating inverse fisheye map')
        inverse_maps = np.load(registration[3])
        # invert maps
        inverse_maps[inverse_maps < 0] = 0

        maps_x_orig = inverse_maps[:, :, 0]
        maps_x_orig[maps_x_orig > 1279] = 1279
        maps_y_orig = inverse_maps[:, :, 1]
        maps_y_orig[maps_y_orig > 1023] = 1023

        map_x = np.ones(inverse_maps.shape[0:2]) * np.nan
        map_y = np.ones(inverse_maps.shape[0:2]) * np.nan
        for x in range(inverse_maps.shape[1]):
            for y in range(inverse_maps.shape[0]):
                map_x[maps_y_orig[y, x], maps_x_orig[y, x]] = x
                map_y[maps_y_orig[y, x], maps_x_orig[y, x]] = y

        grid_x, grid_y = np.mgrid[0:inverse_maps.shape[0], 0:inverse_maps.shape[1]]
        valid_values_x = np.ma.masked_invalid(map_x)
        valid_values_y = np.ma.masked_invalid(map_y)

        valid_idx_x_map_x = grid_x[~valid_values_x.mask]
        valid_idx_y_map_x = grid_y[~valid_values_x.mask]

        valid_idx_x_map_y = grid_x[~valid_values_y.mask]
        valid_idx_y_map_y = grid_y[~valid_values_y.mask]

        map_x_interp = interpolate.griddata((valid_idx_x_map_x, valid_idx_y_map_x), map_x[~valid_values_x.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)
        map_y_interp = interpolate.griddata((valid_idx_x_map_y, valid_idx_y_map_y), map_y[~valid_values_y.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)

        fisheye_maps_interp = np.zeros((map_x_interp.shape[0], map_x_interp.shape[1], 2)).astype(np.uint16)
        fisheye_maps_interp[:, :, 0] = map_x_interp
        fisheye_maps_interp[:, :, 1] = map_y_interp

        np.save('C:\\Drive\\DLC\\transforms\\inverse_fisheye_maps.npy', fisheye_maps_interp)

    return registration



def extract_coordinates_with_dlc(dlc_config_settings, video, registration):
    '''extract coordinates for each frame, given a video and DLC network'''

    analyze_videos(dlc_config_settings['config_file'], video)
    # create_labeled_video(dlc_config_settings['config_file'], video)

    # read the freshly saved coordinates file
    coordinates_file = glob.glob(os.path.dirname(video[0]) + '\\*.h5')[0]
    DLC_network = os.path.basename(coordinates_file)
    DLC_network = DLC_network[DLC_network.find('Deep'):-3]
    body_parts = dlc_config_settings['body parts']

    DLC_dataframe = pd.read_hdf(coordinates_file)
    coordinates = {}

    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), DLC_dataframe[DLC_network]['nose'].values.shape[0]))

    # fisheye correct the coordinates
    registration = invert_fisheye_map(registration, dlc_config_settings['inverse_fisheye_map_location'])
    inverse_fisheye_maps = np.load(registration[4])


    for i, body_part in enumerate(body_parts):
        # initialize coordinates
        coordinates[body_part] = np.zeros((2, len(DLC_dataframe[DLC_network][body_part]['x'].values)))

        # extract coordinates
        for j, axis in enumerate(['x', 'y']):
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values

        # put all together
        all_body_parts[:, i, :] = coordinates[body_part]
        median_positions = np.nanmedian(all_body_parts, axis=1)
        # median_distance = np.sqrt(median_positions[0,:]**2 + median_positions[1,:]**2)

    for body_part in body_parts:

        # get likelihood
        likelihood = DLC_dataframe[DLC_network][body_part]['likelihood'].values

        # remove coordinates with low confidence
        coordinates[body_part][0][likelihood < .9999999] = np.nan
        coordinates[body_part][1][likelihood < .9999999] = np.nan

        # remove coordinates far from rest of body parts
        distance_from_median_position = np.sqrt( (coordinates[body_part][0] - median_positions[0,:])**2 + (coordinates[body_part][1] - median_positions[1,:])**2 )
        coordinates[body_part][0][distance_from_median_position > 50] = np.nan
        coordinates[body_part][1][distance_from_median_position > 50] = np.nan

        # lineraly interporlate the low-confidence time points
        coordinates[body_part][0] = np.array(pd.Series(coordinates[body_part][0]).interpolate())
        coordinates[body_part][0][0:np.argmin(np.isnan(coordinates[body_part][0]))] = coordinates[body_part][0][
            np.argmin(np.isnan(coordinates[body_part][0]))]

        coordinates[body_part][1] = np.array(pd.Series(coordinates[body_part][1]).interpolate())
        coordinates[body_part][1][0:np.argmin(np.isnan(coordinates[body_part][1]))] = coordinates[body_part][1][
            np.argmin(np.isnan(coordinates[body_part][1]))]

        # convert original coordinates to registered coordinates
        coordinates[body_part][0] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 0] - x_offset
        coordinates[body_part][1] = inverse_fisheye_maps[
                                        coordinates[body_part][1].astype(np.uint16) + y_offset, coordinates[body_part][
                                            0].astype(np.uint16) + x_offset, 1] - y_offset

        # affine transform to match model arena
        transformed_points = np.matmul(np.append(registration[0], np.zeros((1, 3)), 0),
                                       np.concatenate((coordinates[body_part][0:1], coordinates[body_part][1:2],
                                                       np.ones((1, len(coordinates[body_part][0])))), 0))
        coordinates[body_part][0] = transformed_points[0, :]
        coordinates[body_part][1] = transformed_points[1, :]

        # plot the coordinates
        # ax.plot(np.sqrt((coordinates[body_part][0] - 500 * 720 / 1000) ** 2 + (coordinates[body_part][1] - 885 * 720 / 1000) ** 2))
        # plt.pause(.01)

    # compute some metrics
    for i, body_part in enumerate(body_parts):
        all_body_parts[:, i, :] = coordinates[body_part]

    coordinates['head_location'] = np.nanmean(all_body_parts[:, 0:5, :], axis=1)
    coordinates['snout_location'] = np.nanmean(all_body_parts[:, 0:3, :], axis=1)
    # coordinates['neck_location'] = np.nanmean(all_body_parts[:, 3:5, :], axis=1)
    coordinates['butt_location'] = np.nanmean(all_body_parts[:, 9:, :], axis=1)
    coordinates['back_location'] = np.nanmean(all_body_parts[:, 6:9, :], axis=1)
    coordinates['center_body_location'] = np.nanmean(all_body_parts[:, 6:, :], axis=1)
    coordinates['center_location'] = np.nanmean(all_body_parts[:, :, :], axis=1)

    delta_position = np.concatenate( ( np.zeros((2,1)), np.diff(coordinates['center_location']) ) , axis = 1)
    coordinates['speed'] = np.sqrt(delta_position[0,:]**2 + delta_position[1,:]**2)

    coordinates['distance_from_shelter'] = np.sqrt((coordinates['center_location'][0] - 500 * 720 / 1000) ** 2 +
                                                   (coordinates['center_location'][1] - 885 * 720 / 1000) ** 2)

    coordinates['speed_toward_shelter'] = np.concatenate( ([0], np.diff(coordinates['distance_from_shelter'])))
########################################################################################################################
if __name__ == "__main__":
    peri_stimulus_video_clip(vidpath='', videoname='', savepath='', start_frame=0., end_frame=100., stim_frame=0,
                             registration=0, fps=False, save_clip=False, display_clip=False, counter=True,
                             make_flight_image=True)