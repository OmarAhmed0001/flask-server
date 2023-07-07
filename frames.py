from datetime import timedelta
import cv2
import numpy as np
import os
import dlib
SAVING_FRAMES_PER_SECOND = 2
dlib_detector=dlib.get_frontal_face_detector()

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05)
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")

def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)

        # if i >9:
        #     break
    print(s)
    return s

def solve(video_file,face_path,frame_path):
    face_list=[]
    frame_list=[]
    filename, _ = os.path.splitext(video_file)
    filename += "-opencv"
    video_name=video_file.split("\\")
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        #os.mkdir(filename)
        pass
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    saving_frames_per_second = min(fps, SAVING_FRAMES_PER_SECOND)
    print(f"Saving frames at {saving_frames_per_second} fps")
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
    count = 0
    while True:
        is_read, frame = cap.read()
        if not is_read:
            break
        faces_rect = dlib_detector(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY))
        frame_duration = count / fps
        try:
            closest_duration = saving_frames_durations[0]
        except IndexError:
            break
        if frame_duration >= closest_duration:
            frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            for face in faces_rect:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                # print(x,y,w,h)
                img=frame[y:y+h,x:x+w]
                try:
                    img=cv2.resize(img,(128,128),interpolation=cv2.INTER_AREA)
                    frame=cv2.resize(frame,(128,128),interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(face_path, f"video{video_name[-1]}face{frame_duration_formatted}.jpg"), img)
                    cv2.imwrite(os.path.join(frame_path, f"video{video_name[-1]}frame{frame_duration_formatted}.jpg"), frame)
                    face_list.append(f"video{video_name[-1]}face{frame_duration_formatted}.jpg")
                    frame_list.append(f"video{video_name[-1]}frame{frame_duration_formatted}.jpg")
                    saving_frames_durations.pop(0)
                except:
                    pass
           
                
        # increment the frame count
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    return face_list,frame_list