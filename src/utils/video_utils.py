import os
import subprocess
import shutil


def create_video_from_intermediate_results(results_path):
    #
    # change this depending on what you want to accomplish (modify out video
    # name, change fps and trim video)
    #
    img_format = (4, '.jpg')
    out_file_name = 'out.mp4'
    fps = 10
    first_frame = 0
    number_of_frames_to_process = len(os.listdir(results_path))
    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        # example: '%4d.png' for (4, '.png')
        img_name_format = '%' + str(img_format[0]) + 'd' + img_format[1]
        pattern = os.path.join(results_path, img_name_format)
        out_video_path = os.path.join(results_path, out_file_name)

        trim_video_command = [
            '-start_number',
            str(first_frame), '-vframes',
            str(number_of_frames_to_process)
        ]
        input_options = ['-r', str(fps), '-i', pattern]
        encoding_options = [
            '-c:v', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p', 
            '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        ]
        subprocess.call([
            ffmpeg, *input_options, *trim_video_command, *encoding_options,
            out_video_path
        ])
    else:
        print(f'{ffmpeg} not found in the system path, aborting.')
