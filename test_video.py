import argparse

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage, InterpolationMode
from tqdm import tqdm

from model import Generator

"""
python test_video.py --input_name video/input/test.mp4 --output_dir video/output
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='netG_epoch_2_20.pth', type=str, help='generator model epoch name')
    parser.add_argument('--input_name', type=str, help='test low resolution video name')
    parser.add_argument('--output_dir', type=str, help='output video save path')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name
    INPUT_PATH = opt.input_name
    OUTPUT_DIR = opt.output_dir
    INPUT_DIR, INPUT_FILE= os.path.splitext(INPUT_PATH)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    videoCapture = cv2.VideoCapture(INPUT_PATH)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    width = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sr_width = int(width * UPSCALE_FACTOR)
    sr_height = int(height * UPSCALE_FACTOR)
    co_width = int(width * UPSCALE_FACTOR * 2 + 10)
    co_height = int(height) * UPSCALE_FACTOR + 10 + int(co_width / int(10 * int(int(width * UPSCALE_FACTOR) // 5 + 1)) * int(int(width * UPSCALE_FACTOR) // 5 - 9))

    sr_video_size = (sr_width, sr_height)
    co_video_size = (co_width, co_height)

    output_sr_name = os.path.join(OUTPUT_DIR, 'out_srf_' + str(UPSCALE_FACTOR) + '_' + INPUT_FILE.split('.')[0] + '.avi')
    output_co_name = os.path.join(OUTPUT_DIR, 'compare_srf_' + str(UPSCALE_FACTOR) + '_' + INPUT_FILE.split('.')[0] + '.avi')
    sr_video_writer = cv2.VideoWriter(output_sr_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, sr_video_size)
    co_video_writer = cv2.VideoWriter(output_co_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, co_video_size)

    with torch.no_grad():
        model = Generator(UPSCALE_FACTOR)
        model.eval()
        if torch.cuda.is_available():
            model = model.to(DEVICE)
        # for cpu
        # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
        model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

        # read frame
        success, frame = videoCapture.read()
        test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]')
        frame_count = 0
        for index in test_bar:
            if success:
                # frame = cv2.resize(frame, (int(width//2), int(height//2)))
                image = ToTensor()(frame).unsqueeze(0)
                frame_count += 1
                if torch.cuda.is_available():
                    image = image.to(DEVICE)

                # print(f"model got frame{frame_count}")
                out = model(image)
                # print(f"model infered frame{frame_count}")
                out_cpu = out.cpu()
                del out
                torch.cuda.empty_cache()
                out_img = out_cpu.data[0].numpy()
                out_img *= 255.0
                out_img = (np.uint8(out_img)).transpose((1, 2, 0))
                # save sr video
                sr_video_writer.write(out_img)

                # make compared video and crop shot of left top\right top\center\left bottom\right bottom
                out_img = ToPILImage()(out_img)
                crop_out_imgs = transforms.FiveCrop(size=out_img.width // 5 - 9)(out_img)
                crop_out_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_out_imgs]
                out_img = transforms.Pad(padding=(5, 0, 0, 5))(out_img)
                co_img = transforms.Resize(size=(sr_video_size[1], sr_video_size[0]), interpolation=InterpolationMode.BICUBIC)(
                    ToPILImage()(frame))
                crop_co_imgs = transforms.FiveCrop(size=co_img.width // 5 - 9)(co_img)
                crop_co_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_co_imgs]
                co_img = transforms.Pad(padding=(0, 0, 5, 5))(co_img)
                # concatenate all the pictures to one single picture
                top_image = np.concatenate((np.asarray(co_img), np.asarray(out_img)), axis=1)
                bottom_image = np.concatenate(crop_co_imgs + crop_out_imgs, axis=1)
                bottom_image = np.asarray(transforms.Resize(
                    size=(int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0]), top_image.shape[1]))(
                    ToPILImage()(bottom_image)))
                final_image = np.concatenate((top_image, bottom_image))
                # save compared video
                co_video_writer.write(final_image)
                # next frame
                success, frame = videoCapture.read()
