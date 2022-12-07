# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:36:23 2022

@author: YASHIM GABRIEL
"""

import argparse
import os
import random
import torch
from flask import Flask, render_template
import cv2 as cv
from torchvision.utils import make_grid, save_image
from PIL import Image
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
# import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import datetime
from torchvision import models
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data import DataLoader
# from pipeline_components.utils import utils
import utils
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from IPython.display import HTML
import subprocess
import time
import shutil
import sys

# from pipeline_components.segmentation import extract_person_masks_from_frames
# from pipeline_components.video_creation import create_videos
# from pipeline_components.constants import *
# from pipeline_components.nst_stylization import stylization
# from pipeline_components.compositor import stylized_frames_mask_combiner

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225])
IMAGENET_MEAN_255 = np.array([123.675, 116.28, 103.53])
# Usually when normalizing 0..255 images only mean-normalization is performed -> that's why standard dev is all 1s here
IMAGENET_STD_NEUTRAL = np.array([1, 1, 1])


class SimpleDataset(Dataset):
    def __init__(self, img_dir, target_width):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir)]

        h, w = load_image(self.img_paths[0]).shape[:2]
        img_height = int(h * (target_width / w))
        self.target_width = target_width
        self.target_height = img_height

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = load_image(self.img_paths[idx], target_shape=(self.target_height, self.target_width))
        tensor = self.transform(img)
        return tensor


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]  # [:, :, ::-1] converts BGR (opencv format...) into RGB

    if target_shape is not None:  # resize section
        if isinstance(target_shape, int) and target_shape != -1:  # scalar -> implicitly setting the width
            current_height, current_width = img.shape[:2]
            new_width = target_shape
            new_height = int(current_height * (new_width / current_width))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:  # set both dimensions to target shape
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def prepare_img(img_path, target_shape, device, batch_size=1, should_normalize=True, is_255_range=False):
    img = load_image(img_path, target_shape=target_shape)

    transform_list = [transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)

    img = transform(img).to(device)
    img = img.repeat(batch_size, 1, 1, 1)

    return img


def post_process_image(dump_img):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy image got {type(dump_img)}'

    mean = IMAGENET_MEAN_1.reshape(-1, 1, 1)
    std = IMAGENET_STD_1.reshape(-1, 1, 1)
    dump_img = (dump_img * std) + mean  # de-normalize
    dump_img = (np.clip(dump_img, 0., 1.) * 255).astype(np.uint8)
    dump_img = np.moveaxis(dump_img, 0, 2)
    return dump_img


def get_next_available_name(input_dir):
    img_name_pattern = re.compile(r'[0-9]{6}\.jpg')
    candidates = [candidate for candidate in os.listdir(input_dir) if re.fullmatch(img_name_pattern, candidate)]

    if len(candidates) == 0:
        return '000000.jpg'
    else:
        latest_file = sorted(candidates)[-1]
        prefix_int = int(latest_file.split('.')[0])
        return f'{str(prefix_int + 1).zfill(6)}.jpg'


def save_and_maybe_display_image(inference_config, dump_img, should_display=False):
    assert isinstance(dump_img, np.ndarray), f'Expected numpy array got {type(dump_img)}.'

    dump_img = post_process_image(dump_img)
    if inference_config['img_width'] is None:
        inference_config['img_width'] = dump_img.shape[0]

    if inference_config['redirected_output'] is None:
        dump_dir = inference_config['output_images_path']
        dump_img_name = os.path.basename(inference_config['content_input']).split('.')[0] + '_width_' + str(inference_config['img_width']) + '_model_' + inference_config['model_name'].split('.')[0] + '.jpg'
    else:  # useful when this repo is used as a utility submodule in some other repo like pytorch-naive-video-nst
        dump_dir = inference_config['redirected_output']
        os.makedirs(dump_dir, exist_ok=True)
        dump_img_name = get_next_available_name(inference_config['redirected_output'])

    cv.imwrite(os.path.join(dump_dir, dump_img_name), dump_img[:, :, ::-1])  # ::-1 because opencv expects BGR (and not RGB) format...

    # Don't print this information in batch stylization mode
    if inference_config['verbose'] and not os.path.isdir(inference_config['content_input']):
        print(f'Saved image to {dump_dir}.')

    if should_display:
        plt.imshow(dump_img)
        plt.show()


class SequentialSubsetSampler(Sampler):
    r"""Samples elements sequentially, always in the same order from a subset defined by size.

    Arguments:
        data_source (Dataset): dataset to sample from
        subset_size: defines the subset from which to sample from
    """

    def __init__(self, data_source, subset_size):
        assert isinstance(data_source, Dataset) or isinstance(data_source, datasets.ImageFolder)
        self.data_source = data_source

        if subset_size is None:  # if None -> use the whole dataset
            subset_size = len(data_source)
        assert 0 < subset_size <= len(data_source), f'Subset size should be between (0, {len(data_source)}].'
        self.subset_size = subset_size

    def __iter__(self):
        return iter(range(self.subset_size))

    def __len__(self):
        return self.subset_size


def get_training_data_loader(training_config, should_normalize=True, is_255_range=False):
    """
        There are multiple ways to make this feed-forward NST working,
        including using 0..255 range (without any normalization) images during transformer net training,
        keeping the options if somebody wants to play and get better results.
    """
    transform_list = [transforms.Resize(training_config['image_size']),
                      transforms.CenterCrop(training_config['image_size']),
                      transforms.ToTensor()]
    if is_255_range:
        transform_list.append(transforms.Lambda(lambda x: x.mul(255)))
    if should_normalize:
        transform_list.append(transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL) if is_255_range else transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1))
    transform = transforms.Compose(transform_list)

    train_dataset = datasets.ImageFolder(training_config['dataset_path'], transform)
    sampler = SequentialSubsetSampler(train_dataset, training_config['subset_size'])
    training_config['subset_size'] = len(sampler)  # update in case it was None
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], sampler=sampler, drop_last=True)
    print(f'Using {len(train_loader)*training_config["batch_size"]*training_config["num_of_epochs"]} datapoints ({len(train_loader)*training_config["num_of_epochs"]} batches) (MS COCO images) for transformer network training.')
    return train_loader


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


# Not used atm, you'd want to use this if you choose to go with 0..255 images in the training loader
def normalize_batch(batch):
    batch /= 255.0
    mean = batch.new_tensor(IMAGENET_MEAN_1).view(-1, 1, 1)
    std = batch.new_tensor(IMAGENET_STD_1).view(-1, 1, 1)
    return (batch - mean) / std


def total_variation(img_batch):
    batch_size = img_batch.shape[0]
    return (torch.sum(torch.abs(img_batch[:, :, :, :-1] - img_batch[:, :, :, 1:])) +
            torch.sum(torch.abs(img_batch[:, :, :-1, :] - img_batch[:, :, 1:, :]))) / batch_size


def print_header(training_config):
    print(f'Learning the style of {training_config["style_img_name"]} style image.')
    print('*' * 80)
    print(f'Hyperparams: content_weight={training_config["content_weight"]}, style_weight={training_config["style_weight"]} and tv_weight={training_config["tv_weight"]}')
    print('*' * 80)

    if training_config["console_log_freq"]:
        print(f'Logging to console every {training_config["console_log_freq"]} batches.')
    else:
        print(f'Console logging disabled. Change console_log_freq if you want to use it.')

    if training_config["checkpoint_freq"]:
        print(f'Saving checkpoint models every {training_config["checkpoint_freq"]} batches.')
    else:
        print(f'Checkpoint models saving disabled.')

    if training_config['enable_tensorboard']:
        print('Tensorboard enabled.')
        print('Run "tensorboard --logdir=runs --samples_per_plugin images=50" from your conda env')
        print('Open http://localhost:6006/ in your browser and you\'re ready to use tensorboard!')
    else:
        print('Tensorboard disabled.')
    print('*' * 80)


def get_training_metadata(training_config):
    num_of_datapoints = training_config['subset_size'] * training_config['num_of_epochs']
    training_metadata = {
        "commit_hash": git.Repo(search_parent_directories=True).head.object.hexsha,
        "content_weight": training_config['content_weight'],
        "style_weight": training_config['style_weight'],
        "tv_weight": training_config['tv_weight'],
        "num_of_datapoints": num_of_datapoints
    }
    return training_metadata


def print_model_metadata(training_state):
    print('Model training metadata:')
    for key, value in training_state.items():
        if key != 'state_dict' and key != 'optimizer_state':
            print(key, ':', value)


def dir_contains_only_models(path):
    assert os.path.exists(path), f'Provided path: {path} does not exist.'
    assert os.path.isdir(path), f'Provided path: {path} is not a directory.'
    list_of_files = os.listdir(path)
    assert len(list_of_files) > 0, f'No models found, use training_script.py to train a model or download pretrained models via resource_downloader.py.'
    for f in list_of_files:
        if not (f.endswith('.pt') or f.endswith('.pth')):
            return False

    return True


# Count how many trainable weights the model has <- just for having a feeling for how big the model is
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



FILE_NAME_NUM_DIGITS = 6
SUPPORTED_VIDEO_EXTENSIONS = ['.mp4']

PERSON_CHANNEL_INDEX = 15  # segmentation stage

IMAGENET_MEAN_1 = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD_1 = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CUDA_EXCEPTION_CODE = 1
ERROR_CODE = 1

def stylized_frames_mask_combiner(relevant_directories, dump_frame_extension, other_style=None):
    # in dirs
    frames_dir = relevant_directories['frames_path']
    mask_frames_dir = relevant_directories['processed_masks_dump_path']
    stylized_frames_dir = relevant_directories['stylized_frames_path']

    # out dirs (we'll dump combined imagery here)
    dump_path = os.path.join(stylized_frames_dir, os.path.pardir)
    model_name_suffix = '_' + \
        os.path.basename(os.path.split(other_style)[
                         0]) if other_style is not None else ''
    dump_path_bkg_masked = os.path.join(
        dump_path, 'composed_background_masked' + model_name_suffix)
    dump_path_person_masked = os.path.join(
        dump_path, 'composed_person_masked' + model_name_suffix)
    os.makedirs(dump_path_bkg_masked, exist_ok=True)
    os.makedirs(dump_path_person_masked, exist_ok=True)

    # if other_stylized_frames_path exists overlay frames are differently styled frames and not original frames
    if other_style is not None:
        overlay_frames_dir = other_style
    else:
        overlay_frames_dir = frames_dir

    if len(os.listdir(dump_path_bkg_masked)) == 0 and len(os.listdir(dump_path_person_masked)) == 0:
        for cnt, (name1, name2, name3) in enumerate(zip(sorted(os.listdir(stylized_frames_dir)), sorted(os.listdir(mask_frames_dir)), sorted(os.listdir(overlay_frames_dir)))):
            # stylized original frame image
            s_img_path = os.path.join(stylized_frames_dir, name1)
            m_img_path = os.path.join(mask_frames_dir, name2)  # mask image
            o_img_path = os.path.join(
                overlay_frames_dir, name3)  # overlay image

            # load input imagery
            s_img = load_image(s_img_path)
            m_img = load_image(m_img_path, target_shape=s_img.shape[:2])
            o_img = load_image(o_img_path, target_shape=s_img.shape[:2])

            # prepare canvas imagery
            combined_img_background = s_img.copy()
            combined_img_person = s_img.copy()

            # create masks
            background_mask = m_img == 0.
            person_mask = m_img == 1.

            # apply masks
            combined_img_background[background_mask] = o_img[background_mask]
            combined_img_person[person_mask] = o_img[person_mask]

            # save combined imagery
            combined_img_background_path = os.path.join(dump_path_bkg_masked, str(
                cnt).zfill(FILE_NAME_NUM_DIGITS) + dump_frame_extension)
            combined_img_person_path = os.path.join(dump_path_person_masked, str(
                cnt).zfill(FILE_NAME_NUM_DIGITS) + dump_frame_extension)
            cv.imwrite(combined_img_background_path,
                       (combined_img_background * 255).astype(np.uint8)[:, :, ::-1])
            cv.imwrite(combined_img_person_path,
                       (combined_img_person * 255).astype(np.uint8)[:, :, ::-1])
    else:
        print('Skipping combining with masks, already done.')

    return {"dump_path_bkg_masked": dump_path_bkg_masked, "dump_path_person_masked": dump_path_person_masked}


def stylization(frames_path, model_name, img_width, stylization_batch_size):
    stylized_frames_dump_dir = os.path.join(
        frames_path, os.path.pardir, os.path.pardir, model_name.split('.')[0], 'stylized')
    os.makedirs(stylized_frames_dump_dir, exist_ok=True)

    if len(os.listdir(stylized_frames_dump_dir)) == 0:
        print('*' * 20, 'Frame stylization stage started', '*' * 20)
        stylization_script_path = os.path.join(os.path.dirname(os.path.abspath(
            __file__)), 'pytorch-nst-feedforward/stylization_script.py')

        return_code = subprocess.call(['python', stylization_script_path, '--content_input', frames_path, '--batch_size', str(
            stylization_batch_size), '--img_width', str(img_width), '--model_name', model_name, '--redirected_output', stylized_frames_dump_dir, '--verbose'])

        if return_code == CUDA_EXCEPTION_CODE:  # pytorch-nst-feedforward will special code for CUDA exception
            print(
                f'Consider making the batch_size (current = {stylization_batch_size} images) or img_width (current = {img_width} px) smaller')
            exit(ERROR_CODE)
    else:
        print('Skipping frame stylization, already done.')

    return {"stylized_frames_path": stylized_frames_dump_dir}


def create_videos(video_metadata, relevant_directories, frame_name_format, delete_source_imagery):
    stylized_frames_path = relevant_directories['stylized_frames_path']
    dump_path_bkg_masked = relevant_directories['dump_path_bkg_masked']
    dump_path_person_masked = relevant_directories['dump_path_person_masked']

    combined_img_bkg_pattern = os.path.join(
        dump_path_bkg_masked, frame_name_format)
    combined_img_person_pattern = os.path.join(
        dump_path_person_masked, frame_name_format)
    stylized_frame_pattern = os.path.join(
        stylized_frames_path, frame_name_format)

    dump_path = os.path.join(stylized_frames_path, os.path.pardir)
    combined_img_bkg_video_path = os.path.join(
        dump_path, f'{os.path.basename(dump_path_bkg_masked)}.mp4')
    combined_img_person_video_path = os.path.join(
        dump_path, f'{os.path.basename(dump_path_person_masked)}.mp4')
    stylized_frame_video_path = os.path.join(dump_path, 'stylized.mp4')

    ffmpeg = 'ffmpeg'
    if shutil.which(ffmpeg):  # if ffmpeg is in system path
        audio_path = relevant_directories['audio_path']

        def build_ffmpeg_call(img_pattern, audio_path, out_video_path):
            input_options = [
                '-r', str(video_metadata['fps']), '-i', img_pattern, '-i', audio_path]
            encoding_options = ['-c:v', 'libx264', '-crf',
                                '25', '-pix_fmt', 'yuv420p', '-c:a', 'copy']
            # libx264 won't work for odd dimensions
            pad_options = ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
            return [ffmpeg] + input_options + encoding_options + pad_options + [out_video_path]

        subprocess.call(build_ffmpeg_call(
            combined_img_bkg_pattern, audio_path, combined_img_bkg_video_path))
        subprocess.call(build_ffmpeg_call(
            combined_img_person_pattern, audio_path, combined_img_person_video_path))
        subprocess.call(build_ffmpeg_call(stylized_frame_pattern,
                        audio_path, stylized_frame_video_path))
    else:
        raise Exception(f'{ffmpeg} not found in the system path, aborting.')

    if delete_source_imagery:
        shutil.rmtree(dump_path_bkg_masked)
        shutil.rmtree(dump_path_person_masked)
        shutil.rmtree(stylized_frames_path)
        print('Deleting stylized/combined source images done.')


def post_process_mask(mask):
    """
    Helper function for automatic mask (produced by the segmentation model) cleaning using heuristics.
    """

    # step1: morphological filtering (helps splitting parts that don't belong to the person blob)
    # hardcoded 13 simply gave nice results
    kernel = np.ones((13, 13), np.uint8)
    opened_mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # step2: isolate the person component (biggest component after background)
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(opened_mask)

    if num_labels > 1:
        # step2.1: find the background component
        h, _ = labels.shape  # get mask height
        # find the most common index in the upper 10% of the image - I consider that to be the background index (heuristic)
        discriminant_subspace = labels[:int(h/10), :]
        bkg_index = np.argmax(np.bincount(discriminant_subspace.flatten()))

        # step2.2: biggest component after background is person (that's a highly probable hypothesis)
        blob_areas = []
        for i in range(0, num_labels):
            blob_areas.append(stats[i, cv.CC_STAT_AREA])
        blob_areas = list(zip(range(len(blob_areas)), blob_areas))
        # sort from biggest to smallest area components
        blob_areas.sort(key=lambda tup: tup[1], reverse=True)
        blob_areas = [a for a in blob_areas if a[0] !=
                      bkg_index]  # remove background component
        # biggest component that is not background is presumably person
        person_index = blob_areas[0][0]
        processed_mask = np.uint8((labels == person_index) * 255)

        return processed_mask
    # only 1 component found (probably background) we don't need further processing
    else:
        return opened_mask


def extract_person_masks_from_frames(processed_video_dir, frames_path, batch_size, segmentation_mask_width, mask_extension):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Currently the best segmentation model in PyTorch (officially implemented)
    segmentation_model = models.segmentation.deeplabv3_resnet101(
        pretrained=True).to(device).eval()
    # print(f'Number of trainable weights in the segmentation model: {utils.count_parameters(segmentation_model)}')

    masks_dump_path = os.path.join(processed_video_dir, 'masks')
    processed_masks_dump_path = os.path.join(
        processed_video_dir, 'processed_masks')
    os.makedirs(masks_dump_path, exist_ok=True)
    os.makedirs(processed_masks_dump_path, exist_ok=True)

    h, w = load_image(os.path.join(
        frames_path, os.listdir(frames_path)[0])).shape[:2]
    segmentation_mask_height = int(h * (segmentation_mask_width / w))
    transform = transforms.Compose([
        transforms.Resize((segmentation_mask_height, segmentation_mask_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN_1, std=IMAGENET_STD_1)
    ])
    dataset = datasets.ImageFolder(os.path.join(
        frames_path, os.path.pardir), transform=transform)
    frames_loader = DataLoader(dataset, batch_size=batch_size)

    if len(os.listdir(masks_dump_path)) == 0 and len(os.listdir(processed_masks_dump_path)) == 0:
        print('*' * 20, 'Person segmentation stage started', '*' * 20)
        with torch.no_grad():
            try:
                processed_imgs_cnt = 0
                for batch_id, (img_batch, _) in enumerate(frames_loader):
                    processed_imgs_cnt += len(img_batch)
                    print(
                        f'Processing batch {batch_id + 1} ({processed_imgs_cnt}/{len(dataset)} processed images).')
                    img_batch = img_batch.to(device)  # shape: (N, 3, H, W)
                    result_batch = segmentation_model(img_batch)['out'].to(
                        'cpu').numpy()  # shape: (N, 21, H, W) (21 - PASCAL VOC classes)
                    for j, out_cpu in enumerate(result_batch):
                        # When for the pixel position (x, y) the biggest (un-normalized) probability
                        # lies in the channel PERSON_CHANNEL_INDEX we set the mask pixel to True
                        mask = np.argmax(
                            out_cpu, axis=0) == PERSON_CHANNEL_INDEX
                        # convert from bool to [0, 255] black & white image
                        mask = np.uint8(mask * 255)

                        # simple heuristics (connected components, etc.)
                        processed_mask = post_process_mask(mask)

                        filename = str(
                            batch_id*batch_size+j).zfill(FILE_NAME_NUM_DIGITS) + mask_extension
                        cv.imwrite(os.path.join(
                            masks_dump_path, filename), mask)
                        cv.imwrite(os.path.join(
                            processed_masks_dump_path, filename), processed_mask)
            except Exception as e:
                print(str(e))
                print(
                    f'Try using smaller segmentation batch size than the current one ({batch_size} images in batch).')
                sys.exit(ERROR_CODE)
    else:
        print('Skipping mask computation, already done.')

    return {'processed_masks_dump_path': processed_masks_dump_path}

basedir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(basedir, 'model')
model_file = os.path.join(model_path, 'Generator400.pth')

# Enables this project to see packages from pytorch-nst-feedforward submodule (e.g. utils)
sys.path.append(os.path.join(os.path.dirname(
    os.path.abspath("__file__")), 'pytorch-nst-feedforward'))

# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
#the latent space is typically a n-dimensional hypersphere with each variable drawn randomly from a Gaussian distribution with a mean of zero and a standard deviation of one.
nz = 100

# Size of feature maps in generator
ngf = 64

# Number of training epochs
num_epochs = 200

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


app = Flask(__name__)
model = Generator(ngpu)
model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

current_time = datetime.datetime.now()

if torch.cuda.is_available():
    model.cuda()
    
gen_img= os.path.join(basedir, './static/generated_image')
sty_img= os.path.join(basedir, './static/styled_image')

@app.route('/')
def home():
    return render_template('index.html', )


@app.route('/generate', methods=['POST'])
def generate():
    # Set random seed for reproducibility
    manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    noise = torch.randn(128, 100, 1, 1, device=device).to(device)

    generate = model(noise)

    generate = generate[1]

    # Normalization of image pixels
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    def denorm(img_tensors):
        return img_tensors * stats[1][0] + stats[0][0]

    new_img = make_grid(denorm(generate.cpu().detach()), nrow=8)
    new_img = transforms.ToPILImage()(new_img)
    img = new_img.resize((128, 128))
    img_path = os.path.join(
        basedir, './data/clip_video/frames/frames/Generated_image.jpg')
    img.save(img_path)
    
    img_path2 = os.path.join(
        basedir, './data/generated_img/Generated_image'+str(current_time.year) + str(current_time.month)  + str(current_time.day)+'_'+ str(current_time.hour) + str(current_time.minute) + str(current_time.second)+'.jpg')
    
    img.save(img_path2)
    
    img_path3 = os.path.join(basedir, './static/generated_image/Generated_image.jpg')
    img2 = new_img.resize((64, 64))
    
    img2.save(img_path3)
    image = [i for i in os.listdir(gen_img) if i.endswith('.jpg')][0]
    return render_template('index.html', generated_img=image)


@app.route('/style', methods=['POST'])
def style():
    ptm_list = ['mosaic_4e5_e2.pth', 'starry_v3.pth','edtaonisl_9e5_33k.pth', 'candy.pth']

    ptm_idx = random.randrange(len(ptm_list))
    ptm = ptm_list[ptm_idx]
    print(f'{ptm} is the pretrained moddel currently in use...')

    # pth = ptm.split('.')[0]
    li = [f'{basedir}/data/clip_video/masks', f'{basedir}/data/clip_video/processed_masks',  f'{basedir}/data/clip_video/mosaic_4e5_e2',  f'{basedir}/data/clip_video/starry_v3', f'{basedir}/data/clip_video/edtaonisl_9e5_33k',  f'{basedir}/data/clip_video/candy']

    for x in li:
        if os.path.exists(x):
            shutil.rmtree(x)
        else:
            pass

    if __name__ == "app":
        
        frame_extension = '.png'
        # don't use .jpg here! bigger size + corrupts the binary property of the mask when loaded
        mask_extension = '.png'
        # e.g. 000023.jpg
        frame_name_format = f'%0{FILE_NAME_NUM_DIGITS}d{frame_extension}'
        data_path = os.path.join(os.path.dirname("__file__"), 'data')

        parser = argparse.ArgumentParser()
        parser.add_argument("--specific_videos", type=str,
                            help="Process only specific videos in data/", default=['video.mp4'])
        parser.add_argument('-f')
        # segmentation stage params (these 2 help with GPU VRAM problems or you can try changing the segmentation model)
        parser.add_argument("--segmentation_mask_width", type=int,
                            help="Segmentation mask size", default=128)
        parser.add_argument("--segmentation_batch_size", type=int,
                            help="Number of images in a batch (segmentation)", default=3)
        # stylization stage params
        parser.add_argument("--img_width", type=int,
                            help="Stylized images width", default=128)
        parser.add_argument("--stylization_batch_size", type=int,
                            help="Number of images in a batch (stylization)", default=3)
        parser.add_argument("--model_name", type=str,
                            help="Model binary to use for stylization", default=ptm)
        # combine stage params
        parser.add_argument("--other_style", type=str,
                            help="Model name without (like 'candy.pth') whose frames you want to use as an overlay", default=None)
        # video creation stage params
        parser.add_argument("--delete_source_imagery", type=bool,
                            help="Should delete imagery after videos are created", default=False)
        # args = parser.parse_args()
        args, _ = parser.parse_known_args()

        # Basic error checking regarding NST submodule and model placement
        nst_submodule_path = os.path.join(
            os.path.dirname("__file__"), 'pytorch-nst-feedforward')
        assert os.path.exists(
            nst_submodule_path), 'Please pull the pytorch-nst-feedforward submodule to use this project.'
        model_path = os.path.join(
            nst_submodule_path, 'models', 'binaries', args.model_name)
        assert os.path.exists(
            model_path), f'Could not find {model_path}. Make sure to place pretrained models in there.'
        ffmpeg = 'ffmpeg'
        assert shutil.which(
            ffmpeg), f'{ffmpeg} not found in the system path. Please add it before running this script.'
        #
        # For every video located under data/ run this pipeline
        #
        for element in os.listdir(data_path):
            maybe_video_path = os.path.join(data_path, element)
            if os.path.isfile(maybe_video_path) and os.path.splitext(maybe_video_path)[1].lower() in SUPPORTED_VIDEO_EXTENSIONS:
                video_path = maybe_video_path
                video_name = os.path.basename(video_path).split('.')[0]

                if args.specific_videos is not None and os.path.basename(video_path) not in args.specific_videos:
                    print(
                        f'Video {os.path.basename(video_path)} not in the specified list of videos {args.specific_videos}. Skipping.')
                    continue

                print(
                    '*' * 20, f'Processing video clip: {os.path.basename(video_path)}', '*' * 20)

                # Create destination directory for this video where everything related to that video will be stored
                processed_video_dir = os.path.join(data_path, 'clip_' + video_name)
                os.makedirs(processed_video_dir, exist_ok=True)

                frames_path = os.path.join(processed_video_dir, 'frames', 'frames')
                os.makedirs(frames_path, exist_ok=True)

                cap = cv.VideoCapture(video_path)
                fps = int(cap.get(cv.CAP_PROP_FPS))

                out_frame_pattern = os.path.join(frames_path, frame_name_format)
                audio_dump_path = os.path.join(
                    processed_video_dir, video_name + '.aac')
                # step1: Extract frames from the videos as well as audio file
                if len(os.listdir(frames_path)) == 0:
                    subprocess.call([ffmpeg, '-i', video_path, '-r', str(fps), '-start_number',
                                    '0', '-qscale:v', '2', out_frame_pattern, '-c:a', 'copy', audio_dump_path])
                else:
                    print('Skip splitting video into frames and audio, already done.')
                print('Stage 1/5 done (split video into frames and audio file).')
                # step2: Extract person segmentation mask from frames
                ts = time.time()
                mask_dirs = extract_person_masks_from_frames(
                    processed_video_dir, frames_path, args.segmentation_batch_size, args.segmentation_mask_width, mask_extension)
                print('Stage 2/5 done (create person segmentation masks).')
                print(
                    f'Time elapsed computing masks: {(time.time() - ts):.3f} [s].')
                # step3: Stylize dumped video frames
                ts = time.time()
                style_dir = stylization(
                    frames_path, args.model_name, args.img_width, args.stylization_batch_size)
                print('Stage 3/5 done (stylize dumped video frames).')
                print(
                    f'Time elapsed stylizing imagery: {(time.time() - ts):.3f} [s].')
                # step4: Combine stylized frames with masks
                relevant_directories = {
                    'frames_path': frames_path, 'audio_path': audio_dump_path}
                relevant_directories.update(mask_dirs)
                relevant_directories.update(style_dir)

                ts = time.time()
                if args.other_style is not None:
                    args.other_style = args.other_style.split(
                        '.')[0] if args.other_style.endswith('.pth') else args.other_style
                    other_style = os.path.join(
                        processed_video_dir, args.other_style, 'stylized')
                    assert os.path.exists(other_style) and os.path.isdir(
                        other_style), f'You first need to create stylized frames for the model {args.other_style}.pth so that you can use it as the other style for this model {args.model_name}.'
                else:
                    other_style = None

                combined_dirs = stylized_frames_mask_combiner(
                    relevant_directories, frame_extension, other_style)
                print('Stage 4/5 done (combine masks with stylized frames).')
                print(
                    f'Time elapsed masking stylized imagery: {(time.time() - ts):.3f} [s].')
                # step5: Create videos
                relevant_directories.update(combined_dirs)

                video_metadata = {'fps': fps}
                create_videos(video_metadata, relevant_directories,
                            frame_name_format, args.delete_source_imagery)
                print('Stage 5/5 done (create videos from overlayed stylized frames).')
    else:
        print(__name__, type(__name__))
    
    new_ptm = ptm.split('.')[0]
    check_img = os.chdir(
        f'{basedir}/data/clip_video/{new_ptm}/composed_background_masked')
    for i in os.listdir(check_img):
        if i.endswith('.png'):
            styled_img = Image.open('{}/data/clip_video/{}/composed_background_masked/000000.png'.format(basedir,new_ptm))
        else:
            return render_template('index.html', styled_img="The image reslution is too low to be styled.")
            
    styled_img = styled_img.resize((64, 64))
    
    # styled_img_path = os.chdir(f'{basedir}/data/styled_generated_img/Generated_image'+str(current_time.year) + str(
    #     current_time.month) + str(current_time.day)+'_' + str(current_time.hour) + str(current_time.minute) + str(current_time.second)+'.png')
    styled_img_path = os.path.join(
        basedir, './data/styled_generated_img/Generated_Style_image'+str(current_time.year) + str(current_time.month)  + str(current_time.day)+'_'+ str(current_time.hour) + str(current_time.minute) + str(current_time.second)+'.png')
    
    
    styled_img.save(styled_img_path)
    
    img_path4 = os.path.join(basedir, './static/styled_image/styled_image.png')
    styled_img.save(img_path4)
    
    image2 = [i for i in os.listdir(sty_img) if i.endswith('.png')][0]
    
    return render_template('index.html', styled_img=image2)
     
if __name__ == "__main__":
    app.run(debug=True)
