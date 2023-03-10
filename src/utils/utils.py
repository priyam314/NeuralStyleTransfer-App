import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
import os
import yaml
import PIL.Image as Image
from src.models.definitions.vgg_nets import Vgg16, Vgg19, Vgg16Experimental

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]


def load_image(img_path, target_shape=None):
    if not os.path.exists(img_path):
        raise Exception(f'Path does not exist: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]
    if target_shape is not None:  # resize section
        current_height, current_width = img.shape[:2]
        new_height = target_shape
        new_width = int(current_width * (new_height / current_height))
        img = cv.resize(img, (new_width, new_height),
                        interpolation=cv.INTER_CUBIC)

    # this need to go after resizing - otherwise cv.resize will push values outside of [0,1] range
    img = img.astype(np.float32)  # convert from uint8 to float32
    img /= 255.0  # get to [0, 1] range
    return img


def getInitImage(content_img, style_img, device):

    if yamlGet("initImage") == 'White Noise Image':
        white_noise_img = np.random.uniform(
            -90., 90., content_img.shape).astype(np.float32)
        init_img = torch.from_numpy(white_noise_img).float().to(device)

    elif yamlGet("initImage") == 'Gaussian Noise Image':
        gaussian_noise_img = np.random.normal(loc=0,
                                              scale=90.,
                                              size=content_img.shape).astype(
                                                  np.float32)
        init_img = torch.from_numpy(gaussian_noise_img).float().to(device)

    elif yamlGet("initImage") == 'Content':
        init_img = content_img

    else:
        # init image has same dimension as content image - this is a hard constraint
        # feature maps need to be of same size for content image and init image
        style_img_resized = prepare_img(style_img,
                                        np.asarray(content_img.shape[2:]),
                                        device)
        init_img = style_img_resized
    return init_img


def prepare_img(img_path, target_shape, device):
    img = load_image(img_path, target_shape=target_shape)

    # normalize using ImageNet's mean
    # [0, 255] range worked much better for me than [0, 1] range (even though PyTorch models were trained on latter)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)
    ])

    img = transform(img).to(device).unsqueeze(0)

    return img


def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img, ) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1]
               )  # [:, :, ::-1] converts rgb into bgr (opencv contraint...)


def save_optimizing_image(optimizing_img, dump_path, img_id):
    img_format = (4, '.jpg')
    saving_freq = yamlGet('reprSavFreq')
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(
        out_img, 0,
        2)  # swap channel from 1st to 3rd position: ch, _, _ -> _, _, chr

    if img_id == yamlGet('iterations') - 1 or \
       (saving_freq > 0 and img_id % saving_freq == 0):

        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] \
            if saving_freq != -1 else None
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])
        print(f"{out_img_name} written to {dump_path}")

    # if should_display:
    #     plt.imshow(np.uint8(get_uint8_range(out_img)))
    #     plt.show()


def get_uint8_range(x):
    if isinstance(x, np.ndarray):
        x -= np.min(x)
        x /= np.max(x)
        x *= 255
        return x
    else:
        raise ValueError(f'Expected numpy array got {type(x)}')


def prepare_model(device):

    model = yamlGet('model')
    if model == 'VGG16':
        model = Vgg16(requires_grad=False, show_progress=True)
    elif model == 'VGG16-Experimental':
        model = Vgg16Experimental(requires_grad=False, show_progress=True)
    elif model == 'VGG19':
        model = Vgg19(requires_grad=False, show_progress=True)
    else:
        raise ValueError(f'{model} not supported.')

    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = list(model.layer_names.keys())

    content_fms_index_name = (content_feature_maps_index,
                              layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(
        device).eval(), content_fms_index_name, style_fms_indices_names


def yamlSet(key, value):
    with open('src/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config[key] = value
    with open('src/config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def yamlGet(key):
    with open('src/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config[key]


def save_numpy_array_as_jpg(array, name):
    image = Image.fromarray(array)
    image.save("src/data/" + str(name) + '.jpg')
    return "src/data/" + str(name) + '.jpg'


def gram_matrix(x, should_normalize=True):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram


def total_variation(y):
    return 


def getImageAndPath(device):

    if yamlGet('reconstruct') == 'Content':
        img_path = yamlGet('contentPath')
    elif yamlGet('reconstruct') == 'Style':
        img_path = yamlGet('stylePath')

    img = prepare_img(img_path, yamlGet('height'), device)

    return img, img_path


def getContentCurrentData(config):
    current_representation = config.current_set_of_feature_maps[
        config.content_feature_maps_index].squeeze(axis=0)
    loss = torch.nn.MSELoss(reduction='mean')(config.target_representation,
                                              current_representation)
    return loss, current_representation


def getStyleCurrentData(config):
    current_representation = [
        gram_matrix(x)
        for cnt, x in enumerate(config.current_set_of_feature_maps)
        if cnt in config.style_feature_maps_indices
    ]
    loss = 0.0
    for gram_gt, gram_hat in zip(config.target_style_representation,
                                 current_representation):
        loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])

    loss /= len(config.target_style_representation)
    return loss, current_representation


def getCurrentData(config):
    if yamlGet('reconstruct') == 'Content':
        return getContentCurrentData(config)

    elif yamlGet('reconstruct') == 'Style':
        return getStyleCurrentData(config)


def getLBFGSReconstructLoss(config, optimizing_img):

    loss = 0.0

    if yamlGet('reconstruct') == 'Content':
        loss = torch.nn.MSELoss(reduction='mean')(
            config.target_content_representation,
            config.neural_net(optimizing_img)[
                config.content_feature_maps_index].squeeze(axis=0))

    else:
        config.current_set_of_feature_maps = config.neural_net(optimizing_img)
        current_style_representation = [
            gram_matrix(fmaps)
            for i, fmaps in enumerate(config.current_set_of_feature_maps)
            if i in config.style_feature_maps_indices
        ]
        for gram_gt, gram_hat in zip(config.target_style_representation,
                                     current_style_representation):

            loss += (1 / len(config.target_style_representation)) * \
                torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])

    return loss


class Config:

    def __init__(self):
        self.target_representation = 0
        self.target_content_representation = 0
        self.target_style_representation = 0
        self.content_feature_maps_index = 0
        self.style_feature_maps_indices = 0
        self.current_set_of_feature_maps = 0
        self.current_representation = 0
        self.neural_net = 0


class Images:

    def getImages(self, device):

        return [
            self.__getContentImage(device),
            self.__getStyleImage(device),
            self.__getInitImage(device),
        ]

    def __getContentImage(self, device):
        return prepare_img(yamlGet('contentPath'), yamlGet('height'), device)

    def __getStyleImage(self, device):
        return prepare_img(yamlGet('stylePath'), yamlGet('height'), device)

    def __getInitImage(self, device):
        return getInitImage(self.__getContentImage(device),
                            self.__getStyleImage(device), device)


def clearDir():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    reconstructPath = os.path.join(path, "reconstruct")
    transferPath = os.path.join(path, "transfer")
    for transfer_file in os.scandir(transferPath):
        os.remove(transfer_file)
    for reconstruct_file in os.scandir(reconstructPath):
        os.remove(reconstruct_file)
