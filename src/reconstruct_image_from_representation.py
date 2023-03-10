import os
import src.utils.utils as utils
from src.utils.video_utils import create_video_from_intermediate_results
import torch
from torch.autograd import Variable
from torch.optim import Adam, LBFGS
import numpy as np


def make_tuning_step(optimizer, config):

    def tuning_step(optimizing_img):

        config.current_set_of_feature_maps = config.neural_net(optimizing_img)
        loss, config.current_representation = utils.getCurrentData(config)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), config.current_representation

    return tuning_step


def reconstruct_image_from_representation():

    dump_path = os.path.join(os.path.dirname(__file__), "data/reconstruct")
    config = utils.Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img, img_path = utils.getImageAndPath(device)
    white_noise_img = np.random.uniform(-90., 90.,
                                        img.shape).astype(np.float32)
    init_img = torch.from_numpy(white_noise_img).float().to(device)
    optimizing_img = Variable(init_img, requires_grad=True)

    # indices pick relevant feature maps (say conv4_1, relu1_1, etc.)
    output = list(utils.prepare_model(device))
    config.neural_net = output[0]
    content_feature_maps_index_name = output[1]
    style_feature_maps_indices_names = output[2]

    config.content_feature_maps_index = content_feature_maps_index_name[0]
    config.style_feature_maps_indices = style_feature_maps_indices_names[0]

    config.current_set_of_feature_maps = config.neural_net(img)

    config.target_content_representation = config.current_set_of_feature_maps[
        config.content_feature_maps_index].squeeze(axis=0)
    config.target_style_representation = [
        utils.gram_matrix(fmaps)
        for i, fmaps in enumerate(config.current_set_of_feature_maps)
        if i in config.style_feature_maps_indices
    ]

    if utils.yamlGet('reconstruct') == "Content":
        config.target_representation = config.target_content_representation
        num_of_feature_maps = config.target_content_representation.size()[0]
        for i in range(num_of_feature_maps):
            feature_map = config.target_content_representation[i].to(
                'cpu').numpy()
            feature_map = np.uint8(utils.get_uint8_range(feature_map))
            # filename = f'fm_{config["model"]}_{content_feature_maps_index_name[1]}_{str(i).zfill(config["img_format"][0])}{config["img_format"][1]}'
            # utils.save_image(feature_map, os.path.join(dump_path, filename))

    elif utils.yamlGet('reconstruct') == "Style":
        config.target_representation = config.target_style_representation
        num_of_gram_matrices = len(config.target_style_representation)
        for i in range(num_of_gram_matrices):
            Gram_matrix = config.target_style_representation[i].squeeze(
                axis=0).to('cpu').numpy()
            Gram_matrix = np.uint8(utils.get_uint8_range(Gram_matrix))
            # filename = f'gram_{config["model"]}_{style_feature_maps_indices_names[1][i]}_{str(i).zfill(config["img_format"][0])}{config["img_format"][1]}'
            # utils.save_image(Gram_matrix, os.path.join(dump_path, filename))

    if utils.yamlGet('optimizer') == 'Adam':
        optimizer = Adam((optimizing_img, ), lr=utils.yamlGet('learning_rate'))
        tuning_step = make_tuning_step(optimizer, config)
        for it in range(utils.yamlGet('optimizer')):
            tuning_step(optimizing_img)
            with torch.no_grad():
                utils.save_optimizing_image(optimizing_img, dump_path, it)

    elif utils.yamlGet('optimizer') == 'LBFGS':
        optimizer = LBFGS((optimizing_img, ),
                          max_iter=utils.yamlGet('optimizer'),
                          line_search_fn='strong_wolfe')
        cnt = 0

        def closure():
            nonlocal cnt
            loss = utils.getLBFGSReconstructLoss(config, optimizing_img)
            loss.backward()
            with torch.no_grad():
                utils.save_optimizing_image(optimizing_img, dump_path, cnt)
                cnt += 1
            return loss

        optimizer.step(closure)

    return dump_path


if __name__ == "__main__":

    # reconstruct style or content image purely from their representation
    results_path = reconstruct_image_from_representation()

    create_video_from_intermediate_results(results_path)
