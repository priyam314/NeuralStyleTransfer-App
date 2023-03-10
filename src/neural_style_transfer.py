import os
import src.utils.utils as utils
from src.utils.video_utils import create_video_from_intermediate_results
import torch
from torch import nn
from torch.optim import Adam, LBFGS
from torch.autograd import Variable


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    
    def forward(self, current):
        return nn.MSELoss(reduction='mean')(self.target, current)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.loss = 0.0

    def forward(self, x, y):
        for gram_gt, gram_hat in zip(x, y):
            self.loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
        self.loss /= len(x)
        return self.loss


class Build(nn.Module):
    def __init__(
            self,
            config,
            target_content_representation,
            target_style_representation,
        ):
        super(Build, self).__init__()
        self.current_set_of_feature_maps = None
        self.current_content_representation = None
        self.current_Style_representation = None
        self.config = config
        self.target_content_representation = target_content_representation
        self.target_style_representation = target_style_representation

    def forward(self, model, x):
        self.current_set_of_feature_maps = model(x)

        self.current_content_representation = self.current_set_of_feature_maps[
            self.config.content_feature_maps_index].squeeze(axis=0)
        self.current_style_representation = [
            utils.gram_matrix(x)
            for cnt, x in enumerate(self.current_set_of_feature_maps)
            if cnt in self.config.style_feature_maps_indices
        ]
        content_loss = ContentLoss(self.target_content_representation)(
            self.current_content_representation)
        style_loss = StyleLoss()(
            self.target_style_representation, 
            self.current_style_representation)
        tv_loss = TotalVariationLoss(x)()

        return Loss()(content_loss, style_loss, tv_loss)


class TotalVariationLoss(nn.Module):
    def __init__(self, y):
        super(TotalVariationLoss, self).__init__()
        self.first = torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
        self.second = torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))
    
    def forward(self):
        return self.first + self.second
           
    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, x, y, z):
        return utils.yamlGet("contentWeight") * x + utils.yamlGet("styleWeight") * y + utils.yamlGet("totalVariationWeight") * z


def neural_style_transfer():

    dump_path = os.path.join(os.path.dirname(__file__), "data/transfer")
    config = utils.Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    content_img, style_img, init_img = utils.Images().getImages(device)
    optimizing_img = Variable(init_img, requires_grad=True)

    output = list(utils.prepare_model(device))
    neural_net = output[0]
    content_feature_maps_index_name = output[1]
    style_feature_maps_indices_names = output[2]

    config.content_feature_maps_index = content_feature_maps_index_name[0]
    config.style_feature_maps_indices = style_feature_maps_indices_names[0]

    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)

    target_content_representation = content_img_set_of_feature_maps[
        config.content_feature_maps_index].squeeze(axis=0)
    target_style_representation = [
        utils.gram_matrix(x)
        for cnt, x in enumerate(style_img_set_of_feature_maps)
        if cnt in config.style_feature_maps_indices
    ]

    if utils.yamlGet('optimizer') == 'Adam':
        optimizer = Adam((optimizing_img, ), lr=utils.yamlGet('learning_rate'))
        for cnt in range(utils.yamlGet("iterations")):

            total_loss = Build(config, target_content_representation, 
                               target_style_representation)(neural_net, 
                                                            optimizing_img)

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                utils.save_optimizing_image(optimizing_img, dump_path, cnt)

    elif utils.yamlGet('optimizer') == 'LBFGS':
        optimizer = LBFGS((optimizing_img, ),
                          max_iter=utils.yamlGet('iterations'),
                          line_search_fn='strong_wolfe')

        def closure():
            total_loss, _, _, _ = build_loss(
                neural_net, optimizing_img, target_content_representation,
                target_style_representation, config)
            total_loss.backward()
            optimizer.zero_grad()
            with torch.no_grad():
                utils.save_optimizing_image(optimizing_img, dump_path, cnt)
            return total_loss

        for cnt in range(utils.yamlGet("iterations")):
            optimizer.step(closure)

    create_video_from_intermediate_results(dump_path)


# some values of weights that worked for figures.jpg, vg_starry_night.jpg
# (starting point for finding good images)
# once you understand what each one does it gets really easy -> also see
# README.md

# lbfgs, content init -> (cw, sw, tv) = (1e5, 3e4, 1e0)
# lbfgs, style   init -> (cw, sw, tv) = (1e5, 1e1, 1e-1)
# lbfgs, random  init -> (cw, sw, tv) = (1e5, 1e3, 1e0)

# adam, content init -> (cw, sw, tv, lr) = (1e5, 1e5, 1e-1, 1e1)
# adam, style   init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)
# adam, random  init -> (cw, sw, tv, lr) = (1e5, 1e2, 1e-1, 1e1)

# original NST Neural Style Transfer) algorithm (Gatys et al.)
# results_path = neural_style_transfer()
# create_video_from_intermediate_results(results_path)
