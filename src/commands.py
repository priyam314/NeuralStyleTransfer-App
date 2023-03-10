from abc import ABC, abstractmethod
import numpy as np
import utils
import torch


class Tuning(ABC):

    @abstractmethod
    def Image(self, image):
        pass


class TuningReconstruction(Tuning):

    def __init__(self, model, optimizer, target_representation,
                 content_feature_maps_index, style_feature_maps_indices):

        self.model = model
        self.optimizer = optimizer
        self.target_representation = target_representation
        self.content_feature_maps_index = content_feature_maps_index
        self.style_feature_maps_indices = style_feature_maps_indices

    def Image(self, image):

        # Finds the current representation
        set_of_feature_maps = self.model(image)
        if utils.yamlGet('reconstruct') == 'Content':
            current_representation = set_of_feature_maps[
                self.content_feature_maps_index].squeeze(axis=0)
        elif utils.yamlGet('reconstruct') == 'Style':
            current_representation = [
                utils.gram_matrix(fmaps)
                for i, fmaps in enumerate(set_of_feature_maps)
                if i in self.style_feature_maps_indices
            ]

        loss = 0.0

        if utils.yamlGet('reconstruct') == 'Content':
            loss = torch.nn.MSELoss(reduction='mean')(
                self.target_representation, current_representation)
        elif utils.yamlGet('reconstruct') == 'Style':
            for gram_gt, gram_hat in zip(self.target_representation,
                                         current_representation):
                loss += (1 / len(self.target_representation)) * \
                    torch.nn.MSELoss(
                    reduction='sum')(gram_gt[0], gram_hat[0])

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item(), current_representation


class Reconstruct(ABC):

    @abstractmethod
    def Visualize(self):
        pass


class ContentReconstruct(Reconstruct):
    """
        tcr -> target_content_representation
    """

    def __init__(self, feature_maps):
        self.fm = feature_maps
        self.tcr = self.fm['set_of_feature_maps'][
            self.fm['content_feature_maps_index_name'][0]].squeeze(axis=0)
        self.nfm = self.tcr.size()[0]

    def Visualize(self):
        for i in range(self.nfm):
            feature_map = self.tcr[i].to('cpu').numpy()
            feature_map = np.uint8(utils.get_uint8_range(feature_map))
            # plt.imshow(feature_map)
            # plt.title(
            #     f'Feature map {i+1}/{num_of_feature_maps} from layer'
            #     f' {content_feature_maps_index_name[1]} '
            #     f'(model={config["model"]}) for'
            #     f' {config["content_img_name"]} image.'
            # )
            # plt.show()
            filename = f'fm_{config["model"]}_{content_feature_maps_index_name[1]}_{str(i).zfill(config["img_format"][0])}{config["img_format"][1]}'
            utils.save_image(feature_map, os.path.join(dump_path, filename))


class StyleReconstruct(Reconstruct):
    pass


class Invoker:
    pass
