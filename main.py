from nst import NeuralStyleTransfer
import argparse
import os
from pprint import pprint
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    
    # fps
    parser.add_argument("-f", "--fps", default=30, type=int,
                       help="frames per second to create video (default: %(default)s)")

    # size
    parser.add_argument("--size", default=128, type=int, choices=[32, 64, 128, 256, 512, 1024], 
                       help="output image/video will not have size similar to content image but square (default: %(default)s)")
    
    # saving_frequency
    parser.add_argument("--sav_freq", default=10, type=int, 
                       help="save image after every nth image (default: %(default)s)")
    
    ######################################## Weights
    # alpha
    parser.add_argument("-a", "--alpha", default=5.0, type=float,
                       help="content weight (default: %(default)s)")
    
    # beta
    parser.add_argument("-b", "--beta", default=7000.0, type=float,
                       help="style weight (default: %(default)s)")
    
    # gamma
    parser.add_argument("-g", "--gamma", default=1.2, type=float,
                       help="total variation loss (default: %(default)s)")
    
    # style-weights
    parser.add_argument("-w", "--style_weights", default=[1e3/n**2 for n in [16.0,32.0,128.0,256.0,512.0]], nargs=5, type=float,
                       help="style weights layer wise (default: %(default)s)")
    
    ######################################## Optimizer
    # lr
    parser.add_argument("-l", "--lr", default=0.06, type=float,
                       help="learning rate (default: %(default)s)")
    
    # optimizer
    parser.add_argument("-o", "--optimizer", choices=["Adam", "LBFGS"], default="LBFGS", 
                       help="optimizer (default: %(default)s)")
    
    # iterations
    parser.add_argument("-e", "--iterations", default=2500, type=int,
                       help="number of iterations to update canvas (default: %(default)s)")
    
    ######################################## Layers
    # content-layers
    parser.add_argument("--content_layers", choices=range(5), type=int, nargs="*",
                       help="choose the index of content layer you want output from (default: %(default)s)")
    
    # style-layers
    parser.add_argument("--style_layers", choices=range(5), type=int, nargs="*",
                       help="applicable if you want to visualize any particular layer (default: %(default)s)")
    
    ######################################## Image
    # init-image
    parser.add_argument("-i", "--init_image", default="noise", choices=["noise", "content", "style"],
                       help="initiation of canvas for reconstruction (default: %(default)s)")
    
    # content-image
    parser.add_argument("--content_image", default=os.path.join(os.path.dirname(__file__), "data/content.jpg"), nargs="?",
                        help="path to content image (default: %(default)s)")
    
    # style-image
    parser.add_argument("--style_image", default=os.path.join(os.path.dirname(__file__), "data/style.jpg"), nargs="?",
                        help="path to style image (default: %(default)s)")
    
    ######################################## Type
    # should-reconstruct
    parser.add_argument("--reconstruct", action="store_true",
                       help="True is you want to transfer content and style semantic over canvas, but False if you want to visualize either content or style semantic (default: %(default)s)")
    
    # visualize
    parser.add_argument("-v", "--visualize", default="none", choices=["content", "style", "both", "none"],
                       help="what do you want to visualize (default: %(default)s)")
    
    # use-conv
    parser.add_argument("-r", "--use_type", default="conv", type=str, choices=["conv", "relu"],
                       help="use conv layer outputs or relu layer outputs for loss calculation (default: %(default)s)")
    
    config = parser.parse_args()
    pprint (vars(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NST = NeuralStyleTransfer(config, device)
    NST.Execute()
    
    