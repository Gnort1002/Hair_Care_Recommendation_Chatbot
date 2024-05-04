# libraries
import cv2
import numpy as np
import torch
import os
import warnings
warnings.filterwarnings('ignore')
import argparse
import sys

sys.path.append('src/swinface_project/')
from model import build_model

def get_swinface_parser():
    parser = argparse.ArgumentParser(description='SwingFace')
    parser.add_argument('--swinface_network', type=str, default='swin_t', help='network')
    parser.add_argument('--fam_kernel_size', type=int, default=3, help='fam_kernel_size')
    parser.add_argument('--fam_in_chans', type=int, default=2112, help='fam_in_chans')
    parser.add_argument('--fam_conv_shared', type=bool, default=False, help='fam_conv_shared')
    parser.add_argument('--fam_conv_mode', type=str, default='split', help='fam_conv_mode')
    parser.add_argument('--fam_channel_attention', type=str, default='CBAM', help='fam_channel_attention')
    parser.add_argument('--fam_spatial_attention', type=str, default=None, help='fam_spatial_attention')
    parser.add_argument('--fam_pooling', type=str, default='max', help='fam_pooling')
    parser.add_argument('--fam_la_num_list', type=list, default=[2 for j in range(11)], help='fam_la_num_list')
    parser.add_argument('--fam_feature', type=str, default='all', help='fam_feature')
    parser.add_argument('--fam', type=str, default='3x3_2112_F_s_C_N_max', help='fam')
    parser.add_argument('--swinface_embedding_size', type=int, default=512, help='embedding_size')
    parser.add_argument('--swinface_weight_path', type=str, default='src/swinface_project/checkpoint_step_79999_gpu_0.pt', help='weight_path')
    return parser


class SwinFaceCfg:
    def __init__(self, args):
        self.network = args.swinface_network
        self.fam_kernel_size = args.fam_kernel_size
        self.fam_in_chans = args.fam_in_chans
        self.fam_conv_shared = args.fam_conv_shared
        self.fam_conv_mode = args.fam_conv_mode
        self.fam_channel_attention = args.fam_channel_attention
        self.fam_spatial_attention = args.fam_spatial_attention
        self.fam_pooling = args.fam_pooling
        self.fam_la_num_list = args.fam_la_num_list
        self.fam_feature = args.fam_feature
        self.fam = args.fam
        self.embedding_size = args.swinface_embedding_size
        self.weight_path = args.swinface_weight_path


class SwinFace:
    """
    SwinFace class for face recognition
    """
    def __init__(self, cfg):
        self.weight_path = cfg.weight_path
        self.model = self.create_swinface_model(cfg, self.weight_path)
    
    def create_swinface_model(self, cfg, weight_path):
        model = build_model(cfg)
        dict_checkpoint = torch.load(weight_path)
        model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
        model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
        model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
        model.om.load_state_dict(dict_checkpoint["state_dict_om"])

        model.eval()
        return model 

    @torch.no_grad()
    def inference(self, model, img):
        if img is None:
            img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
        else:

            img = cv2.resize(img, (112, 112))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        
        output = model(img)

        return output["Recognition"].numpy()

    # facial embedding for all images in directory (jpg and png)
    def inference_in_dir_path(self, model, dir_path):
        files = os.listdir(dir_path)
        files = [f for f in files if f.endswith(".jpg") or f.endswith(".png")]

        embeddings = []
        for f in files:
            img = cv2.imread(os.path.join(dir_path, f))
            facial_embedding = self.inference(model, img)
            embeddings.append(facial_embedding)

        return files, torch.stack(embeddings, dim=0)
    
    # cosine distance between two numpy embeddings
    def compare(self, embedding1, embedding2):
        embedding1, embedding2 = embedding1.squeeze(), embedding2.squeeze()
        return 1- np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def compare_with_db(self, embedding_1, embeeings_db):
        list_of_dist = []
        for embedding_2 in embeeings_db:
            list_of_dist.append(self.compare(embedding_1, embedding_2))
        return np.array(list_of_dist)
            

    
if __name__ == '__main__':
    args = get_swinface_parser().parse_args()
    cfg = SwinFaceCfg(args)
    weight_path = "swinface_project/checkpoint_step_79999_gpu_0.pt"
    swinface = SwinFace(cfg)
    image = cv2.imread("img/001.jpg")
    facial_embedding = swinface.inference(swinface.model, image)
    print(facial_embedding.shape)
