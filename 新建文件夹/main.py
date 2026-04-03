import os
import argparse
import torch
import numpy as np
from load_data import load_dataset
from load_model import build_model
from train import train_model
from utils import load_pretrained_teacher
from evaluation import evaluate_classification, evaluate_retrieval


parser = argparse.ArgumentParser()
# dataset setting
parser.add_argument('--dataset', type=str, default='flickr-30k', choices=['mmimdb', 'vqav2', 'flickr-30k', 'ms-coco'], help='name of the dataset')
# model setting
parser.add_argument('--teacher_model_1', default='clip-ViT-B-16', choices=['clip-ViT-B-16', 'clip-ViT-L-14', 'clip-RN101'], help='name of the model')
parser.add_argument('--teacher_model_2', default='clip-ViT-L-14', choices=['clip-ViT-B-16', 'clip-ViT-L-14', 'clip-RN101'], help='name of the model')
parser.add_argument('--student_model', default='clip-ViT-B-32', choices=['clip-ViT-B-32', 'clip-RN50', 'resnet-bert', 'vit-bert'], help='name of the model')
parser.add_argument('--project_dim', type=int, default=512, choices=[64, 128, 256, 512, 1024, 2048])
# training setting
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
#
parser.add_argument('--distiller', type=str, default='hrad', choices=['none', 'msd', 'albef', 'dsmd', 'kdmcse', 'dclip', 'g2d', 'hrad'], help='name of the distiller')
# experiment detail
parser.add_argument('--gpu', dest='gpu', type=str, default='2', choices=['0', '1', '2', '3','4', '5', '6', '7'])
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

train_loader, val_loader, test_loader, num_category = load_dataset(dataset=args.dataset, batch_size=args.batch_size)

teacher_model_1 = build_model(args.teacher_model_1, num_category, args.project_dim)
teacher_model_2 = build_model(args.teacher_model_2, num_category, args.project_dim)

#load_pretrained_teacher(teacher_model_1, args.teacher_model_1, args.dataset)
#load_pretrained_teacher(teacher_model_2, args.teacher_model_2, args.dataset)
load_pretrained_teacher(teacher_model_1, args.teacher_model_1, args.dataset, args.project_dim)
load_pretrained_teacher(teacher_model_2, args.teacher_model_2, args.dataset, args.project_dim)
student_model = build_model(args.student_model, num_category, args.project_dim)

'''
if num_category:
    eval_result = evaluate_classification(student_model, val_loader)
else:
    eval_result = evaluate_retrieval(student_model, val_loader)
print(eval_result)
'''

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model = train_model(args, train_loader, val_loader, num_category, teacher_model_1, teacher_model_2, student_model)