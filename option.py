import argparse, torch

parser = argparse.ArgumentParser(description='Condformer with LoNPE')

# Hardware specifications
parser.add_argument('--cuda', default=True, action='store_true', help='Use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument("--compile", default=False, action="store_true", help='use torch.compile from PyTorch 2.0')
parser.add_argument('--GPUs', type=int, default=[0], help='GPU IDs')

# data specifications
parser.add_argument('--method', type=str, default='Condformer', help='train dataset name')
parser.add_argument('--dir_data', type=str, default='Datasets/', help='dataset directory')
parser.add_argument('--data_train', type=str, default=['SIDD_Medium_Srgb'], help='train dataset name')
parser.add_argument('--data_test', type=str, default=['SIDD'], help='train dataset name')
parser.add_argument('--n_train', type=int, default=[320], help='number of training samples')
# parser.add_argument('--data_train', type=str, default=['DF2K', 'WED', 'BSD'], help='train dataset name')
# parser.add_argument('--data_test', type=str, default=['CBSD68', 'Kodak24', 'Urban100'], help='train dataset name')
# parser.add_argument('--n_train', type=int, default=[3450, 4744, 400], help='number of training samples')
parser.add_argument('--n_colors', type=int, default=3, help='channels for samples')
parser.add_argument('--value_range', type=int, default=255, help='maximum value of RGB')
parser.add_argument("--shuffle", default=False, action="store_true", help='shuffling the training set')
parser.add_argument("--store_in_ram", default=True, action="store_true", help='store the training set in RAM for acceleration')
parser.add_argument("--save_img", default=True, help='save image when evaluation')
parser.add_argument("--use_matlab", default=False, help='apply matlab engine for PSNR and SSIM calculation')

# Training/Testing specifications
parser.add_argument('--train', type=str, default='test', help='choose train | test | complexity')
parser.add_argument('--iter_epoch', type=int, default=1000, help='iteration in each epoch')
parser.add_argument('--start_epoch', default=-1, type=int, help='start epoch for training')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--resume_Condformer', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--resume_LoNPE', type=str, default='', help='load the model from the specified epoch')
parser.add_argument('--model_path', type=str, default='', help='path to save model')
parser.add_argument('--patch_size', type=int, default=128, help='patch size')
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')

# Optimization specifications
parser.add_argument('--lambda_NE', type=float, default=0, help='>0: train LoNPE net'
                                                                '=0: use pre-trained LoNPE net'
                                                                '<0: do not use prior'
                                                                'np.Inf: use GT prior from LoNPE algorithm')
parser.add_argument('--lambda_DN', type=float, default=1, help='')
parser.add_argument('--optimizer', default={'Condformer': torch.optim.AdamW, 'LoNPE': torch.optim.AdamW}, help='optimizers')
parser.add_argument('--lr', type=float, default={'Condformer': 4e-4, 'LoNPE': 1e-3}, help='initial learning rate')
parser.add_argument('--lr_type', type=str, default={'Condformer': 'Step', 'LoNPE': 'Cosine'}, help='learning rate decay per N epochs')
parser.add_argument('--lr_gamma_1', type=int, default={'Condformer': 50, 'LoNPE': 100}, help='learning rate decay per N epochs')
parser.add_argument('--lr_gamma_2', type=float, default={'Condformer': 0.5, 'LoNPE': 1e-6}, help='min learning rate for convergence')
args = parser.parse_args()