
import argparse
import os
from training import *
from sampling import *


import os
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

parser = argparse.ArgumentParser(description='Argparse Tutorial')

#training parameter
parser.add_argument('--n_epochs', type=int, default =1000 , help='epochs')
parser.add_argument('--alpha', type=float, default = 1.8, help='epochs')
parser.add_argument('--beta_min', type=float, default = 0.1)
parser.add_argument('--beta_max', type=float, default = 20)
parser.add_argument('--num_steps', type=int, default = 1000 )
parser.add_argument('--lr', type=float, default = 1e-4 )
parser.add_argument('--batch_size', type=int, default = 128)
parser.add_argument('--num_workers', type=int, default = 0)
parser.add_argument('--ckpt1', type=str, default = None)
parser.add_argument('--ckpt2', type=str, default = None)
parser.add_argument('--datasets', type=str, default = "CIFAR10")
parser.add_argument('--training_clamp', type=float, default = 20)
parser.add_argument('--b', type=float, default = 0.5)
parser.add_argument('--c', type=float, default = 0.5)
#sampling parameter
parser.add_argument('--path1', type=str, default = 'ckpt.pth' )
parser.add_argument('--path2', type=str, default = 'ckpt.pth' )
parser.add_argument("--sample", action="store_true")
parser.add_argument('--sample_alpha', type=float, default = 2 )
parser.add_argument('--Corrector', type=bool, default = False)
parser.add_argument('--Predictor', type=bool, default = True )
parser.add_argument('--sample_num_steps', type=int, default = 1000 )
parser.add_argument('--sample_batch_size', type=int, default = 64 )
parser.add_argument('--train_sample', action='store_true')


args = parser.parse_args()
#ckpt_name = args.datasets+'ckpt.pth'
ckpt_name=None

def main():
    train(alpha=args.alpha,b = args.b, c=args.c,  beta_min=args.beta_min, beta_max=args.beta_max, lr=args.lr, batch_size=args.batch_size,
              n_epochs=args.n_epochs, num_steps=args.num_steps, datasets=args.datasets,  path1=args.ckpt1,path2=args.ckpt2, training_clamp = args.training_clamp)

if __name__=='__main__':
    main()

