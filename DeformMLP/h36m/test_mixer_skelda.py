import sys
import time
import torch
import os
from datasets.dataset_h36m import H36M_Dataset
from datasets.dataset_h36m_ang import H36M_Dataset_Angle
from utils.data_utils import define_actions
from torch.utils.data import DataLoader
from mlp_h36m import MorphMLP
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
import argparse
from utils.utils_mixer import delta_2_gt, mpjpe_error, euler_error
from h36_3d_viz import visualize
import tqdm
from torch.utils.tensorboard import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================================================================================================

sys.path.append("/PoseForecasters/")
import utils_pipeline

datamode = "gt-gt"
# datamode = "pred-gt"
# datamode = "pred-pred"
jloss_timestep = 0

sconfig = {
    "item_step": 2,
    "window_step": 2,
    # "item_step": 1,
    # "window_step": 1,
    "select_joints": [
        "hip_right",
        "hip_left",
        "knee_right",
        "knee_left",
        "ankle_right",
        "ankle_left",
        "nose",
        "shoulder_right",
        "shoulder_left",
        "elbow_right",
        "elbow_left",
        "wrist_right",
        "wrist_left",
    ],
}

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_rpt.json"
dataset_eval_test = dataset_eval_test.format("test")


# ==================================================================================================


def prepare_sequences(batch, batch_size: int, split: str, device, dmode):

    sequences = utils_pipeline.make_input_sequence(batch, split, dmode)

    # Add empty joints to pad the number of joints to the original number of joints
    s1 = sequences.shape
    s3 = len(sconfig["select_joints"])
    sequences = np.concatenate(
        (sequences, np.zeros((s1[0], s1[1], 22 - s3, 3))), axis=2
    )

    # Merge joints and coordinates to a single dimension
    sequences = sequences.reshape([batch_size, sequences.shape[1], -1])

    # Convert to millimeters
    sequences = sequences * 1000

    sequences = torch.from_numpy(sequences).to(device)

    return sequences

# ==================================================================================================


def test_pretrained(model, args):
    model.eval()

    sconfig["input_n"] = args.input_n
    sconfig["output_n"] = args.output_n

    # Load preprocessed datasets
    dataset_test, dlen = utils_pipeline.load_dataset(dataset_eval_test, "test", sconfig)
    dataset_test = dataset_test["sequences"]
    label_gen_test = utils_pipeline.create_labels_generator(dataset_test, sconfig)

    frame_losses = np.zeros([args.output_n])
    nitems = 0
    stime = time.time()

    with torch.no_grad():
        nbatch = 1

        for batch in tqdm.tqdm(label_gen_test, total=dlen):
            if nbatch == 1:
                batch = [batch]

            nitems += nbatch
            sequences_train = prepare_sequences(
                batch, nbatch, "input", device, datamode
            )
            sequences_gt = prepare_sequences(batch, nbatch, "target", device, datamode)

            if args.delta_x:
                sequences_all = torch.cat((sequences_train, sequences_gt), 1)
                sequences_all_delta = [sequences_all[:, 1, :] - sequences_all[:, 0, :]]
                for i in range(args.input_n + args.output_n - 1):
                    sequences_all_delta.append(sequences_all[:, i + 1, :] - sequences_all[:, i, :])

                sequences_all_delta = torch.stack((sequences_all_delta)).permute(1, 0, 2)
                sequences_train_delta = sequences_all_delta[:, 0:args.input_n, :]
                sequences_predict = model(sequences_train_delta)
                sequences_predict = delta_2_gt(sequences_predict, sequences_train[:, -1, :])

            else:
                sequences_predict = model(sequences_train)

            # Remove the padding again
            s2 = sequences_gt.shape
            s3 = len(sconfig["select_joints"])
            s4 = sequences_predict.shape
            sequences_predict = sequences_predict.reshape([s4[0], s4[1], 22, 3])
            sequences_predict = sequences_predict[:, :, :s3, :]
            sequences_gt = sequences_gt.reshape([s2[0], s2[1], 22, 3])
            sequences_gt = sequences_gt[:, :, :s3, :]

            # Calculate the loss
            loss = torch.sqrt(
                torch.sum((sequences_predict - sequences_gt) ** 2, dim=-1)
            )
            floss = torch.sum(torch.mean(loss, dim=2), dim=0)
            frame_losses += floss.cpu().data.numpy()


    avg_losses = frame_losses / nitems
    avg_losses = np.round(avg_losses, 1)
    print("Averaged frame losses in mm are:", avg_losses)

    ftime = time.time()
    print("Testing took {} seconds".format(int(ftime - stime)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=1, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default='./runs', type=str, help='root path for the logging') #'./runs'

    parser.add_argument('--activation', default='mish', type=str, required=False)  # 'mish', 'gelu'
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=50, type=int, required=False)
    parser.add_argument('--batch_size', default=50, type=int, required=False)  # 100  50  in all original 50
    parser.add_argument('--loader_shuffle', default=True, type=bool, required=False)
    parser.add_argument('--pin_memory', default=False, type=bool, required=False)
    parser.add_argument('--loader_workers', default=4, type=int, required=False)
    parser.add_argument('--load_checkpoint', default=False, type=bool, required=False)
    parser.add_argument('--dev', default='cuda:0', type=str, required=False)
    parser.add_argument('--initialization', type=str, default='none', help='none, glorot_normal, glorot_uniform, hee_normal, hee_uniform')
    parser.add_argument('--use_scheduler', default=True, type=bool, required=False)
    parser.add_argument('--milestones', type=list, default=[15, 25, 35, 40], help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--gamma', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--clip_grad', type=float, default=None, help='select max norm to clip gradients')
    parser.add_argument('--model_path', type=str, default='../checkpoints/h36m/h36_3d_25frames_ckpt', help='directory with the models checkpoints ')
    parser.add_argument('--actions_to_consider', default='all', help='Actions to visualize.Choose either all or a list of actions')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])
    parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--n_viz', type=int, default='5', help='Numbers of sequences to visaluze for each action')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'viz'],
                        help='Choose to train,test or visualize from the model.Either train,test or viz')

    parser.add_argument('--in_chans', default=3, help='number of block')
    parser.add_argument('--layers', type=list, default=[3, 4, 9, 3], help='number of block')
    parser.add_argument('--transitions', type=list, default=[True, True, True, True],
                        help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--segment_dim', type=list, default=[14, 28, 28, 49],
                        help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--t_stride', default=4,
                        help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--patch_size', default=7,
                        help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--mlp_ratios', type=list, default=[3, 3, 3, 3],
                        help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--embed_dims', type=list, default=[80, 224, 392, 784],
                        help='the epochs after which the learning rate is adjusted by gamma')
    parser.add_argument('--attn_drop_rate', type=float, default=0.1,
                        help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help='gamma correction to the learning rate, after reaching the milestone epochs')




    args = parser.parse_args()

    if args.loss_type == 'mpjpe':
        parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
        parser_mpjpe.add_argument('--hidden_dim', default=50, type=int, required=False)
        parser_mpjpe.add_argument('--num_blocks', default=4, type=int, required=False)
        parser_mpjpe.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
        parser_mpjpe.add_argument('--channels_mlp_dim', default=50, type=int, required=False)
        parser_mpjpe.add_argument('--regularization', default=0.1, type=float, required=False)
        parser_mpjpe.add_argument('--pose_dim', default=66, type=int, required=False)
        parser_mpjpe.add_argument('--delta_x', type=bool, default=True, help='predicting the difference between 2 frames')
        parser_mpjpe.add_argument('--lr', default=0.001, type=float, required=False)
        args = parser_mpjpe.parse_args()

    elif args.loss_type == 'angle':
        parser_angle = argparse.ArgumentParser(parents=[parser]) # Parameters for angle
        parser_angle.add_argument('--hidden_dim', default=60, type=int, required=False)
        parser_angle.add_argument('--num_blocks', default=3, type=int, required=False)
        parser_angle.add_argument('--tokens_mlp_dim', default=40, type=int, required=False)
        parser_angle.add_argument('--channels_mlp_dim', default=60, type=int, required=False)
        parser_angle.add_argument('--regularization', default=0.0, type=float, required=False)
        parser_angle.add_argument('--pose_dim', default=48, type=int, required=False)
        parser_angle.add_argument('--lr', default=1e-02, type=float, required=False)
        args = parser_angle.parse_args()



    if args.loss_type == 'angle' and args.delta_x:
        raise ValueError('Delta_x and loss type angle cant be used together.')

    print(args)

    model = MorphMLP(args.output_n)
    if args.mode == 'test':

        model = model.to(args.dev)

        model.load_state_dict(torch.load(args.model_path))

        print('total number of parameters of the network is: ' +
              str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        test_pretrained(model, args)
    elif args.mode=='viz':
       model.load_state_dict(torch.load(os.path.join(args.model_path)))
       model.eval()
       visualize(args.input_n,args.output_n,args.visualize_from,args.data_dir,model,device,args.n_viz,args.skip_rate,args.actions_to_consider)




