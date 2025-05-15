import sys
import torch
import os
from datasets.dataset_h36m import H36M_Dataset
from datasets.dataset_h36m_ang import H36M_Dataset_Angle
from utils.data_utils import define_actions
from torch.utils.data import DataLoader
from mlp_h36m import MorphMLP

import torch.optim as optim
import numpy as np
import argparse
from utils.utils_mixer import delta_2_gt, mpjpe_error, euler_error
import tqdm
from torch.utils.tensorboard import SummaryWriter

# ==================================================================================================

sys.path.append("/PoseForecasters/")
import utils_pipeline

datamode = "gt-gt"
# datamode = "pred-pred"

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

datasets_train = [
    "/datasets/preprocessed/human36m/train_forecast_rpt.json",
]

dataset_eval_test = "/datasets/preprocessed/human36m/{}_forecast_rpt.json"

# ==================================================================================================


def get_log_dir(out_dir):
    dirs = [x[0] for x in os.walk(out_dir)]
    if len(dirs ) < 2:
        log_dir = os.path.join(out_dir, 'exp0')
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = os.path.join(out_dir, 'exp%i'%(len(dirs)-1))
        os.makedirs(log_dir, exist_ok=True)

    return log_dir


def train(model, model_name, args):

    log_dir = get_log_dir(args.root)
    tb_writer = SummaryWriter(log_dir=log_dir)
    print('Save data of the run in: %s'%log_dir)

    device = args.dev

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-05)

    sconfig["input_n"] = args.input_n
    sconfig["output_n"] = args.output_n

    if args.use_scheduler:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=args.gamma)

    # Load preprocessed datasets
    print("Loading datasets ...")
    dataset_train, dlen_train = [], 0
    for dp in datasets_train:
        ds, dlen = utils_pipeline.load_dataset(dp, "train", sconfig)
        dataset_train.extend(ds["sequences"])
        dlen_train += dlen
    dataset_eval, dlen_eval = utils_pipeline.load_dataset(
        dataset_eval_test, "eval", sconfig
    )
    dataset_eval = dataset_eval["sequences"]

    train_loss, val_loss, test_loss = [], [], []

    for epoch in range(args.n_epochs):
        print('Run epoch: %i'%epoch)
        running_loss = 0
        n = 0
        model.train()

        label_gen_train = utils_pipeline.create_labels_generator(dataset_train, sconfig)
        label_gen_eval = utils_pipeline.create_labels_generator(dataset_eval, sconfig)

        nbatch = args.batch_size
        for batch in tqdm.tqdm(
            utils_pipeline.batch_iterate(label_gen_train, batch_size=nbatch),
            total=int(dlen_train / nbatch),
        ):
            batch_dim = len(batch)
            n += batch_dim

            sequences_train = utils_pipeline.make_input_sequence(
                batch, "input", datamode
            )
            sequences_gt = utils_pipeline.make_input_sequence(batch, "target", datamode)

            augment = True
            if augment:
                sequences_train, sequences_gt = utils_pipeline.apply_augmentations(
                    sequences_train, sequences_gt
                )

            # Convert to millimeters
            sequences_train = sequences_train * 1000
            sequences_gt = sequences_gt * 1000

            # Add empty joints to pad the number of joints to the original number of joints
            s1 = sequences_train.shape
            s2 = sequences_gt.shape
            s3 = len(sconfig["select_joints"])
            sequences_train = np.concatenate(
                (sequences_train, np.zeros((s1[0], s1[1], 22 - s3, 3))), axis=2
            )
            sequences_gt = np.concatenate(
                (sequences_gt, np.zeros((s2[0], s2[1], 22 - s3, 3))), axis=2
            )

            # Merge joints and coordinates to a single dimension
            sequences_train = sequences_train.reshape(
                [nbatch, sequences_train.shape[1], -1]
            )
            sequences_gt = sequences_gt.reshape([nbatch, sequences_gt.shape[1], -1])

            sequences_train = torch.from_numpy(sequences_train).to(device)
            sequences_gt = torch.from_numpy(sequences_gt).to(device)

            optimizer.zero_grad()

            if args.delta_x:
                sequences_all = torch.cat((sequences_train, sequences_gt), 1)
                sequences_all_delta = [
                    sequences_all[:, 1, :] - sequences_all[:, 0, :]]
                for i in range(args.input_n+args.output_n-1):
                    sequences_all_delta.append(
                        sequences_all[:, i+1, :] - sequences_all[:, i, :])

                sequences_all_delta = torch.stack(
                    (sequences_all_delta)).permute(1, 0, 2)
                sequences_train_delta = sequences_all_delta[:,
                                                            0:args.input_n, :]
                sequences_predict = model(sequences_train_delta)
                sequences_predict = delta_2_gt(
                    sequences_predict, sequences_train[:, -1, :])

                loss = mpjpe_error(sequences_predict, sequences_gt)

            elif args.loss_type == 'mpjpe':
                sequences_train = sequences_train/1000
                sequences_predict = model(sequences_train)
                loss = mpjpe_error(sequences_predict, sequences_gt)

            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.clip_grad)

            optimizer.step()
            running_loss += loss*batch_dim
        train_loss.append(running_loss.detach().cpu()/n)
        print("Train loss: ", train_loss[-1])

        model.eval()
        with torch.no_grad():
            running_loss = 0
            n = 0

            nbatch = args.batch_size_test
            for batch in tqdm.tqdm(
                utils_pipeline.batch_iterate(label_gen_eval, batch_size=nbatch),
                total=int(dlen_eval / nbatch),
            ):
                batch_dim = len(batch)
                n += batch_dim

                sequences_train = utils_pipeline.make_input_sequence(
                    batch, "input", datamode
                )
                sequences_gt = utils_pipeline.make_input_sequence(batch, "target", datamode)

                # Convert to millimeters
                sequences_train = sequences_train * 1000
                sequences_gt = sequences_gt * 1000

                # Add empty joints to pad the number of joints to the original number of joints
                s1 = sequences_train.shape
                s2 = sequences_gt.shape
                s3 = len(sconfig["select_joints"])
                sequences_train = np.concatenate(
                    (sequences_train, np.zeros((s1[0], s1[1], 22 - s3, 3))), axis=2
                )
                sequences_gt = np.concatenate(
                    (sequences_gt, np.zeros((s2[0], s2[1], 22 - s3, 3))), axis=2
                )

                # Merge joints and coordinates to a single dimension
                sequences_train = sequences_train.reshape(
                    [nbatch, sequences_train.shape[1], -1]
                )
                sequences_gt = sequences_gt.reshape([nbatch, sequences_gt.shape[1], -1])

                sequences_train = torch.from_numpy(sequences_train).to(device)
                sequences_gt = torch.from_numpy(sequences_gt).to(device)

                if args.delta_x:
                    sequences_all = torch.cat(
                        (sequences_train, sequences_gt), 1)
                    sequences_all_delta = [
                        sequences_all[:, 1, :] - sequences_all[:, 0, :]]
                    for i in range(args.input_n+args.output_n-1):
                        sequences_all_delta.append(
                            sequences_all[:, i+1, :] - sequences_all[:, i, :])

                    sequences_all_delta = torch.stack(
                        (sequences_all_delta)).permute(1, 0, 2)
                    sequences_train_delta = sequences_all_delta[:,
                                                                0:args.input_n, :]
                    sequences_predict = model(sequences_train_delta)
                    sequences_predict = delta_2_gt(
                        sequences_predict, sequences_train[:, -1, :])

                    # Remove the padding again
                    s4 = sequences_predict.shape
                    sequences_predict = sequences_predict.reshape([s4[0], s4[1], 22, 3])
                    sequences_predict = sequences_predict[:, :, :s3, :]
                    sequences_predict = sequences_predict.reshape([s4[0], s4[1], -1])
                    sequences_gt = sequences_gt.reshape([s2[0], s2[1], 22, 3])
                    sequences_gt = sequences_gt[:, :, :s3, :]
                    sequences_gt = sequences_gt.reshape([s2[0], s2[1], -1])

                    loss = mpjpe_error(sequences_predict, sequences_gt)

                elif args.loss_type == 'mpjpe':
                    sequences_train = sequences_train/1000
                    sequences_predict = model(sequences_train)
                    loss = mpjpe_error(sequences_predict, sequences_gt)

                running_loss += loss*batch_dim
            val_loss.append(running_loss.detach().cpu()/n)
            print("Validation loss: ", val_loss[-1])

        if args.use_scheduler:
            scheduler.step()

        if args.loss_type == 'mpjpe':
            test_loss.append(val_loss[-1])

        tb_writer.add_scalar('loss/train', train_loss[-1], epoch)
        tb_writer.add_scalar('loss/val', val_loss[-1], epoch)
        tb_writer.add_scalar('loss/test', test_loss[-1], epoch)

        torch.save(model.state_dict(), os.path.join(log_dir, 'model.pt'))
        # TODO write something to save the best model
        # if (epoch+1)%1==0:
        #     print('----saving model-----')
        #     torch.save(model.state_dict(),os.path.join(args.model_path,model_name))
        if epoch!=0:
            if test_loss[epoch] <= min(test_loss):

                print('----saving model-----')
                torch.save(model.state_dict(),os.path.join(args.model_path,model_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False) # Parameters for mpjpe
    parser.add_argument('--data_dir', type=str, default='../datasets/', help='path to the unziped dataset directories(H36m/AMASS/3DPW)')
    parser.add_argument('--input_n', type=int, default=10, help="number of model's input frames")
    parser.add_argument('--output_n', type=int, default=25, help="number of model's output frames")
    parser.add_argument('--skip_rate', type=int, default=5, choices=[1, 5], help='rate of frames to skip,defaults=1 for H36M or 5 for AMASS/3DPW')
    parser.add_argument('--num_worker', default=4, type=int, help='number of workers in the dataloader')
    parser.add_argument('--root', default='./runs', type=str, help='root path for the logging') #'./runs'

    parser.add_argument('--activation', default='mish', type=str, required=False) 
    parser.add_argument('--r_se', default=8, type=int, required=False)

    parser.add_argument('--n_epochs', default=50, type=int, required=False)
    parser.add_argument('--batch_size', default=64, type=int, required=False)
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
    parser.add_argument('--model_path', type=str, default='./checkpoints/h36m', help='directory with the models checkpoints ')
    parser.add_argument('--actions_to_consider', default='all', help='Actions to visualize.Choose either all or a list of actions')
    parser.add_argument('--batch_size_test', type=int, default=256, help='batch size for the test set')
    parser.add_argument('--visualize_from', type=str, default='test', choices=['train', 'val', 'test'], help='choose data split to visualize from(train-val-test)')
    parser.add_argument('--loss_type', type=str, default='mpjpe', choices=['mpjpe', 'angle'])

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
    parser.add_argument('--attn_drop_rate', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help='gamma correction to the learning rate, after reaching the milestone epochs')

    args = parser.parse_args()

    if args.loss_type == 'mpjpe':
        parser_mpjpe = argparse.ArgumentParser(parents=[parser]) # Parameters for mpjpe
        parser_mpjpe.add_argument('--hidden_dim', default=96, type=int, required=False)
        parser_mpjpe.add_argument('--num_blocks', default=4, type=int, required=False)  
        parser_mpjpe.add_argument('--tokens_mlp_dim', default=20, type=int, required=False)
        parser_mpjpe.add_argument('--channels_mlp_dim', default=50, type=int, required=False)  
        parser_mpjpe.add_argument('--regularization', default=0.1, type=float, required=False)  
        parser_mpjpe.add_argument('--pose_dim', default=39, type=int, required=False)
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
        parser_angle.add_argument('--delta_x', type=bool, default=False,
                                  help='predicting the difference between 2 frames')
        args = parser_angle.parse_args()
    
    if args.loss_type == 'angle' and args.delta_x:
        raise ValueError('Delta_x and loss type angle cant be used together.')

    print(args)

    model = MorphMLP(pre_len=args.output_n)
    model = model.to(args.dev)

    print('total number of parameters of the network is: ' +
          str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    model_name = 'h36_3d_'+str(args.input_n)+'_'+str(args.output_n)+'frames_ckpt'

    train(model, model_name, args)
