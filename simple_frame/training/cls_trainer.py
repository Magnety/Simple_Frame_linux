import torch
import torch.nn as nn
import time
import numpy as np
import shutil
from typing import Tuple, List
from simple_frame.network_architecture.neural_network import SegmentationNetwork
from simple_frame.network_architecture.neural_network import ClassficationNetwork
from simple_frame.loss_function.focal_loss import focal_loss
from multiprocessing import Pool
from torch.cuda.amp import GradScaler, autocast

from time import sleep
from collections import OrderedDict
from torch.optim import lr_scheduler
from batchgenerators.utilities.file_and_folder_operations import *
from simple_frame.evaluation.evaluator import aggregate_scores
from simple_frame.inference.segmentation_export import save_segmentation_nifti_from_softmax
from simple_frame.postprocessing.connected_components import determine_postprocessing
from sklearn.model_selection import KFold
from datetime import datetime
import sys
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import matplotlib
from simple_frame.network_architecture.generic_VNet import VNet_class
from simple_frame.data_augmentation.data_augmentation import default_3D_augmentation_params, default_2D_augmentation_params, get_default_augmentation, get_patch_size
from simple_frame.dataloading.dataset_load import load_dataset, DataLoader3D, unpack_dataset
from simple_frame.loss_function.Loss_functions import JI_and_Focal_loss, AutomaticWeightedLoss, softmax_helper, sum_tensor

try:
    from apex import amp
except ImportError:
    amp = None

class Trainer(nn.Module):
    def __init__(self, fold,stage, data_root, out_path, output_fold,raw_path):
        super(Trainer, self).__init__()
        self.fold = fold
        self.stage = stage
        self.data_root = data_root +"/plans_v2.1_stage0"
        self.dataset_directory = out_path
        self.output_checkpoints = self.output_folder = output_fold
        self.was_initialized = False
        self.log_file = self.use_mask_for_norm = None
        self.batch_dice = False
        self.seg_loss = JI_and_Focal_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False})
        self.aw1 = AutomaticWeightedLoss(1)
        self.fp16 = False
        self.plans_file = data_root+"/Plansv2.1_plans_3D.pkl"
        self.fc_path = raw_path
        self.dataset_tr = self.dataset_val = None  # do not need to be used, they just appear if you are using the suggested load_dataset_and_do_split


    def initialize(self, training=True):
        if not self.was_initialized:
            if not os.path.isdir(self.output_folder):
                os.makedirs(self.output_folder)


            self.load_plans_file()

            self.process_plans(self.plans)
            self.batch_size = 20
            self.patch_size = np.array([32,128,128]).astype(int)
            self.do_data_augmentation = True
            self.num_classes = 2
            self.setup_data_aug_params()
            self.initial_lr = 5e-3
            self.lr_scheduler_eps = 1e-3
            self.lr_scheduler_patience = 5
            self.weight_decay = 2e-5

            self.oversample_foreground_percent = 0.33
            self.pad_all_sides = None

            self.patience = 10
            self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
            # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
            # too high the training will take forever
            self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
            self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
            self.save_every = 50
            self.save_latest_only = True
            self.max_num_epochs = 500
            self.num_batches_per_epoch = 300
            self.num_val_batches_per_epoch = 100
            self.also_val_in_tr_mode = False
            self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold

            self.val_eval_criterion_MA = None
            self.val_eval_criterion_MA_class = None
            self.train_loss_MA = None
            self.best_val_eval_criterion_MA = None
            self.best_val_eval_criterion_MA_class = None
            self.best_MA_tr_loss_for_patience = None
            self.best_epoch_based_on_MA_tr_loss = None
            self.all_tr_losses = []
            self.all_val_losses = []
            self.all_val_losses_tr_mode = []
            self.all_val_eval_metrics = []  # does not have to be used
            self.epoch = 0
            self.log_file = None

            self.online_eval_foreground_dc = []
            self.online_eval_tp = []
            self.online_eval_fp = []
            self.online_eval_fn = []

            self.online_eval_foreground_acc = []
            self.online_eval_tp_class = []
            self.online_eval_fp_class = []
            self.online_eval_fn_class = []
            self.online_eval_tn_class = []
            self.online_eval_class_precision = []
            self.online_eval_class_recall = []
            self.online_eval_class_f1 = []
            self.all_val_eval_metrics_acc =[]
            self.all_val_eval_metrics_seg = []
            if training:
                self.train_data, self.val_data = self.get_data()
                unpack_dataset(self.data_root)
                self.train_data, self.val_data = get_default_augmentation(self.train_data, self.val_data,
                                                                     self.data_aug_params[
                                                                         'patch_size_for_spatialtransform'],
                                                                     self.data_aug_params)
                total = 0
                positive = 0
                for k in self.dataset_tr.keys():
                    class_path = self.fc_path + '/classesTr'
                    class_source = open(class_path + '/' + k + '.txt')  # 打开源文件
                    indate = class_source.read()  # 显示所有源文件内容
                    if int(indate) == 1:
                        positive += 1
                    total += 1
                    positive_weight = positive / total
                self.class_loss = focal_loss(alpha=positive_weight)

            #self.network = MTLN3D().cuda()
            self.network = VNet_class(num_classes=self.num_classes, deep_supervision=True).cuda()
            self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                              amsgrad=True)
            self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.8,
                                                               patience=self.lr_scheduler_patience,
                                                               verbose=True, threshold=self.lr_scheduler_eps,
                                                               threshold_mode="abs")

        else:
            self.print_to_log_file('self.was_initialized is True, not running self.initialize again')
        self.was_initialized = True

    def do_split(self):
        """
        This is a suggestion for if your dataset is a dictionary (my personal standard)
        :return:
        """
        if self.fold == "all":
            # if fold==all then we use all images for training and validation
            tr_keys = val_keys = list(self.dataset.keys())
        else:
            splits_file = self.dataset_directory + "/" + "splits_final.pkl"

            # if the split file does not exist we need to create it
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split...")
                splits = []
                all_keys_sorted = np.sort(list(self.dataset.keys()))
                kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
                for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
                    train_keys = np.array(all_keys_sorted)[train_idx]
                    test_keys = np.array(all_keys_sorted)[test_idx]
                    splits.append(OrderedDict())
                    splits[-1]['train'] = train_keys
                    splits[-1]['val'] = test_keys
                save_pickle(splits, splits_file)

            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_pickle(splits_file)
                self.print_to_log_file("The split file contains %d splits." % len(splits))

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                self.print_to_log_file("INFO: You requested fold %d for training but splits "
                                       "contain only %d folds. I am now creating a "
                                       "random (but seeded) 80:20 split!" % (self.fold, len(splits)))
                # if we request a fold that is not in the split file, create a random 80:20 split
                rnd = np.random.RandomState(seed=12345 + self.fold)
                keys = np.sort(list(self.dataset.keys()))
                idx_tr = rnd.choice(len(keys), int(len(keys) * 0.8), replace=False)
                idx_val = [i for i in range(len(keys)) if i not in idx_tr]
                tr_keys = [keys[i] for i in idx_tr]
                val_keys = [keys[i] for i in idx_val]
                self.print_to_log_file("This random 80:20 split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))

        tr_keys.sort()
        val_keys.sort()
        self.dataset_tr = OrderedDict()
        for i in tr_keys:
            self.dataset_tr[i] = self.dataset[i]
        self.dataset_val = OrderedDict()
        for i in val_keys:
            self.dataset_val[i] = self.dataset[i]

    def load_dataset(self):
        self.dataset = load_dataset(self.data_root)

    def get_data(self):
        self.load_dataset()
        self.do_split()
        dl_tr = DataLoader3D(self.dataset_tr, self.basic_generator_patch_size, self.patch_size, self.batch_size,
                             False, oversample_foreground_percent=self.oversample_foreground_percent,
                             pad_mode="constant", pad_sides=self.pad_all_sides,fc_path=self.fc_path)

        dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                              oversample_foreground_percent=self.oversample_foreground_percent,
                              pad_mode="constant", pad_sides=self.pad_all_sides,fc_path=self.fc_path)
        return dl_tr, dl_val
    def load_plans_file(self):
        """
        This is what actually configures the entire experiment. The plans file is generated by experiment planning
        :return:
        """
        self.plans = load_pickle(self.plans_file)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        #self.batch_size = stage_plans['batch_size']
        #liuyiyao
        self.batch_size = 12
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']

        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.print_to_log_file("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):

        timestamp = time.time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = ("%s:" % dt_object, *args)

        if self.log_file is None:
            maybe_mkdir_p(self.output_folder)
            timestamp = datetime.now()
            self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                                 (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                                  timestamp.second))
            with open(self.log_file, 'w') as f:
                f.write("Starting... \n")
        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print("%s: failed to log: " % datetime.fromtimestamp(timestamp), sys.exc_info())
                time.sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    def setup_data_aug_params(self):
        self.data_aug_params = default_3D_augmentation_params
        self.data_aug_params['rotation_x'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_y'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params['rotation_z'] = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        self.data_aug_params["dummy_2D"] = True
        self.print_to_log_file("Using dummy2d data augmentation")
        self.data_aug_params["elastic_deform_alpha"] = \
            default_2D_augmentation_params["elastic_deform_alpha"]
        self.data_aug_params["elastic_deform_sigma"] = \
            default_2D_augmentation_params["elastic_deform_sigma"]
        self.data_aug_params["rotation_x"] = default_2D_augmentation_params["rotation_x"]
        self.data_aug_params["mask_was_used_for_normalization"] = self.use_mask_for_norm
        self.basic_generator_patch_size = get_patch_size(self.patch_size[1:],
                                                         self.data_aug_params['rotation_x'],
                                                         self.data_aug_params['rotation_y'],
                                                         self.data_aug_params['rotation_z'],
                                                         self.data_aug_params['scale_range'])
        self.basic_generator_patch_size = np.array([self.patch_size[0]] + list(self.basic_generator_patch_size))
        patch_size_for_spatialtransform = self.patch_size[1:]
        self.data_aug_params["scale_range"] = (0.7, 1.4)
        self.data_aug_params["do_elastic"] = False
        self.data_aug_params['selected_seg_channels'] = [0]
        self.data_aug_params['patch_size_for_spatialtransform'] = patch_size_for_spatialtransform
        self.data_aug_params["num_cached_per_thread"] = 2

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):

        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']
        class_target = data_dict['class_label']
        feature = data_dict['feature']
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        if not isinstance(class_target, torch.Tensor):
            class_target = torch.from_numpy(class_target).float()
        if not isinstance(feature, torch.Tensor):
            feature = torch.from_numpy(feature).float()
        data = data.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        class_target = class_target.cuda(non_blocking=True)

        feature = feature.cuda(non_blocking=True)
        self.optimizer.zero_grad()
        output = self.network(data,feature)
        #print("///////////output///////////")
        #print(output)
        #output = self.network(data)
        del data

        l10 = self.class_loss(output, class_target.squeeze().long())

        l = self.aw1.cuda()( l10)


        if run_online_evaluation:
            #self.run_online_evaluation(output[2], output[3], target, class_target)
            self.run_online_evaluation_onlycls(output,class_target)
        del target
        if do_backprop:
            if not self.fp16 or amp is None:
                l.backward()
            else:
                with amp.scale_loss(l, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            self.optimizer.step()

        return l.detach().cpu().numpy()
    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def save_checkpoint(self, fname, save_optimizer=True):
        start_time = time.time()
        state_dict = self.network.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        lr_sched_state_dct = None
        if self.lr_scheduler is not None and not isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
            lr_sched_state_dct = self.lr_scheduler.state_dict()
            for key in lr_sched_state_dct.keys():
                lr_sched_state_dct[key] = lr_sched_state_dct[key]
        if save_optimizer:
            optimizer_state_dict = self.optimizer.state_dict()
        else:
            optimizer_state_dict = None

        self.print_to_log_file("saving checkpoint...")
        torch.save({
            'epoch': self.epoch + 1,
            'state_dict': state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'lr_scheduler_state_dict': lr_sched_state_dct,
            'plot_stuff': (self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode,
                           self.all_val_eval_metrics)},
            fname)
        self.print_to_log_file("done, saving took %.2f seconds" % (time.time() - start_time))

    def finish_online_evaluation_onlycls(self):
        ###classification
        self.online_eval_tp_class = np.sum(self.online_eval_tp_class, 0)
        self.online_eval_fp_class = np.sum(self.online_eval_fp_class, 0)
        self.online_eval_fn_class = np.sum(self.online_eval_fn_class, 0)
        self.online_eval_tn_class = np.sum(self.online_eval_tn_class, 0)

        global_acc_per_class = [i for i in [(i + h) / (i + j + k + h+ 1e-8) for i, j, k, h in
                                            zip(self.online_eval_tp_class, self.online_eval_fp_class,
                                                self.online_eval_fn_class, self.online_eval_tn_class)]
                    if not np.isnan(i)]
        global_pre_per_class = [i for i in [(i) / (i + j+ 1e-8) for i, j in
                                            zip(self.online_eval_tp_class, self.online_eval_fp_class)]
                                if not np.isnan(i)]
        global_rec_per_class = [i for i in [(i) / (i + j + 1e-8) for i, j in
                                            zip(self.online_eval_tp_class, self.online_eval_fn_class)]
                                if not np.isnan(i)]

        global_f1_per_class = np.mean(global_pre_per_class)*np.mean(global_rec_per_class)*2/(np.mean(global_pre_per_class)+np.mean(global_rec_per_class)+ 1e-8)

        self.online_eval_foreground_acc = []
        self.online_eval_tp_class = []
        self.online_eval_fp_class = []
        self.online_eval_fn_class = []
        self.online_eval_tn_class = []
        ###segmentation

        self.all_val_eval_metrics.append(np.mean(global_acc_per_class))
        self.all_val_eval_metrics_acc.append(np.mean(global_acc_per_class))
        self.online_eval_class_recall.append(np.mean(global_rec_per_class))
        self.online_eval_class_precision.append(np.mean(global_pre_per_class))
        self.online_eval_class_f1.append(global_f1_per_class)
        self.print_to_log_file("Val global acc per class:", str(global_acc_per_class))
        self.print_to_log_file("Val global pre per class:", str(global_pre_per_class))
        self.print_to_log_file("Val global rec per class:", str(global_rec_per_class))
        self.print_to_log_file("Val global f1 per class:", str(global_f1_per_class))

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.val_eval_criterion_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def load_best_checkpoint(self, train=True, epoch=100):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(self.output_folder+'/'+"epoch_%s_model_best.model"%epoch):
            self.load_checkpoint(self.output_folder+'/'+ "epoch_%s_model_best.model"%epoch, train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True,epoch=100):
        if isfile(self.output_folder+'/'+"model_final_checkpoint.model"):
            return self.load_checkpoint(self.output_folder+'/'+"model_final_checkpoint.model", train=train)
        if isfile(self.output_folder+'/'+ "epoch_%s_model_best.model"%epoch):
            return self.load_checkpoint(self.output_folder+'/'+ "epoch_%s_model_best.model"%epoch, train=train)
        if isfile(self.output_folder+'/'+ "epoch_100_model_best.model"):
            return self.load_best_checkpoint(train)
        raise RuntimeError("No checkpoint found")

    def load_final_checkpoint(self, train=False):
        filename = self.output_folder+'/'+ "model_final_checkpoint.model"
        if not isfile(filename):
            raise RuntimeError("Final checkpoint not found. Expected: %s. Please finish the training first." % filename)
        return self.load_checkpoint(filename, train=train)

    def load_checkpoint(self, fname, train=True):
        self.print_to_log_file("loading checkpoint", fname, "train=", train)
        if not self.was_initialized:
            self.initialize(train)
        # saved_model = torch.load(fname, map_location=torch.device('cuda', torch.cuda.current_device()))
        saved_model = torch.load(fname, map_location=torch.device('cpu'))
        self.load_checkpoint_ram(saved_model, train)

    def maybe_save_checkpoint(self):
        """
        Saves a checkpoint every save_ever epochs.
        :return:
        """
        if self.epoch % self.save_every == (self.save_every - 1):
            self.print_to_log_file("saving scheduled checkpoint file...")
            if not self.save_latest_only:
                self.save_checkpoint(join(self.output_folder, "model_ep_%03.0d.model" % (self.epoch + 1)))
            self.save_checkpoint(join(self.output_folder, "model_latest.model"))
            self.print_to_log_file("done")

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)




            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best segmentation epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_best_cls.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                self.print_to_log_file(
                    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def plot_progress(self):
        """
        Should probably by improved
        :return:
        """
        try:
            font = {'weight': 'normal',
                    'size': 18}
            matplotlib.rc('font', **font)
            fig = plt.figure(figsize=(30, 24))
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()
            x_values = list(range(self.epoch + 1))
            ax.plot(x_values, self.all_tr_losses, color='b', ls='-', label="loss_tr")
            ax.plot(x_values, self.all_val_losses, color='r', ls='-', label="loss_val, train=False")
            if len(self.all_val_losses_tr_mode) > 0:
                ax.plot(x_values, self.all_val_losses_tr_mode, color='g', ls='-', label="loss_val, train=True")
            if len(self.all_val_eval_metrics_acc) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics_acc, color='g', ls='--', label="acc")
            if len(self.online_eval_class_precision) == len(x_values):
                ax2.plot(x_values, self.online_eval_class_precision, color='purple', ls='--', label="pre")
            if len(self.online_eval_class_recall) == len(x_values):
                ax2.plot(x_values, self.online_eval_class_recall, color='pink', ls='--', label="rec")
            if len(self.online_eval_class_f1) == len(x_values):
                ax2.plot(x_values, self.online_eval_class_f1, color='cyan', ls='--', label="f1")
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss")
            ax2.set_ylabel("evaluation metric")
            ax.legend()
            ax2.legend(loc=9)

            fig.savefig(self.output_folder + "/" + "progress.png")
            plt.close()
        except IOError:
            self.print_to_log_file("failed to plot: ", sys.exc_info())
    def on_epoch_end(self):
        self.finish_online_evaluation_onlycls()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()
        self.update_eval_criterion_MA()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()


        continue_training = self.manage_patience()
        return continue_training

    def run_online_evaluation_onlycls(self, output_class, target_class):
        with torch.no_grad():
            ###classification
            #print("/////////////output_class.shape///////////////:",output_class.shape)
            num_classes0 = output_class.shape[1]
            output_class_softmax = softmax_helper(output_class)
            output_class = output_class_softmax.argmax(1)
            target_class = target_class[:, 0]

            axes_class = tuple(range(1, len(target_class.shape)))
            tp_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            fp_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            fn_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            tn_hard_class = torch.zeros((target_class.shape[0], num_classes0 - 1)).to(output_class.device.index)
            for c in range(1, num_classes0):
                tp_hard_class[:, c - 1] = sum_tensor((output_class == c).float() * (target_class == c).float(),
                                                     axes=axes_class)
                fp_hard_class[:, c - 1] = sum_tensor((output_class == c).float() * (target_class != c).float(),
                                                     axes=axes_class)
                fn_hard_class[:, c - 1] = sum_tensor((output_class != c).float() * (target_class == c).float(),
                                                     axes=axes_class)
                tn_hard_class[:, c - 1] = sum_tensor((output_class != c).float() * (target_class != c).float(),
                                                     axes=axes_class)

            tp_hard_class = tp_hard_class.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard_class = fp_hard_class.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard_class = fn_hard_class.sum(0, keepdim=False).detach().cpu().numpy()
            tn_hard_class = tn_hard_class.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_acc.append(list((tp_hard_class + tn_hard_class) / (tp_hard_class + fp_hard_class + fn_hard_class + tn_hard_class + 1e-8)))
            self.online_eval_tp_class.append(list(tp_hard_class))
            self.online_eval_fp_class.append(list(fp_hard_class))
            self.online_eval_fn_class.append(list(fn_hard_class))
            self.online_eval_tn_class.append(list(tn_hard_class))
            ###segmentation

    def run_trainer(self):
        torch.cuda.empty_cache()
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #print("///////////run trainer1////////////////")
        if not self.was_initialized:
            #print("//////////////not initialized//////////////////")
            self.initialize(True)
        #print("//////////////initialized//////////////////")
        while self.epoch < self.max_num_epochs:
            self.print_to_log_file("\nepoch: ", self.epoch)
            epoch_start_time = time.time()
            train_losses_epoch = []
            # train one epoch
            self.network.train()
            #print("batches:",self.num_batches_per_epoch)
            for b in range(self.num_batches_per_epoch):
                l = self.run_iteration(self.train_data, do_backprop=True, run_online_evaluation=False)
                train_losses_epoch.append(l)
            self.all_tr_losses.append(np.mean(train_losses_epoch))
            self.print_to_log_file("train loss : %.4f" % self.all_tr_losses[-1])
            with torch.no_grad():
                # validation with train=False
                self.network.eval()
                val_losses = []
                for b in range(self.num_val_batches_per_epoch):
                    l = self.run_iteration(self.val_data, do_backprop=False,run_online_evaluation=True)
                    val_losses.append(l)
                self.all_val_losses.append(np.mean(val_losses))
                self.print_to_log_file("val loss (train=False): %.4f" % self.all_val_losses[-1])

                if self.also_val_in_tr_mode:
                    self.network.train()
                    # validation with train=True
                    val_losses = []
                    for b in range(self.num_val_batches_per_epoch):
                        l = self.run_iteration(self.val_gen, False)
                        val_losses.append(l)
                    self.all_val_losses_tr_mode.append(np.mean(val_losses))
                    self.print_to_log_file("val loss (train=True): %.4f" % self.all_val_losses_tr_mode[-1])
            epoch_end_time = time.time()
            self.update_train_loss_MA()  # needed for lr scheduler and stopping of training
            continue_training = self.on_epoch_end()
            if not continue_training:
                # allows for early stopping
                break

            self.epoch += 1
            if self.epoch % 20 == 0:
                self.save_checkpoint(join(self.output_folder, 'epoch_{}_model_best.model'.format(self.epoch)))
            self.print_to_log_file("This epoch took %f s\n" % (epoch_end_time - epoch_start_time))

        self.save_checkpoint(join(self.output_folder, "model_final_checkpoint.model"))
        # now we can delete latest as it will be identical with final
        if isfile(join(self.output_folder, "model_latest.model")):
            os.remove(join(self.output_folder, "model_latest.model"))
        if isfile(join(self.output_folder, "model_latest.model.pkl")):
            os.remove(join(self.output_folder, "model_latest.model.pkl"))

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None):
        """
        We need to wrap this because we need to enforce self.network.do_ds = False for prediction
        """

        ds = self.network.do_ds
        self.network.do_ds = False
        current_mode = self.network.training
        self.network.eval()
        assert self.was_initialized, "must initialize, ideally with checkpoint (or train first)"
        if self.dataset_val is None:
            self.load_dataset()
            self.do_split()

        if segmentation_export_kwargs is None:
            if 'segmentation_export_params' in self.plans.keys():
                force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
                interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
                interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
            else:
                force_separate_z = None
                interpolation_order = 1
                interpolation_order_z = 0
        else:
            force_separate_z = segmentation_export_kwargs['force_separate_z']
            interpolation_order = segmentation_export_kwargs['interpolation_order']
            interpolation_order_z = segmentation_export_kwargs['interpolation_order_z']

        # predictions as they come from the network go here
        output_folder = self.output_folder + "/" + validation_folder_name
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # this is for debug purposes
        my_input_args = {'do_mirroring': do_mirroring,
                         'use_sliding_window': use_sliding_window,
                         'step_size': step_size,
                         'save_softmax': save_softmax,
                         'use_gaussian': use_gaussian,
                         'overwrite': overwrite,
                         'validation_folder_name': validation_folder_name,
                         'debug': debug,
                         'all_in_gpu': all_in_gpu,
                         'segmentation_export_kwargs': segmentation_export_kwargs,
                         }
        save_json(my_input_args, output_folder + "/" + "validation_args.json")

        if do_mirroring:
            if not self.data_aug_params['do_mirror']:
                raise RuntimeError("We did not train with mirroring so you cannot do inference with mirroring enabled")
            mirror_axes = self.data_aug_params['mirror_axes']
        else:
            mirror_axes = ()

        pred_gt_tuples = []

        export_pool = Pool(4)
        results = []
        cls_tp = 0
        cls_fp = 0
        cls_tn = 0
        cls_fn = 0

        for k in self.dataset_val.keys():
            properties = self.dataset[k]['properties']
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]

            data = np.load(self.dataset[k]['data_file'])['data']
            feature_path = self.fc_path+'/featuresTr'
            class_path = self.fc_path+'/classesTr'
            feature = np.load(feature_path+ '/' + k + '.npy')
            class_source = open(class_path + '/' + k + '.txt')  # 打开源文件
            indate = class_source.read()  # 显示所有源文件内容
            class_target= np.array(float(indate))
            if not isinstance(class_target, torch.Tensor):
                class_target = torch.from_numpy(class_target).float()


            if not isinstance(feature, torch.Tensor):
                feature = torch.from_numpy(feature).float()
            feature = feature.cuda(non_blocking=True)
            feature= feature.unsqueeze(dim=0)
            connect_mask_box = properties['connect_mask_box']
            #print(k, data.shape)
            data[-1][data[-1] == -1] = 0
            pred = self.predict_preprocessed_data_return_cls_and_softmax(data[:-1],feature,connect_mask_box,
                                                                                 do_mirroring=do_mirroring,
                                                                                 mirror_axes=mirror_axes,
                                                                                 use_sliding_window=use_sliding_window,
                                                                                 step_size=step_size,
                                                                                 use_gaussian=use_gaussian,
                                                                                 all_in_gpu=all_in_gpu,
                                                                                 mixed_precision=self.fp16)
            if not isinstance(pred, torch.Tensor):
                pred = torch.from_numpy(pred).float()
            softmax_pred = softmax_helper(pred)
            #output_class = output_class_softmax.argmax(1)
            """print("////////pred.shape//////")
            print(pred)
            print("////////softmax_pred//////")
            print(softmax_pred)"""
            output_class = softmax_pred.argmax(1)
            """ print("/////////class target//////////")
            print(class_target)
            print("////////output_class//////")
            print(output_class)"""
            if output_class == class_target:
                if output_class == 0:
                    cls_tn+=1
                else:
                    cls_tp+=1
            else:
                if output_class ==0:
                    cls_fn+=1
                else:
                    cls_fp+=1
            target = open(output_folder + '/%s.txt'%k , 'w')  # 打开目的文件
            target.write("class_label:"+str(class_target)+"\nclass_predict:"+str(output_class))
        acc = (cls_tp+cls_tn)/(cls_fp+cls_fn+cls_tp+cls_tn+1e-8)
        rec = (cls_tp)/(cls_tp+cls_fn+1e-8)
        pre = (cls_tp)/(cls_tp+cls_fp+1e-8)
        f1 = (rec*pre*2)/(rec+pre+1e-8)
        target = open(output_folder + '/metrics.txt', 'w')  # 打开目的文件
        target.write("acc:"+ str(acc)+"\nrec:"+ str(rec)+"\npre:"+ str(pre)+ "\nf1:"+ str(f1))
                #softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        self.network.train(current_mode)

        self.network.do_ds = ds
        #return ret
    def preprocess_predict_nifti(self, input_files: List[str], output_file: str = None,
                                 softmax_ouput_file: str = None, mixed_precision: bool = True) -> None:
        """
        Use this to predict new data
        :param input_files:
        :param output_file:
        :param softmax_ouput_file:
        :param mixed_precision:
        :return:
        """
        print("preprocessing...")
        d, s, properties = self.preprocess_patient(input_files)
        print("predicting...")
        pred = self.predict_preprocessed_data_return_cls_and_softmax(d, do_mirroring=self.data_aug_params["do_mirror"],
                                                                     mirror_axes=self.data_aug_params['mirror_axes'],
                                                                     use_sliding_window=True, step_size=0.5,
                                                                     use_gaussian=True, pad_border_mode='constant',
                                                                     pad_kwargs={'constant_values': 0},
                                                                     verbose=True, all_in_gpu=False,
                                                                     mixed_precision=mixed_precision)[1]
        pred = pred.transpose([0] + [i + 1 for i in self.transpose_backward])

        if 'segmentation_export_params' in self.plans.keys():
            force_separate_z = self.plans['segmentation_export_params']['force_separate_z']
            interpolation_order = self.plans['segmentation_export_params']['interpolation_order']
            interpolation_order_z = self.plans['segmentation_export_params']['interpolation_order_z']
        else:
            force_separate_z = None
            interpolation_order = 1
            interpolation_order_z = 0

        print("resampling to original spacing and nifti export...")
        save_segmentation_nifti_from_softmax(pred, output_file, properties, interpolation_order,
                                             self.regions_class_order, None, None, softmax_ouput_file,
                                             None, force_separate_z=force_separate_z,
                                             interpolation_order_z=interpolation_order_z)
        print("done")

    def predict_preprocessed_data_return_cls_and_softmax(self, data: np.ndarray,feature, connect_mask_box,do_mirroring: bool = True,
                                                         mirror_axes: Tuple[int] = None,
                                                         use_sliding_window: bool = True, step_size: float = 0.5,
                                                         use_gaussian: bool = True, pad_border_mode: str = 'constant',
                                                         pad_kwargs: dict = None, all_in_gpu: bool = False,
                                                         verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param data:
        :param do_mirroring:
        :param mirror_axes:
        :param use_sliding_window:
        :param step_size:
        :param use_gaussian:
        :param pad_border_mode:
        :param pad_kwargs:
        :param all_in_gpu:
        :param verbose:
        :return:
        """
        if pad_border_mode == 'constant' and pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        if do_mirroring and mirror_axes is None:
            mirror_axes = self.data_aug_params['mirror_axes']

        if do_mirroring:
            assert self.data_aug_params["do_mirror"], "Cannot do mirroring as test time augmentation when training " \
                                                      "was done without mirroring"

        valid = list((ClassficationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))
        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(data,feature,connect_mask_box, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        """print("///////////ret////////////////")
        print(ret)"""
        self.network.train(current_mode)
        return ret

    def load_checkpoint_ram(self, checkpoint, train=True):
        """
        used for if the checkpoint is already in ram
        :param checkpoint:
        :param train:
        :return:
        """
        if not self.was_initialized:
            self.initialize(train)

        new_state_dict = OrderedDict()
        print("new_state_dict:", new_state_dict)

        curr_state_dict_keys = list(self.network.state_dict().keys())
        print("curr_state_dict_keys:", curr_state_dict_keys)

        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        if self.fp16:
            self._maybe_init_amp()
            if 'amp_grad_scaler' in checkpoint.keys():
                self.amp_grad_scaler.load_state_dict(checkpoint['amp_grad_scaler'])

        self.network.load_state_dict(new_state_dict)
        self.epoch = checkpoint['epoch']
        if train:
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            if optimizer_state_dict is not None:
                self.optimizer.load_state_dict(optimizer_state_dict)

            if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'load_state_dict') and checkpoint[
                'lr_scheduler_state_dict'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

            if issubclass(self.lr_scheduler.__class__, _LRScheduler):
                self.lr_scheduler.step(self.epoch)

        self.all_tr_losses, self.all_val_losses, self.all_val_losses_tr_mode, self.all_val_eval_metrics = checkpoint[
            'plot_stuff']

        # load best loss (if present)
        if 'best_stuff' in checkpoint.keys():
            self.best_epoch_based_on_MA_tr_loss, self.best_MA_tr_loss_for_patience, self.best_val_eval_criterion_MA = \
            checkpoint[
                'best_stuff']

        # after the training is done, the epoch is incremented one more time in my old code. This results in
        # self.epoch = 1001 for old trained models when the epoch is actually 1000. This causes issues because
        # len(self.all_tr_losses) = 1000 and the plot function will fail. We can easily detect and correct that here
        if self.epoch != len(self.all_tr_losses):
            self.print_to_log_file("WARNING in loading checkpoint: self.epoch != len(self.all_tr_losses). This is "
                                   "due to an old bug and should only appear when you are loading old models. New "
                                   "models should have this fixed! self.epoch is now set to len(self.all_tr_losses)")
            self.epoch = len(self.all_tr_losses)
            self.all_tr_losses = self.all_tr_losses[:self.epoch]
            self.all_val_losses = self.all_val_losses[:self.epoch]
            self.all_val_losses_tr_mode = self.all_val_losses_tr_mode[:self.epoch]
            self.all_val_eval_metrics = self.all_val_eval_metrics[:self.epoch]

        self._maybe_init_amp()

    def _maybe_init_amp(self):
        if self.fp16 and self.amp_grad_scaler is None:
            self.amp_grad_scaler = GradScaler()