import torch
import torch.nn as nn
import time
import numpy as np
import shutil
from typing import Tuple, List
from simple_frame.network_architecture.neural_network import SegmentationNetwork
from multiprocessing import Pool
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
    def __init__(self, fold, data_root, out_path, output_fold):
        super(Trainer, self).__init__()
        self.fold = fold
        self.data_root = data_root
        self.dataset_directory = out_path
        self.output_checkpoints = self.output_folder = output_fold
        self.was_initialized = False
        self.log_file = self.use_mask_for_norm = None
        self.batch_dice = False
        self.seg_loss = JI_and_Focal_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False, 'square': False})
        self.class_loss = torch.nn.CrossEntropyLoss().cuda()
        self.aw1 = AutomaticWeightedLoss(1)
        self.fp16 = False

    def initialize(self, training=True):
        self.batch_size = 8
        self.patch_size = np.array([32,128,128]).astype(int)
        self.do_data_augmentation = True
        self.num_classes = 2
        self.setup_data_aug_params()
        self.initial_lr = 5e-3
        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 10
        self.weight_decay = 2e-5

        self.oversample_foreground_percent = 0.33
        self.pad_all_sides = None

        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.save_every = 50
        self.save_latest_only = True
        self.max_num_epochs = 1000
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
        self.all_val_eval_metrics_dc =[]

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

        if training:
            self.train_data, self.val_data = self.get_data()
            unpack_dataset(self.data_root)
            self.train_data, self.val_data = get_default_augmentation(self.train_data, self.val_data,
                                                                 self.data_aug_params[
                                                                     'patch_size_for_spatialtransform'],
                                                                 self.data_aug_params)

        #self.network = MTLN3D().cuda()
        self.network = VNet_class(num_classes=self.num_classes, deep_supervision=True).cuda()
        self.optimizer = torch.optim.Adam(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")
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
                             pad_mode="constant", pad_sides=self.pad_all_sides)
        dl_val = DataLoader3D(self.dataset_val, self.patch_size, self.patch_size, self.batch_size, False,
                              oversample_foreground_percent=self.oversample_foreground_percent,
                              pad_mode="constant", pad_sides=self.pad_all_sides)
        return dl_tr, dl_val

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

        #output = self.network(data)
        del data
        l0 = self.seg_loss(output, target)

        l = self.aw1.cuda()(l0)
        l = l0
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

    def finish_online_evaluation_onlyseg(self):
        ###classification

        # self.all_val_eval_metrics.append(np.mean(global_acc_per_class))


        ###segmentation
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]



        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))
        self.all_val_eval_metrics_dc.append(np.mean(global_dc_per_class))

        self.print_to_log_file("Val global dc per class:", str(global_dc_per_class))

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []



    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)
        self.print_to_log_file("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def load_best_checkpoint(self, train=True):
        if self.fold is None:
            raise RuntimeError("Cannot load best checkpoint if self.fold is None")
        if isfile(self.output_folder+'/'+"model_best.model"):
            self.load_checkpoint(self.output_folder+'/'+ "model_best.model", train=train)
        else:
            self.print_to_log_file("WARNING! model_best.model does not exist! Cannot load best checkpoint. Falling "
                                   "back to load_latest_checkpoint")
            self.load_latest_checkpoint(train)

    def load_latest_checkpoint(self, train=True):
        if isfile(self.output_folder+'/'+"model_final_checkpoint.model"):
            return self.load_checkpoint(self.output_folder+'/'+"model_final_checkpoint.model", train=train)
        if isfile(self.output_folder+'/'+ "model_latest.model"):
            return self.load_checkpoint(self.output_folder+'/'+ "model_latest.model", train=train)
        if isfile(self.output_folder+'/'+ "model_best.model"):
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
                self.best_val_eval_criterion_MA_class = self.val_eval_criterion_MA_class

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)
            self.print_to_log_file(
                "current best_val_eval_criterion_MA_class is %.4f0" % self.best_val_eval_criterion_MA_class)
            self.print_to_log_file("current val_eval_criterion_MA_class is %.4f" % self.val_eval_criterion_MA_class)

            if self.val_eval_criterion_MA_class > self.best_val_eval_criterion_MA_class:
                self.best_val_eval_criterion_MA_class = self.val_eval_criterion_MA_class
                self.print_to_log_file("saving best classification epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_best_class.model"))

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                self.print_to_log_file("saving best segmentation epoch checkpoint...")
                self.save_checkpoint(join(self.output_folder, "model_best_seg.model"))

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
            if len(self.all_val_eval_metrics_seg) == len(x_values):
                ax2.plot(x_values, self.all_val_eval_metrics_dc, color='black', ls='--', label="dc")
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
        self.finish_online_evaluation_onlyseg()  # does not have to do anything, but can be used to update self.all_val_eval_
        # metrics

        self.plot_progress()

        self.maybe_update_lr()

        self.maybe_save_checkpoint()

        self.update_eval_criterion_MA()

        continue_training = self.manage_patience()
        return continue_training
    def run_online_evaluation_onlyseg(self, output, target):
        with torch.no_grad():
            ###classification
            ###segmentation
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))


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
            if self.epoch % 5 == 0:
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
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
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

        for k in self.dataset_val.keys():
            properties = load_pickle(self.dataset[k]['properties_file'])
            fname = properties['list_of_data_files'][0].split("/")[-1][:-12]
            if overwrite or (not isfile(output_folder + "/" + fname + ".nii.gz")) or \
                    (save_softmax and not isfile(output_folder + "/" + fname + ".npz")):
                data = np.load(self.dataset[k]['data_file'])['data']

                print(k, data.shape)
                data[-1][data[-1] == -1] = 0

                softmax_pred = self.predict_preprocessed_data_return_seg_and_softmax(data[:-1],
                                                                                     do_mirroring=do_mirroring,
                                                                                     mirror_axes=mirror_axes,
                                                                                     use_sliding_window=use_sliding_window,
                                                                                     step_size=step_size,
                                                                                     use_gaussian=use_gaussian,
                                                                                     all_in_gpu=all_in_gpu,
                                                                                     mixed_precision=self.fp16)[1]

                softmax_pred = softmax_pred.transpose([0] + [i + 1 for i in self.transpose_backward])

                if save_softmax:
                    softmax_fname = output_folder + "/" + fname + ".npz"
                else:
                    softmax_fname = None

                """There is a problem with python process communication that prevents us from communicating obejcts
                larger than 2 GB between processes (basically when the length of the pickle string that will be sent is
                communicated by the multiprocessing.Pipe object then the placeholder (\%i I think) does not allow for long
                enough strings (lol). This could be fixed by changing i to l (for long) but that would require manually
                patching system python code. We circumvent that problem here by saving softmax_pred to a npy file that will
                then be read (and finally deleted) by the Process. save_segmentation_nifti_from_softmax can take either
                filename or np.ndarray and will handle this automatically"""
                if np.prod(softmax_pred.shape) > (2e9 / 4 * 0.85):  # *0.85 just to be save
                    np.save(output_folder + "/" + fname + ".npy", softmax_pred)
                    softmax_pred = output_folder + "/" + fname + ".npy"

                results.append(export_pool.starmap_async(save_segmentation_nifti_from_softmax,
                                                         ((softmax_pred, output_folder + "/" + fname + ".nii.gz",
                                                           properties, interpolation_order, self.regions_class_order,
                                                           None, None,
                                                           softmax_fname, None, force_separate_z,
                                                           interpolation_order_z),
                                                          )
                                                         )
                               )

            pred_gt_tuples.append([output_folder + "/" + fname + ".nii.gz",
                                   self.gt_niftis_folder + "/" + fname + ".nii.gz"])

        _ = [i.get() for i in results]
        self.print_to_log_file("finished prediction")

        # evaluate raw predictions
        self.print_to_log_file("evaluation of raw predictions")
        task = self.dataset_directory.split("/")[-1]
        job_name = self.experiment_name
        _ = aggregate_scores(pred_gt_tuples, labels=list(range(self.num_classes)),
                             json_output_file=output_folder + "/" + "summary.json",
                             json_name=job_name + " val tiled %s" % (str(use_sliding_window)),
                             json_author="Fabian",
                             json_task=task, num_threads=4)

        if run_postprocessing_on_folds:
            # in the old simple_frame we would stop here. Now we add a postprocessing. This postprocessing can remove everything
            # except the largest connected component for each class. To see if this improves results, we do this for all
            # classes and then rerun the evaluation. Those classes for which this resulted in an improved dice score will
            # have this applied during inference as well
            self.print_to_log_file("determining postprocessing")
            determine_postprocessing(self.output_folder, self.gt_niftis_folder, validation_folder_name,
                                     final_subf_name=validation_folder_name + "_postprocessed", debug=debug)
            # after this the final predictions for the vlaidation set can be found in validation_folder_name_base + "_postprocessed"
            # They are always in that folder, even if no postprocessing as applied!

        # detemining postprocesing on a per-fold basis may be OK for this fold but what if another fold finds another
        # postprocesing to be better? In this case we need to consolidate. At the time the consolidation is going to be
        # done we won't know what self.gt_niftis_folder was, so now we copy all the niftis into a separate folder to
        # be used later
        gt_nifti_folder = self.output_folder_base + "/" + "gt_niftis"
        if not os.path.isdir(gt_nifti_folder):
            os.makedirs(gt_nifti_folder)
        for f in subfiles(self.gt_niftis_folder, suffix=".nii.gz"):
            success = False
            attempts = 0
            e = None
            while not success and attempts < 10:
                try:
                    shutil.copy(f, gt_nifti_folder)
                    success = True
                except OSError as e:
                    attempts += 1
                    sleep(1)
            if not success:
                print("Could not copy gt nifti file %s into folder %s" % (f, gt_nifti_folder))
                if e is not None:
                    raise e

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
        pred = self.predict_preprocessed_data_return_seg_and_softmax(d, do_mirroring=self.data_aug_params["do_mirror"],
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

    def predict_preprocessed_data_return_seg_and_softmax(self, data: np.ndarray, do_mirroring: bool = True,
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

        valid = list((SegmentationNetwork, nn.DataParallel))
        assert isinstance(self.network, tuple(valid))

        current_mode = self.network.training
        self.network.eval()
        ret = self.network.predict_3D(data, do_mirroring=do_mirroring, mirror_axes=mirror_axes,
                                      use_sliding_window=use_sliding_window, step_size=step_size,
                                      patch_size=self.patch_size, regions_class_order=self.regions_class_order,
                                      use_gaussian=use_gaussian, pad_border_mode=pad_border_mode,
                                      pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose,
                                      mixed_precision=mixed_precision)
        self.network.train(current_mode)
        return ret