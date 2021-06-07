import os
import warnings
import argparse

import torch.cuda

from simple_frame.training.cls_trainer import Trainer
from simple_frame.utilities.task_name_id_conversion import convert_id_to_task_name
from simple_frame.paths import *
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("task", help="can be task name or task id")

    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')

    parser.add_argument('--data_root', default='/home/ubuntu/liuyiyao/Simple_Frame_data_raw_base/preprocessed',
                        type=str, help='root directory path of data')
    parser.add_argument("--valbest", required=False, default=True, help="select the best training weights to test",
                        action="store_true")
    parser.add_argument("-val", "--validation_only",default=False, help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    parser.add_argument("--npz", required=False, default=False, action="store_true",help="if set then tuframework will ")
    #print(torch.cuda.is_available())

    sys.argv = ['main_cls.py','100','2']
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    args = parser.parse_args()
    data_root = args.data_root
    validation_only = args.validation_only #validate True
    val_folder = args.val_folder
    fold = args.fold
    valbest = args.valbest
    task = args.task
    if fold == 'all':
        pass
    else:
        fold = int(fold)
    if not task.startswith("Task"):
        task_id = int(task)
        task = convert_id_to_task_name(task_id)
    out_path = network_training_output_dir_base+'/'+task
    raw_path = raw_data+'/'+task
    data_root = data_root +'/' +task
    out_checkpoints = os.path.join(out_path, "Fold" + str(fold) + "_checkpoints")
    if not os.path.exists(str(out_checkpoints)):
        os.makedirs(str(out_checkpoints))

#seg or cls
    model_trainer = Trainer(fold,1, data_root, out_path, out_checkpoints,raw_path,val_folder)

    #validate only
    #model_trainer.initialize(not validation_only)

    if not validation_only:
        if args.continue_training:
            model_trainer.load_latest_checkpoint()
        model_trainer.run_trainer()
    else:
        if valbest:
            model_trainer.load_best_checkpoint(train=False)
        else:
            model_trainer.load_final_checkpoint(train=False)

    model_trainer.network.eval()

    # predict validation
    model_trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder)
    #
    # model_trainer.initialize(not test_best)
    #
    # if test_best:
    #     model_trainer.load_checkpoint(train=False)
    #     model_trainer.validate(validation_restore_path="validation")

