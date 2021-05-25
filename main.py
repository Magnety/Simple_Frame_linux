import os
import warnings
import argparse

from training.trainer import Trainer
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help='0, 1, ..., 5 or \'all\'')
    parser.add_argument('--data_root', default='G:/tuFramework_data_raw_base/tuFramework_preprocessed/Task100_Breast_c_f_noclsmask/tuData_plans_v2.1_stage0',
                        type=str, help='root directory path of data')
    parser.add_argument("--valbest", required=False, default=False, help="select the best training weights to test",
                        action="store_true")
    parser.add_argument("-val", "--validation_only", help="use this if you want to only run the validation",
                        action="store_true")
    parser.add_argument("-c", "--continue_training", help="use this if you want to continue a training",
                        action="store_true")
    parser.add_argument("--val_folder", required=False, default="validation_raw",
                        help="name of the validation folder. No need to use this for most people")
    sys.argv = ['main.py','all']
    args = parser.parse_args()
    data_root = args.data_root
    validation_only = args.validation_only
    val_folder = args.val_folder
    fold = args.fold
    valbest = args.valbest

    if fold == 'all':
        pass
    else:
        fold = int(fold)
    out_path = "G:/tuFramework_data_store"
    out_checkpoints = os.path.join(out_path, "Fold" + str(fold) + "_checkpoints")
    if not os.path.exists(str(out_checkpoints)):
        os.mkdir(str(out_checkpoints))


    model_trainer = Trainer(fold, data_root, out_path, out_checkpoints)
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
    """model_trainer.validate(save_softmax=args.npz, validation_folder_name=val_folder,
                     run_postprocessing_on_folds=not disable_postprocessing_on_folds)"""
    #
    # model_trainer.initialize(not test_best)
    #
    # if test_best:
    #     model_trainer.load_checkpoint(train=False)
    #     model_trainer.validate(validation_restore_path="validation")

