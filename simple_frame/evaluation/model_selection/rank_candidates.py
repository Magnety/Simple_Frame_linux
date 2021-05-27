#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from simple_frame.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir =  network_training_output_dir+"/"+"summary_jsons_fold0_new"
    output_file =  network_training_output_dir+"/"+ "summary.csv"

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "simple_framePlans"

    overwrite_plans = {
        'simple_frameTrainerV2_2': ["simple_framePlans", "simple_framePlansisoPatchesInVoxels"], # r
        'simple_frameTrainerV2': ["simple_framePlansnonCT", "simple_framePlansCT2", "simple_framePlansallConv3x3",
                            "simple_framePlansfixedisoPatchesInVoxels", "simple_framePlanstargetSpacingForAnisoAxis",
                            "simple_framePlanspoolBasedOnSpacing", "simple_framePlansfixedisoPatchesInmm", "simple_framePlansv2.1"],
        'simple_frameTrainerV2_warmup': ["simple_framePlans", "simple_framePlansv2.1", "simple_framePlansv2.1_big", "simple_framePlansv2.1_verybig"],
        'simple_frameTrainerV2_cycleAtEnd': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_cycleAtEnd2': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_reduceMomentumDuringTraining': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_graduallyTransitionFromCEToDice': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_independentScalePerAxis': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_Mish': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_Ranger_lr3en4': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_fp32': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_GN': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_momentum098': ["simple_framePlans", "simple_framePlansv2.1"],
        'simple_frameTrainerV2_momentum09': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_DP': ["simple_framePlansv2.1_verybig"],
        'simple_frameTrainerV2_DDP': ["simple_framePlansv2.1_verybig"],
        'simple_frameTrainerV2_FRN': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_resample33': ["simple_framePlansv2.3"],
        'simple_frameTrainerV2_O2': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_ResencUNet': ["simple_framePlans_FabiansResUNet_v2.1"],
        'simple_frameTrainerV2_DA2': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_allConv3x3': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_ForceBD': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_ForceSD': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_LReLU_slope_2en1': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_lReLU_convReLUIN': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_ReLU': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_ReLU_biasInSegOutput': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_ReLU_convReLUIN': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_lReLU_biasInSegOutput': ["simple_framePlansv2.1"],
        #'simple_frameTrainerV2_Loss_MCC': ["simple_framePlansv2.1"],
        #'simple_frameTrainerV2_Loss_MCCnoBG': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_Loss_DicewithBG': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_Loss_Dice_LR1en3': ["simple_framePlansv2.1"],
        'simple_frameTrainerV2_Loss_Dice': ["simple_framePlans", "simple_framePlansv2.1"],
        'simple_frameTrainerV2_Loss_DicewithBG_LR1en3': ["simple_framePlansv2.1"],
        # 'simple_frameTrainerV2_fp32': ["simple_framePlansv2.1"],
        # 'simple_frameTrainerV2_fp32': ["simple_framePlansv2.1"],
        # 'simple_frameTrainerV2_fp32': ["simple_framePlansv2.1"],
        # 'simple_frameTrainerV2_fp32': ["simple_framePlansv2.1"],
        # 'simple_frameTrainerV2_fp32': ["simple_framePlansv2.1"],

    }

    trainers = ['simple_frameTrainer'] + ['simple_frameTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'simple_frameTrainerNewCandidate24_2',
        'simple_frameTrainerNewCandidate24_3',
        'simple_frameTrainerNewCandidate26_2',
        'simple_frameTrainerNewCandidate27_2',
        'simple_frameTrainerNewCandidate23_always3DDA',
        'simple_frameTrainerNewCandidate23_corrInit',
        'simple_frameTrainerNewCandidate23_noOversampling',
        'simple_frameTrainerNewCandidate23_softDS',
        'simple_frameTrainerNewCandidate23_softDS2',
        'simple_frameTrainerNewCandidate23_softDS3',
        'simple_frameTrainerNewCandidate23_softDS4',
        'simple_frameTrainerNewCandidate23_2_fp16',
        'simple_frameTrainerNewCandidate23_2',
        'simple_frameTrainerVer2',
        'simple_frameTrainerV2_2',
        'simple_frameTrainerV2_3',
        'simple_frameTrainerV2_3_CE_GDL',
        'simple_frameTrainerV2_3_dcTopk10',
        'simple_frameTrainerV2_3_dcTopk20',
        'simple_frameTrainerV2_3_fp16',
        'simple_frameTrainerV2_3_softDS4',
        'simple_frameTrainerV2_3_softDS4_clean',
        'simple_frameTrainerV2_3_softDS4_clean_improvedDA',
        'simple_frameTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'simple_frameTrainerV2_3_softDS4_radam',
        'simple_frameTrainerV2_3_softDS4_radam_lowerLR',

        'simple_frameTrainerV2_2_schedule',
        'simple_frameTrainerV2_2_schedule2',
        'simple_frameTrainerV2_2_clean',
        'simple_frameTrainerV2_2_clean_improvedDA_newElDef',

        'simple_frameTrainerV2_2_fixes', # running
        'simple_frameTrainerV2_BN', # running
        'simple_frameTrainerV2_noDeepSupervision', # running
        'simple_frameTrainerV2_softDeepSupervision', # running
        'simple_frameTrainerV2_noDataAugmentation', # running
        'simple_frameTrainerV2_Loss_CE', # running
        'simple_frameTrainerV2_Loss_CEGDL',
        'simple_frameTrainerV2_Loss_Dice',
        'simple_frameTrainerV2_Loss_DiceTopK10',
        'simple_frameTrainerV2_Loss_TopK10',
        'simple_frameTrainerV2_Adam', # running
        'simple_frameTrainerV2_Adam_simple_frameTrainerlr', # running
        'simple_frameTrainerV2_SGD_ReduceOnPlateau', # running
        'simple_frameTrainerV2_SGD_lr1en1', # running
        'simple_frameTrainerV2_SGD_lr1en3', # running
        'simple_frameTrainerV2_fixedNonlin', # running
        'simple_frameTrainerV2_GeLU', # running
        'simple_frameTrainerV2_3ConvPerStage',
        'simple_frameTrainerV2_NoNormalization',
        'simple_frameTrainerV2_Adam_ReduceOnPlateau',
        'simple_frameTrainerV2_fp16',
        'simple_frameTrainerV2', # see overwrite_plans
        'simple_frameTrainerV2_noMirroring',
        'simple_frameTrainerV2_momentum09',
        'simple_frameTrainerV2_momentum095',
        'simple_frameTrainerV2_momentum098',
        'simple_frameTrainerV2_warmup',
        'simple_frameTrainerV2_Loss_Dice_LR1en3',
        'simple_frameTrainerV2_NoNormalization_lr1en3',
        'simple_frameTrainerV2_Loss_Dice_squared',
        'simple_frameTrainerV2_newElDef',
        'simple_frameTrainerV2_fp32',
        'simple_frameTrainerV2_cycleAtEnd',
        'simple_frameTrainerV2_reduceMomentumDuringTraining',
        'simple_frameTrainerV2_graduallyTransitionFromCEToDice',
        'simple_frameTrainerV2_insaneDA',
        'simple_frameTrainerV2_independentScalePerAxis',
        'simple_frameTrainerV2_Mish',
        'simple_frameTrainerV2_Ranger_lr3en4',
        'simple_frameTrainerV2_cycleAtEnd2',
        'simple_frameTrainerV2_GN',
        'simple_frameTrainerV2_DP',
        'simple_frameTrainerV2_FRN',
        'simple_frameTrainerV2_resample33',
        'simple_frameTrainerV2_O2',
        'simple_frameTrainerV2_ResencUNet',
        'simple_frameTrainerV2_DA2',
        'simple_frameTrainerV2_allConv3x3',
        'simple_frameTrainerV2_ForceBD',
        'simple_frameTrainerV2_ForceSD',
        'simple_frameTrainerV2_ReLU',
        'simple_frameTrainerV2_LReLU_slope_2en1',
        'simple_frameTrainerV2_lReLU_convReLUIN',
        'simple_frameTrainerV2_ReLU_biasInSegOutput',
        'simple_frameTrainerV2_ReLU_convReLUIN',
        'simple_frameTrainerV2_lReLU_biasInSegOutput',
        'simple_frameTrainerV2_Loss_DicewithBG_LR1en3',
        #'simple_frameTrainerV2_Loss_MCCnoBG',
        'simple_frameTrainerV2_Loss_DicewithBG',
        # 'simple_frameTrainerV2_Loss_Dice_LR1en3',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
        # 'simple_frameTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = summary_files_dir+"/"+ "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str)
                        if not isfile(summary_file):
                            summary_file =  summary_files_dir+"/"+ "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str)
                            if not isfile(summary_file):
                                summary_file =  summary_files_dir+"/"+ "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str)
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
