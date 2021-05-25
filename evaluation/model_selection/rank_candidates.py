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
from tuframework.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir =  network_training_output_dir+"/"+"summary_jsons_fold0_new"
    output_file =  network_training_output_dir+"/"+ "summary.csv"

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "tuframeworkPlans"

    overwrite_plans = {
        'tuframeworkTrainerV2_2': ["tuframeworkPlans", "tuframeworkPlansisoPatchesInVoxels"], # r
        'tuframeworkTrainerV2': ["tuframeworkPlansnonCT", "tuframeworkPlansCT2", "tuframeworkPlansallConv3x3",
                            "tuframeworkPlansfixedisoPatchesInVoxels", "tuframeworkPlanstargetSpacingForAnisoAxis",
                            "tuframeworkPlanspoolBasedOnSpacing", "tuframeworkPlansfixedisoPatchesInmm", "tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_warmup': ["tuframeworkPlans", "tuframeworkPlansv2.1", "tuframeworkPlansv2.1_big", "tuframeworkPlansv2.1_verybig"],
        'tuframeworkTrainerV2_cycleAtEnd': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_cycleAtEnd2': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_reduceMomentumDuringTraining': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_graduallyTransitionFromCEToDice': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_independentScalePerAxis': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_Mish': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_Ranger_lr3en4': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_fp32': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_GN': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_momentum098': ["tuframeworkPlans", "tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_momentum09': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_DP': ["tuframeworkPlansv2.1_verybig"],
        'tuframeworkTrainerV2_DDP': ["tuframeworkPlansv2.1_verybig"],
        'tuframeworkTrainerV2_FRN': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_resample33': ["tuframeworkPlansv2.3"],
        'tuframeworkTrainerV2_O2': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_ResencUNet': ["tuframeworkPlans_FabiansResUNet_v2.1"],
        'tuframeworkTrainerV2_DA2': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_allConv3x3': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_ForceBD': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_ForceSD': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_LReLU_slope_2en1': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_lReLU_convReLUIN': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_ReLU': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_ReLU_biasInSegOutput': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_ReLU_convReLUIN': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_lReLU_biasInSegOutput': ["tuframeworkPlansv2.1"],
        #'tuframeworkTrainerV2_Loss_MCC': ["tuframeworkPlansv2.1"],
        #'tuframeworkTrainerV2_Loss_MCCnoBG': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_Loss_DicewithBG': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_Loss_Dice_LR1en3': ["tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_Loss_Dice': ["tuframeworkPlans", "tuframeworkPlansv2.1"],
        'tuframeworkTrainerV2_Loss_DicewithBG_LR1en3': ["tuframeworkPlansv2.1"],
        # 'tuframeworkTrainerV2_fp32': ["tuframeworkPlansv2.1"],
        # 'tuframeworkTrainerV2_fp32': ["tuframeworkPlansv2.1"],
        # 'tuframeworkTrainerV2_fp32': ["tuframeworkPlansv2.1"],
        # 'tuframeworkTrainerV2_fp32': ["tuframeworkPlansv2.1"],
        # 'tuframeworkTrainerV2_fp32': ["tuframeworkPlansv2.1"],

    }

    trainers = ['tuframeworkTrainer'] + ['tuframeworkTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'tuframeworkTrainerNewCandidate24_2',
        'tuframeworkTrainerNewCandidate24_3',
        'tuframeworkTrainerNewCandidate26_2',
        'tuframeworkTrainerNewCandidate27_2',
        'tuframeworkTrainerNewCandidate23_always3DDA',
        'tuframeworkTrainerNewCandidate23_corrInit',
        'tuframeworkTrainerNewCandidate23_noOversampling',
        'tuframeworkTrainerNewCandidate23_softDS',
        'tuframeworkTrainerNewCandidate23_softDS2',
        'tuframeworkTrainerNewCandidate23_softDS3',
        'tuframeworkTrainerNewCandidate23_softDS4',
        'tuframeworkTrainerNewCandidate23_2_fp16',
        'tuframeworkTrainerNewCandidate23_2',
        'tuframeworkTrainerVer2',
        'tuframeworkTrainerV2_2',
        'tuframeworkTrainerV2_3',
        'tuframeworkTrainerV2_3_CE_GDL',
        'tuframeworkTrainerV2_3_dcTopk10',
        'tuframeworkTrainerV2_3_dcTopk20',
        'tuframeworkTrainerV2_3_fp16',
        'tuframeworkTrainerV2_3_softDS4',
        'tuframeworkTrainerV2_3_softDS4_clean',
        'tuframeworkTrainerV2_3_softDS4_clean_improvedDA',
        'tuframeworkTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'tuframeworkTrainerV2_3_softDS4_radam',
        'tuframeworkTrainerV2_3_softDS4_radam_lowerLR',

        'tuframeworkTrainerV2_2_schedule',
        'tuframeworkTrainerV2_2_schedule2',
        'tuframeworkTrainerV2_2_clean',
        'tuframeworkTrainerV2_2_clean_improvedDA_newElDef',

        'tuframeworkTrainerV2_2_fixes', # running
        'tuframeworkTrainerV2_BN', # running
        'tuframeworkTrainerV2_noDeepSupervision', # running
        'tuframeworkTrainerV2_softDeepSupervision', # running
        'tuframeworkTrainerV2_noDataAugmentation', # running
        'tuframeworkTrainerV2_Loss_CE', # running
        'tuframeworkTrainerV2_Loss_CEGDL',
        'tuframeworkTrainerV2_Loss_Dice',
        'tuframeworkTrainerV2_Loss_DiceTopK10',
        'tuframeworkTrainerV2_Loss_TopK10',
        'tuframeworkTrainerV2_Adam', # running
        'tuframeworkTrainerV2_Adam_tuframeworkTrainerlr', # running
        'tuframeworkTrainerV2_SGD_ReduceOnPlateau', # running
        'tuframeworkTrainerV2_SGD_lr1en1', # running
        'tuframeworkTrainerV2_SGD_lr1en3', # running
        'tuframeworkTrainerV2_fixedNonlin', # running
        'tuframeworkTrainerV2_GeLU', # running
        'tuframeworkTrainerV2_3ConvPerStage',
        'tuframeworkTrainerV2_NoNormalization',
        'tuframeworkTrainerV2_Adam_ReduceOnPlateau',
        'tuframeworkTrainerV2_fp16',
        'tuframeworkTrainerV2', # see overwrite_plans
        'tuframeworkTrainerV2_noMirroring',
        'tuframeworkTrainerV2_momentum09',
        'tuframeworkTrainerV2_momentum095',
        'tuframeworkTrainerV2_momentum098',
        'tuframeworkTrainerV2_warmup',
        'tuframeworkTrainerV2_Loss_Dice_LR1en3',
        'tuframeworkTrainerV2_NoNormalization_lr1en3',
        'tuframeworkTrainerV2_Loss_Dice_squared',
        'tuframeworkTrainerV2_newElDef',
        'tuframeworkTrainerV2_fp32',
        'tuframeworkTrainerV2_cycleAtEnd',
        'tuframeworkTrainerV2_reduceMomentumDuringTraining',
        'tuframeworkTrainerV2_graduallyTransitionFromCEToDice',
        'tuframeworkTrainerV2_insaneDA',
        'tuframeworkTrainerV2_independentScalePerAxis',
        'tuframeworkTrainerV2_Mish',
        'tuframeworkTrainerV2_Ranger_lr3en4',
        'tuframeworkTrainerV2_cycleAtEnd2',
        'tuframeworkTrainerV2_GN',
        'tuframeworkTrainerV2_DP',
        'tuframeworkTrainerV2_FRN',
        'tuframeworkTrainerV2_resample33',
        'tuframeworkTrainerV2_O2',
        'tuframeworkTrainerV2_ResencUNet',
        'tuframeworkTrainerV2_DA2',
        'tuframeworkTrainerV2_allConv3x3',
        'tuframeworkTrainerV2_ForceBD',
        'tuframeworkTrainerV2_ForceSD',
        'tuframeworkTrainerV2_ReLU',
        'tuframeworkTrainerV2_LReLU_slope_2en1',
        'tuframeworkTrainerV2_lReLU_convReLUIN',
        'tuframeworkTrainerV2_ReLU_biasInSegOutput',
        'tuframeworkTrainerV2_ReLU_convReLUIN',
        'tuframeworkTrainerV2_lReLU_biasInSegOutput',
        'tuframeworkTrainerV2_Loss_DicewithBG_LR1en3',
        #'tuframeworkTrainerV2_Loss_MCCnoBG',
        'tuframeworkTrainerV2_Loss_DicewithBG',
        # 'tuframeworkTrainerV2_Loss_Dice_LR1en3',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
        # 'tuframeworkTrainerV2_Ranger_lr3en4',
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
