import os
import numpy as np

class Config:
    new_filename = 'train.sh'
    cluster = False #True

    target =  "age" 
    experiment = "face-robustness" 

    # methods = ["vanilla"]
    methods = ["mixup"]

    config = "config.yml"
    trainfile = "train.py"
    label_shift = False
    spurious_correlation = False

    if label_shift and spurious_correlation:
        raise NotImplementedError


    if label_shift:
        experiment = "face-robustness" 
        comment = "label"
    elif spurious_correlation:
        comment = "spurious"
    else:
        comment="shift"


    baseline = ["baseline_race"]
    group_baseline_age = ["baseline_race", "baseline_gender"]
    group_baseline_spurious = ["spurious_correlations_baseline_young"]
    groups_race_helper = ["Black", "East_Asian", "Indian", "Latino_Hispanic", "Middle_Eastern" , "Southeast_Asian", "White" ]
    group_shift_race = ["data_shift_{}_{}".format(a, b) for a in groups_race_helper for b in [0.0, 0.25, 0.5, 0.75]]
    group_shift_gender = ["split_gen_1", "split_gen_2", "split_gen_3", "split_gen_4"]

    label_shift_race = ["left_label_shift_{}_{}".format(a, b) for a in groups_race_helper for b in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]]


    spurious_correlations1b = ["spurious_correlations_young_{}".format(b) for b in range(7)]  # 1b
    spurious_correlations2b = ["spurious_correlations_iff_young_{}".format(d) for d in range(7)]  # 2b
    
    group_race_sp = group_baseline_spurious  + spurious_correlations1b  + spurious_correlations2b 


    groups_target_age = group_baseline_age + group_shift_race + group_shift_gender
    groups_target_age_label_bias = baseline + label_shift_race





def main():

    if Config.cluster:
        python = 'python3.7 '
    else:
        python = 'python '

    with open(os.path.join(os.getcwd(), Config.new_filename), 'w') as file:

        if Config.target == "age":
            if not Config.label_shift  and not Config.spurious_correlation:
                groups = Config.groups_target_age
            elif Config.label_shift:
                groups = Config.groups_target_age_label_bias
            elif Config.spurious_correlation:
                groups = Config.group_race_sp


        for group in groups:
            file.writelines('\n#group: {}\n'.format(group))
            for method in Config.methods:
                for x in range(0,1):
                    temp_scale = "true"
                    if method=="vanilla":
                        num_ens="1"
                        mixup="false"
                        sequential="false"
                    elif method=="mixup":
                        num_ens="1"
                        mixup="true"
                        sequential="false"
                    elif method=="ensemble":
                        num_ens="3"
                        mixup="false"
                        sequential="true"

                    line = python + Config.trainfile +' -c ' + '"' + 'configs/' + Config.config + '"' + \
                        ' -experiment ' + Config.experiment +  \
                        ' -comment ' + Config.comment +  \
                        ' -lr 0.00001 -e 25' + \
                        ' -target '  + Config.target +  \
                        ' -group '  + group +  \
                        ' -sequential ' + sequential +  \
                        ' -num_ens ' + num_ens +  \
                        ' -mixup ' + mixup +  \
                        ' -temp_scale ' + temp_scale +  \
                        '\n'

                    file.writelines(line)



if __name__ == "__main__":
    main()


