import splitfolders

splitfolders.ratio("../datasets/base_datasets/dataset_aug/", output="../datasets/divided_datasets/divide_dataset_aug/",
    seed=42, ratio=(.8, .1, .1), group_prefix=None, move=False)