import splitfolders

splitfolders.ratio("dataset_aft/", output="divide_dataset_aft/",
    seed=42, ratio=(.8, .1, .1), group_prefix=None, move=False)