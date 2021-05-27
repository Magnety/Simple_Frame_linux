import numpy as np
from scipy.ndimage import label
import SimpleITK as sitk
def connect_mask_box(image: np.ndarray, for_which_classes: list=None, minimum__size=100):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]
    assert 0 not in for_which_classes, "cannot remove background"
    all_loc = {}
    print("for which:",for_which_classes)
    for c in for_which_classes:
        print(c)
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        print(c)

        lmap, num_objects = label(mask.astype(int))
        print("num_objects:",num_objects)
        print("lamap.shape:",lmap.shape)
        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() #* volume_per_voxel
        print(object_sizes)
        """
        largest_removed[c] = None
        kept_size[c] = None
        """
        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            all_loc[c] = []
            for object_id in range(1, num_objects + 1):
                loc = np.argwhere(lmap == object_id)
                # we only remove objects that are not the largest
                if object_sizes[object_id] >= minimum__size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    x_min = min(loc[:, 0])
                    x_max = max(loc[:, 0])
                    y_min = min(loc[:, 1])
                    y_max = max(loc[:, 1])
                    z_min = min(loc[:, 2])
                    z_max = max(loc[:, 2])
                    each_loc = [x_min,x_max,y_min,y_max,z_min,z_max]
                    all_loc[c].append(each_loc)
    print(all_loc)
    return all_loc

input_path ="G:/Spine_Compitition/Spine/Mask/mask_case2.nii.gz"
label_data = sitk.ReadImage(input_path)
label_np = sitk.GetArrayFromImage(label_data)
connect_mask_box(label_np)
