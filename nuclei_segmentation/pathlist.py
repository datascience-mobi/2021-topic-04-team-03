from pathlib import Path

def pathlist(loadpath, searchstring_list):
    """Picture search.

    @param loadpath: path to folder
    @param searchstring_list: list of strings to be searched
    @return: list of paths to pictures with one of the elements in searchstring_list in filename.
    """

    path_list = []

    for searchstring in searchstring_list:
        for element in Path(loadpath).rglob(searchstring):
            path_list.append(element)

    return path_list


if __name__ == "_main_":
    N2DH_GOWT1_gt_list = pathlist("../Data/N2DH-GOWT1", "man_seg")
    print(N2DH_GOWT1_gt_list)
    N2DH_GOWT1_img_list = pathlist("../Data/N2DH-GOWT1", "t")
    print(N2DH_GOWT1_img_list)

    N2DL_HeLa_gt_list = pathlist("../Data/N2DL-HeLa", "man_seg")
    print(N2DL_HeLa_gt_list)
    N2DL_HeLa_img_list = pathlist("../Data/N2DL-HeLa", "t")
    print(N2DL_HeLa_img_list)

    NIH3T3_gt_list = pathlist("../Data/NIH3T3", ".png")
    print(NIH3T3_gt_list)
    NIH3T3_img_list = pathlist("../Data/NIH3T3", "dna-")
    print(NIH3T3_img_list)



