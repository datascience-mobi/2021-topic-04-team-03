from pathlib import Path


def pathlist(loadpath, searchstring_list):
    """Picture search.

    @param loadpath: path to folder
    @param searchstring_list: list of strings to be searched
    @return: list of paths to pictures with one of the elements in searchstring_list in filename.
    """

    pathlist = []

    for searchstring in searchstring_list:
        for element in Path(loadpath).rglob(searchstring):
            pathlist.append(element)

    return pathlist


if __name__ == "__main__":
    N2DH_GOWT1_gt_list = pathlist("../Data/N2DH-GOWT1/gt", ".tif")
    print(N2DH_GOWT1_gt_list)
    N2DH_GOWT1_img_list = pathlist("../Data/N2DH-GOWT1/img", ".tif")
    print(N2DH_GOWT1_img_list)

    N2DL_HeLa_gt_list = pathlist("../Data/N2DL-HeLa/gt", "m.tif")
    print(N2DL_HeLa_gt_list)
    N2DL_HeLa_img_list = pathlist("../Data/N2DL-HeLa/img", ".tif")
    print(N2DL_HeLa_img_list)

    NIH3T3_gt_list = pathlist("../Data/NIH3T3/gt", ".png")
    print(NIH3T3_gt_list)
    NIH3T3_img_list = pathlist("../Data/NIH3T3/im", ".png")
    print(NIH3T3_img_list)
