import pathlib as pl

N2DL_HeLa_im = [str(pl.Path('../Data/N2DL-HeLa/img/*.tif'))]
N2DL_HeLa_gt = [str(pl.Path('../Data/N2DL-HeLa/gt/*.tif'))]

NIH3T3_im = [str(pl.Path('../Data/NIH3T3/img/*.png'))]
NIH3T3_gt = [str(pl.Path('../Data/NIH3T3/gt/*.png'))]

N2DH_GOWT1_im = [str(pl.Path('../Data/N2DH_GOWT1/img/*.tif'))]
N2DH_GOWT1_gt = [str(pl.Path('../Data/N2DH_GOWT1/gt/*.tif'))]