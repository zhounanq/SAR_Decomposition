# -*- coding: utf-8 -*-

"""
Decompose the Sentinel-1 Stokes vector

%% Inputs:
%% - Elements of the wave coherency matrix C2 (either from HH-HV or VH-VV data)
%
%% Outputs:
%% - mv, mp, alpha and delta
%% - RGB, alpha and HSV images

Author: Zhou Ya'nan
Date: 2021-09-16
"""
import os
import numpy as np
import rasterio

os.environ['PROJ_LIB'] = r'D:\develop-envi\anaconda3\envs\py38d\Lib\site-packages\pyproj\proj_dir\share\proj'


def stokes_decomposition_s1(c2metrix_path, pol_mode, stokes_path):
    """

    :param c2metrix_path:
    :param pol_mode:
    :param stokes_path:
    :return:
    """
    assert pol_mode in ['HH-HV', 'VH-VV']

    """ load the C2 elements """
    with rasterio.open(c2metrix_path) as src:
        width = src.width
        height = src.height
        bands = src.count
        crs = src.crs
        transform = src.transform
        profile = src.profile
        src_data = src.read()
    assert bands == 4

    c11 = src_data[0]
    c12_real = src_data[1]
    c12_imag = src_data[2]
    c22 = src_data[3]

    """ build the Stokes vector from C2 """
    s1 = c11 + c22
    s2 = c11 - c22
    s3 = 2 * c12_real
    s4 = 2 * c12_imag

    """ apply the decomposition """
    if pol_mode == 'HH-HV':
        F = 1 / 2.0
    else:
        F = - 1 / 2.0

    """ solve the quadratic (volume term) """
    a = 0.75
    b = -2 * (s1 - (F * s2))
    c = s1 * s1 - s2 * s2 - s3 * s3 - s4 * s4
    delta1 = b * b
    delta2 = 4 * a * c
    delta = delta1 - delta2

    mv1 = -b + np.sqrt(delta)
    mv1 = mv1 / (2 * a)
    mv2 = (-b - np.sqrt(delta))
    mv2 = mv2 / (2 * a)

    """ check which solution satisfies s1 > mv """
    ind = np.nonzero(s1 > 0)  # exclude pixels outside the imaged scene
    flag = np.zeros(2, dtype=bool)
    flag[0] = np.all(s1[ind] <= mv1[ind])
    flag[1] = np.all(s1[ind] <= mv2[ind])

    """ obtain mv, mp, alpha and delta (polarized term) """
    if not flag[0]:
        mv = mv1
    else:
        mv = mv2
    mp = s1 - mv

    if pol_mode == 'HH-HV':
        alpha = 0.5 * np.arccos((s2 - F * mv) / mp)
    else:
        alpha = 0.5 * np.arccos(-(s2 - F * mv) / mp)
    delta = np.angle(s3 + s4 * 1j)

    """ save mv, mp, alpha and delta """
    # profile['dtype'] = 'float32'
    # profile['count'] = 3

    dst_data = np.stack((mv, mp, alpha, delta), axis=0)
    with rasterio.open(stokes_path, 'w', **profile) as dst:
        dst.write(dst_data)

    # cleanup
    return stokes_path


def main():
    print("##########################################################")
    print("### Sentinel-1 Pol Decomposition #########################")
    print("##########################################################")

    # cmd line
    # opts = parse_args()
    # dataset = opts.dataset

    c2metrix_path = r'J:\FF\experiment_dataset\2022-franch_agri\s1_slc\S1A_IW_SLC__1SDV\1_1\S1A_IW_SLC__1SDV_20190503T055125_20190503T055152_027059_030C51_B1FD_Orb_Cal_Deb1_mat_Spk_TC.tif'
    stokes_path = r'J:\FF\experiment_dataset\2022-franch_agri\s1_slc\S1A_IW_SLC__1SDV\1_1\S1A_IW_SLC__1SDV_20190503T055125_20190503T055152_027059_030C51_B1FD_Orb_Cal_Deb1_mat_Spk_TC_STOKES.tif'
    pol_mode = 'VH-VV'

    result = stokes_decomposition_s1(c2metrix_path, pol_mode, stokes_path)

    print("### Complete! #############################################")


if __name__ == "__main__":
    main()
