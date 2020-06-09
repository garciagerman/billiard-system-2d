import billiard_library as gbc
import numpy as np
import os 

#---parameters---#
heights_ = np.round(np.linspace(start=0, stop=2, endpoint=True, num=101), 4)

r_ = 0.5
w_ = 0.01
cell_len = 1+w_

print(heights_)


##---paths---#
BWWall_path_ = "../newdat/bumps_with_wall_redo_part2_high_K/"
if not os.path.isdir(BWWall_path_):
    print("directory created for bumps with wall part2")
    os.mkdir(BWWall_path_)

BumpWall_Keys, BumpWall_Ps = [], []
for h_ in heights_:
    print(f"\t current width={w_} height={h_}...")
    segments_ = gbc.bumps_with_wall_redo(
        radius_= r_,
        wall_width_= w_,
        wall_height_= h_
    )

    scatter_ = lambda x, y : gbc.billiard_cell(
        init_angle_=x,
        position_=y,
        angle_only_=True,
        boundary_=segments_)

    P, theta_partition = gbc.finite_P(
        scatter_=scatter_,
        entry_N=3*(10**4),
        theta_N=500,
        scale_=cell_len
    )

    BumpWall_Ps.append(P)
    BumpWall_Keys.append(np.array([h_, r_, w_]))
BumpWall_Keys, BumpWall_Ps = np.array(BumpWall_Keys), np.array(BumpWall_Ps)

np.save(file=BWWall_path_ + "keys.npy", arr=BumpWall_Keys, allow_pickle=False)
np.save(file=BWWall_path_ + "p_mats.npy", arr=BumpWall_Ps, allow_pickle=False)
np.save(file=BWWall_path_ + "theta_partition.npy", arr=theta_partition, allow_pickle=False)
