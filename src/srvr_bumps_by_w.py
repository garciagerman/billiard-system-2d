import billiard_library as gbc
import numpy as np
import os 


# # Bumps with Wall

# In[20]:

#---parameters---#
height_ = 0.50
r_ = 0.50

unit_widths_ = np.round(np.linspace(start=0, stop=0.95, endpoint=True, num=80), 4)
arg_widths_ = (2*unit_widths_*r_)/(1-unit_widths_)
#arg_widths_ = np.concatenate([[0], arg_widths_])


#print(widths_)

print(f"total heights for bumps w. WALL {len(arg_widths_)}...")


# In[ ]:

##---paths---#
BWWall_path_ = "../newdat/bumps_by_w/"
if not os.path.isdir(BWWall_path_):
    print("directory created for bumps by W")
    os.mkdir(BWWall_path_)


# In[ ]:

BumpWall_Keys, BumpWall_Ps = [], []

for i_, w_ in enumerate(arg_widths_):
    print(f"\t current width={unit_widths_[int(i_)]}, argwidth={w_}...")
    segments_ = gbc.bumps_with_wall(radius_= r_, wall_width_= w_, wall_height_= height_)

    scatter_ = lambda x, y : gbc.billiard_cell(
        init_angle_=x,
        position_=y,
        angle_only_=True,
        boundary_=segments_)

    P, theta_partition = gbc.finite_P(scatter_=scatter_, entry_N=2*(10**4), theta_N=500)

    BumpWall_Ps.append(P)
    BumpWall_Keys.append(np.array([height_, r_, unit_widths_[int(i_)]]))


BumpWall_Keys, BumpWall_Ps = np.array(BumpWall_Keys), np.array(BumpWall_Ps)

np.save(file=BWWall_path_ + "keys.npy", arr=BumpWall_Keys, allow_pickle=False)
np.save(file=BWWall_path_ + "p_mats.npy", arr=BumpWall_Ps, allow_pickle=False)
np.save(file=BWWall_path_ + "theta_partition.npy", arr=theta_partition, allow_pickle=False)