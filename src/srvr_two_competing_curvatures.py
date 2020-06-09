import billiard_library as gbc
import numpy as np
import os 


# # Bumps with Wall

# In[20]:

#---parameters---#
heights_ = np.round(np.linspace(start=-0.75, stop=0.75, endpoint=True, num=150), 4)

#r_ = 0.75

#unit_widths_ = np.array([0, 0.01, 0.1])
#arg_widths_ = (2*unit_widths_*r_)/(1-unit_widths_)
#arg_widths_ = np.concatenate([[0], arg_widths_])

print(heights_)
#print(f"arg widths: {arg_widths_}")
#print(f"total heights for bumps w. WALL {len(heights_)*len(arg_widths_)}...")


# In[ ]:

##---paths---#
BWWall_path_ = "../newdat/two_competing_curves/"
if not os.path.isdir(BWWall_path_):
    print("directory created for two_competing_curves")
    os.mkdir(BWWall_path_)


# In[ ]:

BumpWall_Keys, BumpWall_Ps = [], []

for h_ in heights_:
    print(f"\t current height={h_}...")
    segments_ = gbc.arc_with_arc_walls(
        radius_= 2,
        width_=0.1,
        height_=h_)

    scatter_ = lambda x, y : gbc.billiard_cell(
        init_angle_=x,
        position_=y,
        angle_only_=True,
        boundary_=segments_)

    P, theta_partition = gbc.finite_P(
        scatter_=scatter_,
        entry_N=3*(10**4),
        theta_N=500)

    BumpWall_Ps.append(P)
    BumpWall_Keys.append(np.array([h_, 2, 0.1]))




# In[ ]:



BumpWall_Keys, BumpWall_Ps = np.array(BumpWall_Keys), np.array(BumpWall_Ps)



np.save(file=BWWall_path_ + "keys.npy", arr=BumpWall_Keys, allow_pickle=False)
np.save(file=BWWall_path_ + "p_mats.npy", arr=BumpWall_Ps, allow_pickle=False)
np.save(file=BWWall_path_ + "theta_partition.npy", arr=theta_partition, allow_pickle=False)


print("DONE")
