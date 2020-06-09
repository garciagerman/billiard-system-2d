import billiard_library as gbc
import numpy as np
import os 


# # Bumps with Wall

# In[20]:

#---parameters---#
K_list = np.round(
    np.linspace(start=0.08, stop=2, endpoint=False, num=80), 4)
radius_list = np.round([1/K for K in K_list][::-1], 3)
print(radius_list)

print(f"total heights for bumps {len(radius_list)}...")


# In[ ]:

##---paths---#
BWWall_path_ = "../newdat/bumps_by_r/"
if not os.path.isdir(BWWall_path_):
    print("directory created for bumps by R")
    os.mkdir(BWWall_path_)


# In[ ]:

BumpWall_Keys, BumpWall_Ps = [], []

for r_ in radius_list:
    print(f"\t current radius={r_}...")
    segments_ = gbc.bumps_family(r_)

    scatter_ = lambda x, y : gbc.billiard_cell(
        init_angle_=x,
        position_=y,
        angle_only_=True,
        boundary_=segments_)
    #print("starting on P...")
    P, theta_partition = gbc.finite_P(scatter_=scatter_, entry_N=3*(10**4), theta_N=500)

    BumpWall_Ps.append(P)
    BumpWall_Keys.append(np.array([r_]))




# In[ ]:



BumpWall_Keys, BumpWall_Ps = np.array(BumpWall_Keys), np.array(BumpWall_Ps)



np.save(file=BWWall_path_ + "keys.npy", arr=BumpWall_Keys, allow_pickle=False)

np.save(file=BWWall_path_ + "p_mats.npy", arr=BumpWall_Ps, allow_pickle=False)

np.save(file=BWWall_path_ + "theta_partition.npy", arr=theta_partition, allow_pickle=False)



#

## In[ ]:

#

#del BumpWall_Keys, BumpWall_Ps, theta_partition
