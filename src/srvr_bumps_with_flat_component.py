#---script for bumprs with hole and bumps from paper: dim 500 matrices---#
#---bumps with wall are left out---#

import billiard_library as gbc
import numpy as np
import os 


# # Semicircles

depths_ = np.round(np.linspace(start=0, stop=1, num=30, endpoint=True), 4)

print(depths_)

print(f"total loops for bumps w. well {len(depths_)}...")


# In[17]:

#---paths---#

BWHole_path_ = "../newdat/bumps_with_flat/"
if not os.path.isdir(BWHole_path_):
    print("directory created")
    os.mkdir(BWHole_path_)


# In[ ]:

#---matrices---#

BWH_Keys, BWH_Ps = [], []

for d_ in depths_:
    print(f"\t current depth={d_}..")
    segments_ = gbc.bumps_with_flat(d_)
    
    scatter_ = lambda x, y : gbc.billiard_cell(
        init_angle_=x,
        position_=y,
        angle_only_=True,
        boundary_=segments_)
    
    P, theta_partition = gbc.finite_P(scatter_=scatter_, entry_N=2*(10**4), theta_N=500)
    
    BWH_Ps.append(P)
    BWH_Keys.append(d_)
    
BWH_Keys, BWH_Ps = np.array(BWH_Keys), np.array(BWH_Ps) 


# In[15]:


np.save(file=BWHole_path_ + "keys.npy", arr=BWH_Keys, allow_pickle=False)
np.save(file=BWHole_path_ + "p_mats.npy", arr=BWH_Ps, allow_pickle=False)
np.save(file=BWHole_path_ + "theta_partition.npy", arr=theta_partition, allow_pickle=False)


# In[ ]:

del BWH_Keys, BWH_Ps, theta_partition

print("DONE")
