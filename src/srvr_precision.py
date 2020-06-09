import billiard_library as gbc
import numpy as np
import os 


# # Bumps with Wall

# In[20]:

#---parameters---#
#radius_list = np.round(np.linspace(start=0.08, stop=1, endpoint=False, num=80), 4)

dims_ = np.linspace(start=10, stop=1000, num=50, dtype=int)
#hi_ = np.linspace(start=500, stop=2000, num=31, dtype=int )

#low_ = np.linspace(start=10, stop=50, num=2, dtype=int)
#hi_ = np.linspace(start=50, stop=100, num=2, dtype=int )

#dims_ = np.concatenate([low_, hi_[1:]])

print(dims_)

print(f"total DIMS for bumps{len(dims_)}...")


# In[ ]:
for mult_ in [0.2, 0.5, 1, 10]:
    ##---paths---#
    BWWall_path_ = f"../newdat/precision_test/entry_sample_{int(mult_*1000)}/"
    if not os.path.isdir(BWWall_path_):
        print(f"directory created at {BWWall_path_}")
        os.mkdir(BWWall_path_)
        
    print(f"Entry Sampling {int(mult_*(10**3))}...")

        
    BumpWall_Keys, BumpWall_Ps = [], []
    for d_ in dims_:
        print(f"\t current dimension={d_}...")
        segments_ = gbc.bumps_family(2)

        scatter_ = lambda x, y : gbc.billiard_cell(
            init_angle_=x,
            position_=y,
            angle_only_=True,
            boundary_=segments_)
        
        #print("starting on P...")
        P, theta_partition = gbc.finite_P(scatter_=scatter_, entry_N=int(mult_*(10**3)), theta_N=d_)

        BumpWall_Ps.append(P)
        BumpWall_Keys.append(d_)
        
    # convert to numpy array
    BumpWall_Keys, BumpWall_Ps = np.array(BumpWall_Keys), np.array(BumpWall_Ps)

    # save out files
    np.save(file=BWWall_path_ + "keys.npy", arr=BumpWall_Keys, allow_pickle=True)
    np.save(file=BWWall_path_ + "p_mats.npy", arr=BumpWall_Ps, allow_pickle=True)
    np.save(file=BWWall_path_ + "theta_partition.npy", arr=theta_partition, allow_pickle=True)
    
    # delete before starting a new run
    del BumpWall_Keys, BumpWall_Ps, theta_partition
