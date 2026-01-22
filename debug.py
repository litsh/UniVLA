import pickle 

with open('//inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/UniVLA/data_storage/meta/libero_all_norm.pkl','rb') as f: 
    data=pickle.load(f)
print(data[0]['reasoning'])