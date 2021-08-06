
import numpy as np
import collections

def PostProcessing(bulk = "" , singleCell="" , result="", resultPath =""):

        f = open(resultPath, "r")
        lines = f.readlines()
        count = 0
        mutations = []
        mutations_clones = []
        
        for line in lines:
            count += 1
            if (count > 1):
                temp = line.split(" ")[2].split("\n")[0].split(",")
                mutations.extend(temp)
                mutations_clones.append(temp)
        
        repeated_mutation = [item for item, count in collections.Counter(mutations).items() if count > 1]
        mutations = list(dict.fromkeys(mutations))
        mutations = sorted(list(mutations))
       
       
        f = open(bulk, "r")
        lines = f.readlines()
        
        VAFs =[]
        count = 0
        for line in lines:
            count += 1
            if (count > 1):
                VAFs = line.split(",")     
       
        
        for mut in repeated_mutation:
            mut_vaf = float(VAFs[int(mut.split("mut")[1])])
            clones = []
            clones_idx = []
            for idx, clone in enumerate(mutations_clones):
               
                if(mut in clone):
                    clones.append(clone)
                    clones_idx.append(idx)
                
               
                clones_mean_vafs = []
                for i in range(len(clones)):
                    meanVAF = 0
                    count = 0
                    
                    for j in range(len(clones[i])):
                        if(clones[i][j] != mut):
                            count+=1
                            meanVAF+= float(VAFs[int(clones[i][j].split("mut")[1])])
                    meanVAF = meanVAF/count
                    clones_mean_vafs.append(meanVAF)
                
            min_diff = 1000 
            selected_clone_index = 0
            for v_idx, clone_mean_vaf in enumerate(clones_mean_vafs):
                if(abs(clone_mean_vaf - mut_vaf) < min_diff):
                    min_diff = abs(clone_mean_vaf - mut_vaf)
                    selected_clone_index = clones_idx[v_idx]
                     
            for ii in clones_idx:
                if(ii != selected_clone_index and (mut in mutations_clones[ii])):
                    mutations_clones[ii].remove(mut)  
            
        for  clone in mutations_clones:
            print(clone)
        
 
if __name__ == '__main__':
       
    PostProcessing(bulk = "./input/post-process/bulk.csv" , singleCell="" , result="" ,resultPath = "./input/post-process/tree.txt" )     
        