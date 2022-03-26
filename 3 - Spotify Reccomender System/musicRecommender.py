from getModels import *

###################################################################################
## the code is not in the most modular form you'd expect it to be though, sry :D ##
###################################################################################

def main() -> None:
    getDatabase()
    user_input_whole = getUserInput()
    user_input = user_input_whole[FEATURES]

    mixes = [ _ for _ in range(5)]
    mixes[0] = loadPCA(user_input, PATH)
    mixes[1] = loadFA(user_input, PATH)
    mixes[2] = loadAE(user_input, PATH)
    mixes[3] = loadBCPCA(user_input, PATH)
    mixes[4] = loadVAE(user_input, PATH)

    for i, mix in enumerate(mixes) : 
        mix.to_csv('OutputMixes/mix_' + str(i+1) +'.csv')

    useSimilarities(user_input) #reccommending using all the clusters 
    

    

if __name__ == "__main__":
    main()
