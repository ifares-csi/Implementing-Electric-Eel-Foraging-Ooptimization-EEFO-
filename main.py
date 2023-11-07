'''
   Electric Eel Foraging Ooptimization (EEFO) for 23 functions              
   EEFO Python code v1.0. 


 - Python Code : Ibrahim Fares (Ibrahim Ahmed)
 - Email: ifares.cs@gmail.com
 -  The code is based on the following paper:                                
    W. Zhao, L. Wang, Z. Zhang, H. Fan, J. Zhang, S. Mirjalili, N. Khodadadi,
    Q. Cao, Electric eel foraging optimization: A new bio-inspired optimizer 
    for engineering applications,Expert Systems With Applications, 238,      
    (2024),122200, https://doi.org/10.1016/j.eswa.2023.122200. 


'''

import matplotlib.pyplot as plt
from  EFFO import fn_EEFO
# Function to run EEFO
def run_EEFO(FunIndex, MaxIt, PopSize):
    BestX, BestF, HisBestF = fn_EEFO(FunIndex, MaxIt, PopSize)
    print(f'The best fitness of F{FunIndex} is: {BestF}')

    if BestF > 0:
        plt.semilogy(HisBestF, 'r', linewidth=2)
    else:
        plt.plot(HisBestF, 'r', linewidth=2)

    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.title(f'F{FunIndex}')
    plt.show()



def main():
 # Define your values for MaxIt, PopSize, and FunIndex
    MaxIteration = 500
    PopSize = 50
    FunIndex = 1
    # Define your values for MaxIt, PopSize, and FunIndex
    MaxIteration = 500
    PopSize = 50
    FunIndex = 1

    # Run the EEFO algorithm
    run_EEFO(FunIndex, MaxIteration, PopSize)

if __name__ == "__main__":
    main()
