import sys
import matplotlib.pyplot as plt

relations = {'tau' : ['Lifetime of an EPR pair', '6a'], 'mu' : ['Total time', '6b'],'pgen' : ['Probability of EPR pair generation','6c'], 'pbsm' : ['Probability of BSM', '6d'], 'tgen' : ['Processing time of EPR pair generation', '6e'], 'tbsm' : ['Processing time of BSM', '6f']}
#To read both model and simulator's data
def read_xls(filenames):
    RESULTS_PATH = "./results"
    file1_path = RESULTS_PATH + "/" + filenames[0] + ".csv"
    file2_path = RESULTS_PATH + "/" + filenames[1] + ".csv"
    success_model_val = []
    success_prob_range = []
    tau_range = []
    #Input model data
    with open(file1_path,'r') as fin:
        fin.readline()
        while True:
            line = fin.readline()
            if not line:
                break
            a,b = line.split(',')
            tau_range.append(float(a))
            success_model_val.append(float(b))
    
    #Input simulator data
    with open(file2_path,'r') as fin:
        fin.readline()
        while True:
            line = fin.readline()
            if not line:
                break
            a,b = line.split(',')
            success_prob_range.append(float(b))
    return tau_range, success_model_val, success_prob_range

def plot(x_range, x_label, success_model_val, success_prob_range):
    print(f'x_range: {x_range}')
    print(f'success_model_val: {success_model_val}')
    print(f'success_prob_range: {success_prob_range}')
    fig, ax = plt.subplots() 
    ax.plot(x_range, success_model_val, color='blue',marker=(4,0,45),fillstyle='none')
    ax.plot(x_range, success_prob_range, color = 'red',marker=(3,0,0),fillstyle='none')
    ax.set_ylim(ymin=0, ymax=1)
    ax.legend(['Model','Simulator'],loc='upper left')
    plt.xlabel(relations[x_label][0])
    plt.ylabel('Maximum Success Probability')
    plt.savefig('./figures/'+relations[x_label][1]+'.png')

"""
    Max succ prob vs tau
"""
def vs_tau():
    filenames = ['model-tau', 'simulator-tau']
    tau_range, success_model_val, success_prob_range = read_xls(filenames)
    plot(tau_range, 'tau', success_model_val, success_prob_range)

"""
    Max succ prob vs p_gen
"""
def vs_p_gen():
    filenames = ['model-pgen', 'simulator-pgen']
    p_gen_range, success_model_val, success_prob_range = read_xls(filenames)
    plot(p_gen_range, 'pgen', success_model_val, success_prob_range)

"""
    Max succ prob vs p_bsm
"""
def vs_p_bsm():
    filenames = ['model-pbsm', 'simulator-pbsm']
    p_bsm_range, success_model_val, success_prob_range = read_xls(filenames)
    plot(p_bsm_range, 'pbsm', success_model_val, success_prob_range)

"""
    Max succ prob vs t_gen
"""
def vs_t_gen():
    filenames = ['model-tgen', 'simulator-tgen']
    t_gen_range, success_model_val, success_prob_range = read_xls(filenames)
    plot(t_gen_range, 'tgen', success_model_val, success_prob_range)

"""
    Max succ prob vs t_bsm
"""
def vs_t_bsm():
    filenames = ['model-tbsm', 'simulator-tbsm']
    t_bsm_range, success_model_val, success_prob_range = read_xls(filenames)
    plot(t_bsm_range, 'tbsm', success_model_val, success_prob_range)

"""
    Max succ prob vs mu
"""
def vs_mu():
    filenames = ['model-mu', 'simulator-mu']
    mu_range, success_model_val, success_prob_range = read_xls(filenames)
    plot(mu_range, 'mu', success_model_val, success_prob_range)


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        vs_tau()
        vs_p_gen()
        vs_p_bsm()
        vs_t_gen()
        vs_t_bsm()
        vs_mu()
    else:
        choice = sys.argv[1]
        if choice == "tau":
            vs_tau()
        elif choice == "pgen":
            vs_p_gen()
        elif choice == "pbsm":
            vs_p_bsm()
        elif choice == "tgen":
            vs_t_gen()
        elif choice == "tbsm":
            vs_t_bsm()
        elif choice == "mu":
            vs_mu()

