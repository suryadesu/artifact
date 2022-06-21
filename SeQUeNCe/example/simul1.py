import sys
import subprocess
import matplotlib.pyplot as plt

"""
    Max succ prob vs mu
"""
# mu_range = []
# success_prob_range = []
# for mu in range(10, 101, 10):
#     # print(f"Running for tau = {mu}")
#     mu_range.append(mu)
#     #success_prob = run_simulation(max_execution_time = mu, epr_life = 15, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
#     success_prob = float(subprocess.check_output([sys.executable, "simul2.py", str(mu), "15", "0.5", "0.5", "0.002", "5", "10"]).decode())
#     print(f'Success probability : {success_prob}')
#     success_prob_range.append(success_prob)

# print(f'mu_range: {mu_range}')
# print(f'success_prob_range: {success_prob_range}')
# fig, ax = plt.subplots()
# ax.plot(mu_range, success_prob_range, color = 'blue')
# ax.set_ylim(ymin=0, ymax=1)
# plt.xlabel('mu')
# plt.ylabel('Success Probability')
# plt.show()

"""
    Max succ prob vs t_bsm
"""
# t_bsm_range = []
# success_prob_range = []
# for t_bsm in range(0, 51, 5):
#     print(f"Running for t_bsm = {t_bsm}")
#     t_bsm_range.append(t_bsm)
#     # success_prob = run_simulation(max_execution_time = 100, epr_life = 50, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 10, model_swap_time = t_bsm)
#     success_prob = float(subprocess.check_output([sys.executable, "simul2.py", "100", "50" ,"0.5", "0.5", "0.002", "10", str(t_bsm)]).decode())
#     print(f'Success probability : {success_prob}')
#     success_prob_range.append(success_prob)

# print(f't_bsm_range: {t_bsm_range}')
# print(f'success_prob_range: {success_prob_range}')
# fig, ax = plt.subplots()
# ax.plot(t_bsm_range, success_prob_range, color = 'blue')
# ax.set_ylim(ymin=0, ymax=1)
# plt.xlabel('t_bsm')
# plt.ylabel('Success Probability')
# plt.show()

"""
    Max succ prob vs t_gen
"""
# t_gen_range = []
# success_prob_range = []
# for t_gen in range(0, 51, 5):
#     if t_gen == 0 :
#         t_gen = 0.0000001
    
#     print(f"Running for t_gen = {t_gen}")
#     t_gen_range.append(t_gen)
#     # success_prob = run_simulation(max_execution_time = 100, epr_life = 50, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = t_gen, model_swap_time = 50)
#     success_prob = float(subprocess.check_output([sys.executable, "simul2.py", "100", "50" ,"0.5", "0.5", "0.002", str(t_gen), "50"]).decode())
#     print(f'Success probability : {success_prob}')
#     success_prob_range.append(success_prob)
    
#     if t_gen == 0.0000001:
#         t_gen = 0

# print(f't_gen_range: {t_gen_range}')
# print(f'success_prob_range: {success_prob_range}')
# fig, ax = plt.subplots()
# ax.plot(t_gen_range, success_prob_range, color = 'blue')
# ax.set_ylim(ymin=0, ymax=1)
# plt.xlabel('t_gen')
# plt.ylabel('Success Probability')
# plt.show()

"""
    Max succ prob vs p_bsm
"""
# p_bsm_range = []
# success_prob_range = []
# for p_bsm in range(0, 11, 1):
#     # print(f"Running for p_bsm = {p_bsm/10}")
#     p_bsm_range.append(p_bsm/10)
#     # success_prob = run_simulation(max_execution_time = 50, epr_life = 15, gen_success_probability = 0.5, swap_succ_prob = p_bsm/10, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
#     success_prob = float(subprocess.check_output([sys.executable, "simul2.py", "50", "15" ,"0.5", str(p_bsm/10), "0.002", "5", "10"]).decode())
#     print(f'Success probability : {success_prob}')
#     success_prob_range.append(success_prob)

# print(f'p_bsm_range: {p_bsm_range}')
# print(f'success_prob_range: {success_prob_range}')
# fig, ax = plt.subplots()
# ax.plot(p_bsm_range, success_prob_range, color = 'blue')
# ax.set_ylim(ymin=0, ymax=1)
# plt.xlabel('p_bsm')
# plt.ylabel('Success Probability')
# plt.show()

"""
    Max succ prob vs p_gen
"""
# p_gen_range = []
# success_prob_range = []
# for p_gen in range(0, 11, 1):
#     # print(f"Running for p_bsm = {p_gen/10}")
#     p_gen_range.append(p_gen/10)
#     #success_prob = run_simulation(max_execution_time = 50, epr_life = 15, gen_success_probability = p_gen/10, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
#     success_prob = float(subprocess.check_output([sys.executable, "simul2.py", "50", "15" ,str(p_gen/10), "0.5", "0.002", "5", "10"]).decode())
#     print(f'Success probability : {success_prob}')
#     success_prob_range.append(success_prob)

# print(f'p_gen_range: {p_gen_range}')
# print(f'success_prob_range: {success_prob_range}')
# fig, ax = plt.subplots()
# ax.plot(p_gen_range, success_prob_range, color = 'blue')
# ax.set_ylim(ymin=0, ymax=1)
# plt.xlabel('p_gen')
# plt.ylabel('Success Probability')
# plt.show()

"""
    Max succ prob vs tau
"""
def vs_tau():
    tau_range = []
    success_prob_range = []
    for tau in range(0, 51, 5):
        if tau == 0 :
            tau = 0.0000001
        
        tau_range.append(tau)
        # success_prob = run_simulation(max_execution_time = 50, epr_life = tau, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
        success_prob = float(subprocess.check_output([sys.executable, "simul2.py", "50", str(tau) ,"0.5", "0.5", "0.002", "5", "10"]).decode())
        print(f'Success probability : {success_prob}')
        success_prob_range.append(success_prob)

        if tau == 0.0000001:
            tau = 0

    print(f'tau_range: {tau_range}')
    print(f'success_prob_range: {success_prob_range}')
    fig, ax = plt.subplots()
    ax.plot(tau_range, success_prob_range, color = 'blue')
    ax.set_ylim(ymin=0, ymax=1)
    plt.xlabel('tau')
    plt.ylabel('Success Probability')
    plt.show()

def read_xls():
    pass

if __name__ == "__main__":
    choice = sys.argv[1]
    print("Inside main, choice: ", choice)
    if choice == "tau":
        print("in choice")
        vs_tau()