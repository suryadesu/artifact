import sequence
from numpy import random
from sequence.kernel.timeline import Timeline
from sequence.topology.topology import Topology
from sequence.topology.node import *
import math, sys
import networkx as nx
import matplotlib.pyplot as plt

from sequence.kernel.eventlist import EventList

def run_simulation(max_execution_time, epr_life, gen_success_probability, swap_succ_prob, sim_gen_time, model_gen_time, model_swap_time):

    #max_execution_time = 1000

    def iteration():
        random.seed(0)
        network_config = "../example/linear-3-node.json"
        # network_config = "../example/linear-4-node.json"
        # network_config = "../example/linear-2-node.json"

        tl = Timeline(4e12)
        network_topo = Topology("network_topo", tl)
        network_topo.load_config(network_config)
        tl.set_topology(network_topo)
        tl.gen_threshold = 100

        tl.src, tl.dst = "a", "c"
        # tl.src, tl.dst = "a", "d"
        #max_execution_time = 50
        #epr_lifetime = 15   #Need to look into this
        epr_lifetime = epr_life
        #tl.gen_success_probability = 0.5
        tl.gen_success_probability = gen_success_probability
        #swap_succ_prob = 0.5
        #sim_swap_gen_time = 0.001
        #model_swap_gen_time = 5
        #sim_gen_time = 0.002
        #model_gen_time = 5
        #model_swap_time = 10
        #unit_time = sim_swap_gen_time/model_swap_gen_time
        unit_time = sim_gen_time/model_gen_time
        #max_execution_time = 30 * unit_time
        epr_lifetime *= unit_time
    

        def set_parameters(topology: Topology):
            # set memory parameters
            MEMO_FREQ = 2e3
            #MEMO_EXPIRE = 10 #inf ---- 1e-3..... starts at 0s, final entanglement at 0.4s (for a line graph )
            MEMO_EXPIRE = epr_lifetime
            MEMO_EFFICIENCY = 1
            MEMO_FIDELITY = 0.9349367588934053
            #MEMO_FIDELITY = 0.99
            for node in topology.get_nodes_by_type("QuantumRouter"):
                node.memory_array.update_memory_params("frequency", MEMO_FREQ)
                node.memory_array.update_memory_params("coherence_time", MEMO_EXPIRE)
                node.memory_array.update_memory_params("efficiency", MEMO_EFFICIENCY)
                node.memory_array.update_memory_params("raw_fidelity", MEMO_FIDELITY)

            # set detector parameters
            #DETECTOR_EFFICIENCY = 0.9
            DETECTOR_EFFICIENCY = 0.99
            DETECTOR_COUNT_RATE = 5e7
            DETECTOR_RESOLUTION = 100
            for node in topology.get_nodes_by_type("BSMNode"):
                node.bsm.update_detectors_params("efficiency", DETECTOR_EFFICIENCY)
                node.bsm.update_detectors_params("count_rate", DETECTOR_COUNT_RATE)
                node.bsm.update_detectors_params("time_resolution", DETECTOR_RESOLUTION)
                
            # set entanglement swapping parameters
            #SWAP_SUCC_PROB = 0.95
            SWAP_SUCC_PROB = swap_succ_prob
            #SWAP_SUCC_PROB = 0.99
            #SWAP_SUCC_PROB = 0.95
            
            #SWAP_DEGRADATION = 0.99
            #SWAP_DEGRADATION = 1
            SWAP_DEGRADATION = 0.99
            
            for node in topology.get_nodes_by_type("QuantumRouter"):
                node.network_manager.protocol_stack[1].set_swapping_success_rate(SWAP_SUCC_PROB)
                node.network_manager.protocol_stack[1].set_swapping_degradation(SWAP_DEGRADATION)
                
            # set quantum channel parameters
            ATTENUATION = 1e-5
            #ATTENUATION = 1e-10
            #ATTENUATION = 1e-8
            #ATTENUATION = attenuation
            QC_FREQ = 1e11
            for qc in topology.qchannels:
                qc.attenuation = ATTENUATION
                qc.frequency = QC_FREQ

        set_parameters(network_topo)

        tl.init()
        tl.events = EventList()

        nm = network_topo.nodes[tl.src].network_manager
        nm.request(tl.dst, start_time=3e12, end_time=20e12, memory_size=1, target_fidelity=0.4)
        #nm.request("b", start_time=3e12, end_time=20e12, memory_size=1, target_fidelity=0.6)
        
        try:
            tl.run()
        except SystemExit:
            print('This Iteration has Failed')

        is_a_entangled_c, is_c_entangled_a = False,False
        #is_a_entangled_d, is_d_entangled_a = False,False
        #is_a_entangled_b, is_b_entangled_a = False,False
        print("A memories")
        print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:\tState:")
        for info in network_topo.nodes["a"].resource_manager.memory_manager:
            if info.remote_node == 'c' and info.state == 'ENTANGLED':
            #if info.remote_node == 'b' and info.state == 'ENTANGLED':
                is_a_entangled_c = True
                #is_a_entangled_b = True
            print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                                str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
                                                
        print("B memories")
        print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
        for info in network_topo.nodes["b"].resource_manager.memory_manager:
            if info.remote_node == 'a' and info.state == 'ENTANGLED':
                is_b_entangled_a = True
            print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                                str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))                                

        #print(f'is_a_entangled_b : {is_a_entangled_b}  and is_b_entangled_a: {is_b_entangled_a}')

        
        print("C memories")
        print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
        for info in network_topo.nodes["c"].resource_manager.memory_manager:
            if info.remote_node == 'a' and info.state == 'ENTANGLED':
                is_c_entangled_a = True
            print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                                str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

        # print("D memories")
        # print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:\tState:")
        # for info in network_topo.nodes["d"].resource_manager.memory_manager:
        #     if info.remote_node == 'a' and info.state == 'ENTANGLED':
        #         is_d_entangled_a = True
        #     print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
        #                                         str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
        

        print('---------Decision Paramter Values---------')
        #print('gen/swap time per operation: ',sim_swap_gen_time)
        print('max time allowed: ', max_execution_time)
        print('epr life time: ', epr_lifetime)
        print('gen_exec_count: ', tl.gen_exec_count)
        print('swap_exec_count: ', tl.swap_exec_count)
        print('gen_threshold: ', tl.gen_threshold)
        print('gen_success_probability: ', tl.gen_success_probability)
        print('model_gen_time: ', model_gen_time)
        print('model_swap_time: ', model_swap_time)
        #print('swap_success_probability: ', tl.gen_success_probability)
        #total_time_taken = sim_swap_gen_time*(sum(map(sum, tl.gen_exec_count))+sum(map(sum,tl.swap_exec_count)))/2
        
        total_time_taken = model_gen_time*sum(map(sum, tl.gen_exec_count))/2 + model_swap_time*sum(map(sum,tl.swap_exec_count))/2

        print('Total time taken:   ', total_time_taken)

        tl.gen_exec_count = [[0 for i in range(4)] for j in range(4)]
        tl.swap_exec_count = [[0 for i in range(4)] for j in range(4)]

        #print()
        #print('gen_exec_count: ', tl.gen_exec_count)
        #print('swap_exec_count: ', tl.swap_exec_count)

        #if is_a_entangled_c and is_c_entangled_a:
        #if is_a_entangled_b and is_b_entangled_a:
        
        if tl.stop_rules:   #This signifies E2E swap has happened
            if total_time_taken<=max_execution_time:
                print('True')
                return True, total_time_taken, tl.hasSwapFailed
            else:
                print('False in max time comparison')
                return False, total_time_taken, tl.hasSwapFailed
            #return True, total_time_taken
        else:
            print('False in both entangled')
            return False, total_time_taken, tl.hasSwapFailed

    success_count, over_time_limit, swap_failed = 0, 0, 0
    total_trials = 2000
    times = {'0'}
    for i in range(total_trials):
        # tl = Timeline(4e12)
        # tl.set_topology(network_topo)
        # tl.gen_threshold = 5
        # tl.gen_success_probability = 0.1
        flag, total_time_taken, hasSwapFailed = iteration()
        times.add(total_time_taken)
        if flag:
            success_count += 1
        if total_time_taken > max_execution_time and not hasSwapFailed:
            over_time_limit += 1
        if hasSwapFailed:
            swap_failed += 1

    print('------------------------------')
    print(f'Total Trials:  {total_trials}')
    print(f'Successful Trials:  {success_count}')
    print(f'Success probability : {success_count/total_trials}')
    print(f'over_time_limit:  {over_time_limit}')
    print(f'swap_failed:  {swap_failed}')
    print(f'Times: {times}')
    print('------------------------------')
    return success_count/total_trials

"""
    Max succ prob vs mu
"""
# mu_range = []
# success_prob_range = []
# for mu in range(0, 101, 10):
#     # print(f"Running for tau = {mu}")
#     mu_range.append(mu)
#     success_prob = run_simulation(max_execution_time = mu, epr_life = 15, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
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
# for t_bsm in range(15, 16, 5):
#     print(f"Running for t_bsm = {t_bsm}")
#     t_bsm_range.append(t_bsm)
#     success_prob = run_simulation(max_execution_time = 100, epr_life = 50, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 10, model_swap_time = t_bsm)
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
# for t_gen in range(40, 41, 5):
#     if t_gen == 0 :
#         t_gen = 0.0000001
    
#     print(f"Running for t_gen = {t_gen}")
#     t_gen_range.append(t_gen)
#     success_prob = run_simulation(max_execution_time = 100, epr_life = 50, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = t_gen, model_swap_time = 50)
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
# for p_bsm in range(6, 7, 1):
#     # print(f"Running for p_bsm = {p_bsm/10}")
#     p_bsm_range.append(p_bsm/10)
#     success_prob = run_simulation(max_execution_time = 50, epr_life = 15, gen_success_probability = 0.5, swap_succ_prob = p_bsm/10, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
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
p_gen_range = []
success_prob_range = []
for p_gen in range(0, 11, 1):
    # print(f"Running for p_bsm = {p_gen/10}")
    p_gen_range.append(p_gen/10)
    success_prob = run_simulation(max_execution_time = 50, epr_life = 15, gen_success_probability = p_gen/10, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
    success_prob_range.append(success_prob)

print(f'p_gen_range: {p_gen_range}')
print(f'success_prob_range: {success_prob_range}')
fig, ax = plt.subplots()
ax.plot(p_gen_range, success_prob_range, color = 'blue')
ax.set_ylim(ymin=0, ymax=1)
plt.xlabel('p_gen')
plt.ylabel('Success Probability')
plt.show()

"""
    Max succ prob vs tau
"""
# tau_range = []
# success_prob_range = []
# for tau in range(0, 51, 5):
#     if tau == 0 :
#         tau = 0.0000001
    
#     tau_range.append(tau)
#     success_prob = run_simulation(max_execution_time = 50, epr_life = tau, gen_success_probability = 0.5, swap_succ_prob = 0.5, sim_gen_time = 0.002, model_gen_time = 5, model_swap_time = 10)
#     success_prob_range.append(success_prob)

#     if tau == 0.0000001:
#         tau = 0

# print(f'tau_range: {tau_range}')
# print(f'success_prob_range: {success_prob_range}')
# fig, ax = plt.subplots()
# ax.plot(tau_range, success_prob_range, color = 'blue')
# ax.set_ylim(ymin=0, ymax=1)
# plt.xlabel('tau')
# plt.ylabel('Success Probability')
# plt.show()