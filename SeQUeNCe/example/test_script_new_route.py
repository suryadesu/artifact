import sequence
from numpy import random
from sequence.kernel.timeline import Timeline
from sequence.topology.topology import Topology
from sequence.topology.node import *
import math, sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from timeout import timeout

@timeout(50)
def conti_code(fidelityIntermediate, fidelityE2E, isvirtual, dest, attenuation, seed, src):

  random.seed(0)
  #network_config = "simple_topology.json"
  network_config = "linear_test_topology.json"

  tl = Timeline(4e12)
  network_topo = Topology("network_topo", tl)
  network_topo.load_config(network_config)

  def set_parameters(topology: Topology, attenuation, seed):
      # set memory parameters
      MEMO_FREQ = 2e3
      MEMO_EXPIRE = 0
      MEMO_EFFICIENCY = 1
      MEMO_FIDELITY = 0.9349367588934053
      #MEMO_FIDELITY = 0.99
      for node in topology.get_nodes_by_type("QuantumRouter"):
          node.random_seed = seed
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
      #SWAP_SUCC_PROB = 0.90
      #SWAP_SUCC_PROB = 0.99
      SWAP_SUCC_PROB = 0.95
      #SWAP_SUCC_PROB = 0.50
      
      #SWAP_DEGRADATION = 0.99
      #SWAP_DEGRADATION = 1
      SWAP_DEGRADATION = 0.99
      
      for node in topology.get_nodes_by_type("QuantumRouter"):
          node.network_manager.protocol_stack[1].set_swapping_success_rate(SWAP_SUCC_PROB)
          node.network_manager.protocol_stack[1].set_swapping_degradation(SWAP_DEGRADATION)
          
      # set quantum channel parameters
      #ATTENUATION = 1e-5
      #ATTENUATION = 1e-10
      #ATTENUATION = 1e-8
      ATTENUATION = attenuation
      QC_FREQ = 1e11
      for qc in topology.qchannels:
          qc.attenuation = ATTENUATION
          qc.frequency = QC_FREQ


  set_parameters(network_topo, attenuation, seed)

  if isvirtual == 'True':
      
      #In case of Linear Topology
      node1 = "a"
      node2 = "c"
      nm = network_topo.nodes[node1].network_manager
      nm.createvirtualrequest(node2, start_time=3e12, end_time=20e12, memory_size=1, target_fidelity=fidelityIntermediate)

      #the start and end nodes may be edited as desired 
      node1 = "e"
      node2 = "g"
      nm = network_topo.nodes[node1].network_manager
      nm.createvirtualrequest(node2, start_time=2e12, end_time=20e12, memory_size=1, target_fidelity=fidelityIntermediate)

      node1 = "i"
      node2 = "l"
      nm = network_topo.nodes[node1].network_manager
      nm.createvirtualrequest(node2, start_time=2e12, end_time=20e12, memory_size=1, target_fidelity=fidelityIntermediate)
      

      """
      #In case of Extended Star Topology
      node1 = "d"
      node2 = "j"
      nm = network_topo.nodes[node1].network_manager
      nm.createvirtualrequest(node2, start_time=3e12, end_time=20e12, memory_size=1, target_fidelity=0.5)
      #nm.request(node2, start_time=3e12, end_time=20e12, memory_size=1, target_fidelity=0.5)
      
       
      node1 = "h"
      node2 = "g"
      nm = network_topo.nodes[node1].network_manager
      nm.createvirtualrequest(node2, start_time=3e12, end_time=20e12, memory_size=1, target_fidelity=0.5)
      #nm.request(node2, start_time=3e12, end_time=20e12, memory_size=1, target_fidelity=0.5)
      """

      tl.init()
      tl.run()

      

      """
      event = tl.events.pop()
      tl.run_counter += 1
      while event.time < 5e12:
        #print('Event start time: ', event.time)
        #print('event.process.owner.name: ', event.process.owner.name)
        try:
          #print('type(event.process.act_params[1]).__name__: ', type(event.process.act_params[1]).__name__)
          message_name = type(event.process.act_params[1]).__name__
        except:
          message_name = None
        
        #if type(event.process.act_params[1]).__name__ ==  'ResourceManagerMessage':
        #    print('remote protocol: ', event.process.act_params[1].ini_protocol.name)
        
        if not (message_name == 'ResourceManagerMessage'):
          #If message is of another type then push it back
          tl.events.push(event)
          tl.run_counter -= 1
        
        event = tl.events.pop()
        tl.run_counter += 1
        #print('length of events: ', len(tl.events))


      #After exiting push the loop breaking event back
      tl.events.push(event)
      tl.run_counter -= 1
      """
      """
      try:
        while True:
          event = tl.events.pop()
          tl.run_counter += 1
      except:
        print('Emptied the event queue')
        print('size of event queue: ', len(tl.events))  
      """

      """
      print('#####################################################')  
      print('After Virtual Link Creation')
      print('#####################################################')  
      print('tl.run_counter: ', tl.run_counter)
      tl.run_counter = 0
      print("M memories")
      print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
      for info in network_topo.nodes["m"].resource_manager.memory_manager:
        if str(info.state) != 'RAW':
          print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                              str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

      
      print("F memories")
      print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
      for info in network_topo.nodes["f"].resource_manager.memory_manager:
        if str(info.state) != 'RAW':
          print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                              str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
      
      
      print("D memories")
      print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
      for info in network_topo.nodes["d"].resource_manager.memory_manager:
        if str(info.state) != 'RAW':
          print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                              str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

      
      print("J memories")
      print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
      for info in network_topo.nodes["j"].resource_manager.memory_manager:
        if str(info.state) != 'RAW':
          print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                              str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
      
      print("H memories")
      print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
      for info in network_topo.nodes["h"].resource_manager.memory_manager:
        if str(info.state) != 'RAW':
          print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                              str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

      print("G memories")
      print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
      for info in network_topo.nodes["g"].resource_manager.memory_manager:
        if str(info.state) != 'RAW':
          print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                              str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
      """
  if isvirtual != 'True':
    tl.init()

  tl.stop_time = 20e12#setting the simulation stop time, but ts not necessary that the simulation will stop at this, if all
                      #the simulation stops at the termination of last valid event, if valid events conitinue to be beyond this
                      #stop time then simulation stops at stop time.
  
  print('Simulator time: ', tl.time)
  req_start_time = 5e12
  nm2 = network_topo.nodes[src].network_manager
  nm2.request(dest, start_time=req_start_time, end_time=20e12, memory_size=1, target_fidelity=fidelityE2E)

  tl.run()

  
  #Find the max entanglement time at A or I to find max timestep taken
  max_time = -10
  max_time_fidelity = 0
  max_time_state = None
  for info in network_topo.nodes[src].resource_manager.memory_manager:
      if info.remote_node == dest and info.entangle_time > max_time:
          max_time = info.entangle_time
          max_time_fidelity = info.fidelity
          max_time_state = info.state

  if max_time < req_start_time:
      max_time = req_start_time

  """
  print("A memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["a"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

  print("F memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["f"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
  """

  """nx_graph = network_topo.generate_nx_graph()
  network_topo.plot_graph(nx_graph)

  #Obtaining the virtual graph
  virt_graph = network_topo.get_virtual_graph()

  network_topo.plot_graph(virt_graph)"""

  """
  print('#####################################################')  
  print('Real Time Demand Usage')
  print('#####################################################')  

  print("M memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["m"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

  
  print("F memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["f"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
  
  
  print("D memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["d"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

  
  print("J memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["j"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
  
  print("H memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["h"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))

  print("G memories")
  print("Index:\tEntangled Node:\tFidelity:\tEntanglement Time:")
  for info in network_topo.nodes["g"].resource_manager.memory_manager:
    if str(info.state) != 'RAW':
      print("{:6}\t{:15}\t{:9}\t{}\t{}".format(str(info.index), str(info.remote_node),
                                          str(info.fidelity), str(info.entangle_time * 1e-12),str(info.state)))
  """
  #print(f"['{(max_time - req_start_time)*1e-12}','{round(max_time_fidelity, 3)}','{round(fidelityE2E, 3)}', '{max_time_state}']")
  return [(max_time - req_start_time)*1e-12, round(max_time_fidelity, 3), round(fidelityE2E, 3), max_time_state]


#%matplotlib inline

import os
import statistics as stats
import sys
import time
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import ast
import json
import networkx as nx

from collections import OrderedDict
from operator import itemgetter

runtimes = []


def timeit_wrapper(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        return_val = func(*args, **kwargs)
        end = time.perf_counter()
        runtimes.append(end - start)
        return return_val
    return wrapper

try:
    
    fidelityIntermediate=[x for x in np.arange(0.7,0.71,0.03)]
    fidelityE2E = [x for x in np.arange(0.3,0.31,0.03)]
    num_trials = len(fidelityE2E)*len(fidelityIntermediate)
    
    #with open('simple_topology.json') as handle:
    with open('linear_test_topology.json') as handle:
        topology_json = json.loads(handle.read())

    #print(topology_json["qconnections"])

    G = nx.Graph()
    for conn in topology_json["qconnections"]:
        #print(f'src: {conn["node1"]} and dest: {conn["node2"]}')
        G.add_edge(conn["node1"], conn["node2"], color='b') 

    colors = nx.get_edge_attributes(G,'color').values()
    #nx.draw(G, edge_color=colors, with_labels = True)
    #plt.show()

    all_pair_shortest_dict = nx.floyd_warshall(G)
    all_pair_shortest_dict = json.loads(json.dumps(all_pair_shortest_dict))
    #print(all_pair_shortest_dict)

    source_nodes = ['a']
    #source_nodes = ['m']
    #dests = ['f', 'j', 'l']
    dests = ['k']
    #source_wise_dests = {}
    #max_distance_from_src = 0
    #number_of_nodes_at_distance = [0 for i in range(20)]

    """
    for src in source_nodes:
        dest_dict = all_pair_shortest_dict[src]
        dest_dict = OrderedDict(sorted(dest_dict.items(), key=itemgetter(1)))
        source_wise_dests[src] = dest_dict

        for (dest, dist) in source_wise_dests[src].items():
            if dist > max_distance_from_src:
                max_distance_from_src = int(dist)
            if src != dest:
                number_of_nodes_at_distance[int(dist)-1] = number_of_nodes_at_distance[int(dist)-1]+1
    """

    #print(source_wise_dests)

    #max_distance_from_src = 0
    #for (dest, dist) in source_wise_dests['a'].items():
    #    print(dest, dist)
    #    if dist > max_distance_from_src:
    #        max_distance_from_src = int(dist)


    #Change this
    #For Time vs Distance
    #destinations = ['c','d','e','f','g','h','i','j','k','l','m']
    attenuation = [1e-3]

    #For Time vs Attenuation
    #destinations = ['i']
    #attenuation = [10**(-x) for x in range(1, 5, 1)]

    
except IndexError:
    num_trials = 5

simul_trials =5
skip_counter = 0
Avg_Physical_Ent_Time = [0 for i in range(3)]
Avg_Virtual_Ent_Time = [0 for i in range(3)]
Avg_fidelity_physical = [0 for i in range(3)]
Avg_fidelity_virtual = [0 for i in range(3)]

list_Avg_Physical_Ent_Time = []
list_Avg_Virtual_Ent_Time = []
list_Avg_fidelity_physical = []
list_Avg_fidelity_virtual = []

index_source_counter = {}
#num_trails_for_index_phy = [0 for i in range(max_distance_from_src)]
#num_trails_for_index_virt = [0 for i in range(max_distance_from_src)]
attenuation = [1e-3]

virtual_counter = 0
physical_counter = 0
equal = 0

print(f'Running simulation for: {simul_trials} trials')
for i in range(simul_trials):
    print('Trial #: ', i)

    if i%10 == 0 and i != 0:
      Avg_Physical_Ent_Time_interval_100 = [0 for i in range(3)]
      Avg_Virtual_Ent_Time_interval_100 = [0 for i in range(3)]
      Avg_fidelity_physical_interval_100 = [0 for i in range(3)]
      Avg_fidelity_virtual_interval_100 = [0 for i in range(3)]

      for j in range(3):
        Avg_Physical_Ent_Time_interval_100[j] = Avg_Physical_Ent_Time[j]/(i-skip_counter)
        Avg_Virtual_Ent_Time_interval_100[j] = Avg_Virtual_Ent_Time[j]/(i-skip_counter)
        Avg_fidelity_physical_interval_100[j] = Avg_fidelity_physical[j]/(i-skip_counter)
        Avg_fidelity_virtual_interval_100[j] = Avg_fidelity_virtual[j]/(i-skip_counter)

      list_Avg_Physical_Ent_Time.append(Avg_Physical_Ent_Time_interval_100)
      list_Avg_Virtual_Ent_Time.append(Avg_Virtual_Ent_Time_interval_100)
      list_Avg_fidelity_physical.append(Avg_fidelity_physical_interval_100)
      list_Avg_fidelity_virtual.append(Avg_fidelity_virtual_interval_100)

    #iter_seed = np.random.randint(2, 100, dtype='uint32')
    Physical_Ent_Time = []
    Virtual_Ent_Time = []
    distance_from_src = []
    fidelity_physical = []
    fidelity_virtual = []

    for src in source_nodes:
        print('Src: ', src)
        index = 0
        for dest in dests:            
        #for (dest, dist) in source_wise_dests[src].items():           
            if src == dest:
              continue

            for f_i in fidelityIntermediate:
                for f_e2e in fidelityE2E:            
                    for atten in attenuation:                        
                        index_of_dest = dests.index(dest)
                        seeds = [0 for i in range(simul_trials)]
                        #print(f"Running for Destination: {dest}  intermediate fidelity value {round(f_i, 3)} and E2E fidelity value {round(f_e2e, 3)} and attenuation {atten} \n", end='', flush=True)
                        print(f"Running for Destination: {dest} at the distance of {index_of_dest} hops")
                        
                        #print('Running for Physical')                        
                        try:
                          retvalPhy = conti_code(f_i, f_e2e, 'False' ,dest, atten, seeds[i], src)
                        except:
                          print('Skipping this step')
                          skip_counter += 1
                          continue
                        print('From retval ---- ', retvalPhy)
                        #print(Physical_Ent_Time)
                        #if retvalPhy[3] != None:
                        #  num_trails_for_index_phy[int(dist)-1] = num_trails_for_index_phy[int(dist)-1]+1
                        
                        #print('Running for Virtual')
                        try:
                          retvalVirt = conti_code(f_i, f_e2e, 'True' ,dest, atten, seeds[i], src)
                        except:
                          print('Skipping this step')
                          skip_counter += 1
                          continue
                        #Virtual_Ent_Time.append(float(ast.literal_eval(retvalVirt)[0]))
                        
                        #Physical_Ent_Time.append(float(ast.literal_eval(retvalPhy)[0]))
                        Physical_Ent_Time.append(retvalPhy[0])
                        Avg_Physical_Ent_Time[index_of_dest] = Avg_Physical_Ent_Time[index_of_dest] + Physical_Ent_Time[-1]
                        #fidelity_physical.append(float(ast.literal_eval(retvalPhy)[1]))
                        fidelity_physical.append(retvalPhy[1])
                        Avg_fidelity_physical[index_of_dest] = Avg_fidelity_physical[index_of_dest] + fidelity_physical[-1]

                        Virtual_Ent_Time.append(retvalVirt[0])
                        Avg_Virtual_Ent_Time[index_of_dest] = Avg_Virtual_Ent_Time[index_of_dest] + Virtual_Ent_Time[-1]
                        #fidelity_virtual.append(float(ast.literal_eval(retvalVirt)[1]))
                        fidelity_virtual.append(retvalVirt[1])
                        Avg_fidelity_virtual[index_of_dest] = Avg_fidelity_virtual[index_of_dest] + fidelity_virtual[-1]
                        #print(Virtual_Ent_Time)
                        #if retvalVirt[3] != None:
                        #  num_trails_for_index_virt[int(dist)-1] = num_trails_for_index_virt[int(dist)-1]+1

                        print('From retval ---- ', retvalVirt)
                        
                        if retvalVirt[0] > retvalPhy[0]:
                          virtual_counter = virtual_counter+1
                        elif retvalVirt[0] < retvalPhy[0]:
                          physical_counter = physical_counter+1
                        else:
                          equal = equal+1
                        
                        #seed = seed + 1
                        #if int(dist)-1 not in index_source_counter:
                        #    index_source_counter[int(dist)-1] = {src}
                        #else:
                        #    source_set = index_source_counter[int(dist)-1]
                        #    source_set.add(src)
                        #    index_source_counter[int(dist)-1] = source_set   

            index = index + 1
            #print("ran in {}s".format(runtimes[-1]))

    #print("mean time: {}".format(stats.mean(runtimes)))
    #print("min time:  {}".format(min(runtimes)))
    #print("max time:  {}".format(max(runtimes)))
    #print("standard deviation: {}".format(stats.stdev(runtimes)))
"""
for i in range(3):
  Avg_Physical_Ent_Time[i] = Avg_Physical_Ent_Time[i]/(simul_trials*len(index_source_counter[i])*number_of_nodes_at_distance[i])
  Avg_Virtual_Ent_Time[i] = Avg_Virtual_Ent_Time[i]/(simul_trials*len(index_source_counter[i])*number_of_nodes_at_distance[i])
  Avg_fidelity_physical[i] = Avg_fidelity_physical[i]/(simul_trials*len(index_source_counter[i])*number_of_nodes_at_distance[i])
  Avg_fidelity_virtual[i] = Avg_fidelity_virtual[i]/(simul_trials*len(index_source_counter[i])*number_of_nodes_at_distance[i])


  #This code should be commented
  Avg_Physical_Ent_Time[i] = Avg_Physical_Ent_Time[i]/(num_trails_for_index_phy[i]*len(index_source_counter[i])*number_of_nodes_at_distance[i])
  Avg_Virtual_Ent_Time[i] = Avg_Virtual_Ent_Time[i]/(num_trails_for_index_virt[i]*len(index_source_counter[i])*number_of_nodes_at_distance[i])
  Avg_fidelity_physical[i] = Avg_fidelity_physical[i]/(num_trails_for_index_phy[i]*len(index_source_counter[i])*number_of_nodes_at_distance[i])
  Avg_fidelity_virtual[i] = Avg_fidelity_virtual[i]/(num_trails_for_index_virt[i]*len(index_source_counter[i])*number_of_nodes_at_distance[i])
"""
for i in range(3):
  Avg_Physical_Ent_Time[i] = Avg_Physical_Ent_Time[i]/simul_trials
  Avg_Virtual_Ent_Time[i] = Avg_Virtual_Ent_Time[i]/simul_trials
  Avg_fidelity_physical[i] = Avg_fidelity_physical[i]/simul_trials
  Avg_fidelity_virtual[i] = Avg_fidelity_virtual[i]/simul_trials


for lst in list_Avg_Physical_Ent_Time:
  print('Avg_Physical_Ent_Time = ', lst)
print('Avg_Physical_Ent_Time = ',Avg_Physical_Ent_Time)
print()

for lst in list_Avg_Virtual_Ent_Time:
  print('Avg_Virtual_Ent_Time = ',lst)
print('Avg_Virtual_Ent_Time = ',Avg_Virtual_Ent_Time)
print()

print('Skipped Iterations: ', skip_counter)

for lst in list_Avg_fidelity_physical:
  print('Avg_fidelity_physical = ', lst)
print('Avg_fidelity_physical = ',Avg_fidelity_physical)
print()

for lst in list_Avg_fidelity_virtual:
  print('Avg_fidelity_virtual = ',lst)
print('Avg_fidelity_virtual = ',Avg_fidelity_virtual)
print()
 
list_Avg_fidelity_physical = []
list_Avg_fidelity_virtual = []

#print(Avg_Physical_Ent_Time)

#print(Avg_Virtual_Ent_Time)

#print(Avg_fidelity_physical)

#print(Avg_fidelity_virtual)

"""
max_distance_from_src = 3
print('X axis')
print([i for i in range(1, max_distance_from_src+1)])
print()
#print('index_source_counter', index_source_counter)
print()
"""
#print('number_of_nodes_at_distance[int(dist)-1]', number_of_nodes_at_distance[int(dist)-1])

"""
#Change this
#For change in distance from source
fig, ax = plt.subplots()
ax.plot(['f' , 'j', 'l'], Avg_Physical_Ent_Time, color = 'blue' ,label = r'Time for physical')
ax.plot(['f' , 'j', 'l'], Avg_Virtual_Ent_Time, color = 'red', label = r'Time for virtual')

ax.legend(loc = 'upper left')
plt.xlabel('node')
plt.ylabel('Entanglement Time')
plt.show()
"""

"""
print(attenuation)
#For change in attenuation
ax.plot(attenuation, Physical_Ent_Time, color = 'blue' ,label = r'Time for physical')
ax.plot(attenuation, Virtual_Ent_Time, color = 'red', label = r'Time for virtual')

ax.set_xscale('log')
ax.legend(loc = 'upper left')
plt.xlabel('Attenuation')
plt.ylabel('Entanglement Time')
plt.show()
"""

"""
fig, ax = plt.subplots()
ax.plot(['f' , 'j', 'l'], Avg_fidelity_physical, alpha= 0.5,  color = 'blue' ,label = r'Fidelity for physical')
ax.plot(['f' , 'j', 'l'], Avg_fidelity_virtual, '--' , alpha= 0.5, color = 'red', label = r'Fidelity for virtual')

ax.legend(loc = 'upper right')
plt.xlabel('node')
plt.ylabel('Entanglement Fidelity')
plt.show()
"""

fig, ax = plt.subplots()
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.bar(['virtual greater','physical greater','equal'], [virtual_counter, physical_counter, equal])
plt.show()
