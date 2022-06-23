# SeQUeNCe: Simulator of QUantum Network Communication

SeQUeNCe is an open source, discrete-event simulator for quantum networks. As described in our [paper](http://arxiv.org/abs/2009.12000), the simulator includes 5 modules on top of a simulation kernel:
* Hardware
* Entanglement Management
* Resource Management
* Network Management
* Application

These modules can be edited by users to define additional functionality and test protocol schemes, or may be used as-is to test network parameters and topologies.

## Installing
SeQUeNCe requires an installation of Python 3.7. this can be found at the [Python Website](https://www.python.org/downloads/). Then, simply download the package, navigate to its directory, and install with
```
$ pip install .
```
Or, using the included makefile,
```
$ make install
```
This will install the sequence library as well as the numpy, json5, and pandas dependancies.

## Usage Examples
Many examples of SeQUeNCe in action can be found in the example folder. These include both quantum key distribution and entanglement distribution examples.

### Starlight Experiments
Code for the experiments performed in our paper can be found in the file `starlight_experiments.py`. This script uses the `starlight.json` file (also within the example folder) to specify the network topology.
