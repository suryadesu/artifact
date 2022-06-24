#!/bin/sh

set -x

PATH_TO_PRISM="prism/prism/bin"
mkdir results
mkdir figures

cd prism/

$PATH_TO_PRISM/prism model.prism prop.props -const mu=50,tau=0:5:50,pgen=0.5,pbsm=0.5,tgen=5,tbsm=10 -exportresults ../results/model-tau.csv:csv -ptamethod digital
$PATH_TO_PRISM/prism model.prism prop.props -const mu=10:10:100,tau=15,pgen=0.5,pbsm=0.5,tgen=5,tbsm=10 -exportresults ../results/model-mu.csv:csv -ptamethod digital
$PATH_TO_PRISM/prism model.prism prop.props -const mu=50,tau=15,pgen=0:0.1:1,pbsm=0.5,tgen=5,tbsm=10 -exportresults ../results/model-pgen.csv:csv -ptamethod digital
$PATH_TO_PRISM/prism model.prism prop.props -const mu=50,tau=15,pgen=0.5,pbsm=0:0.1:1,tgen=5,tbsm=10 -exportresults ../results/model-pbsm.csv:csv -ptamethod digital
$PATH_TO_PRISM/prism model.prism prop.props -const mu=100,tau=50,pgen=0.5,pbsm=0.5,tgen=0:5:50,tbsm=50 -exportresults ../results/model-tgen.csv:csv -ptamethod digital
$PATH_TO_PRISM/prism model.prism prop.props -const mu=100,tau=50,pgen=0.5,pbsm=0.5,tgen=10,tbsm=0:5:50 -exportresults ../results/model-tbsm.csv:csv -ptamethod digital

cd ../

cd SeQUeNCe/simulation-3nodes
python3 main.py

cd ../../

python3 plot.py

