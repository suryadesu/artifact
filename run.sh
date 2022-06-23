#!/bin/sh

set -x

mkdir results
mkdir figures

cd prism/

CURRDIR=$(pwd)
export PATH="${CURRDIR}/prism/prism/bin:$PATH"

prism model.prism prop.props -const mu=50,tau=0:5:50,pe=0.5,pm=0.5,tgen=5,tbsm=10 -exportresults ../results/model-tau.csv:csv -ptamethod digital
prism model.prism prop.props -const mu=10:10:100,tau=15,pe=0.5,pm=0.5,tgen=5,tbsm=10 -exportresults ../results/model-mu.csv:csv -ptamethod digital
prism model.prism prop.props -const mu=50,tau=15,pe=0:0.1:1,pm=0.5,tgen=5,tbsm=10 -exportresults ../results/model-pe.csv:csv -ptamethod digital
prism model.prism prop.props -const mu=50,tau=15,pe=0.5,pm=0:0.1:1,tgen=5,tbsm=10 -exportresults ../results/model-pm.csv:csv -ptamethod digital
prism model.prism prop.props -const mu=100,tau=50,pe=0.5,pm=0.5,tgen=0:5:50,tbsm=50 -exportresults ../results/model-tgen.csv:csv -ptamethod digital
prism model.prism prop.props -const mu=100,tau=50,pe=0.5,pm=0.5,tgen=10,tbsm=0:5:50 -exportresults ../results/model-tbsm.csv:csv -ptamethod digital

cd ../

cd SeQUeNCe/example
python main.py

cd ../../

python plot.py

