#!/bin/sh

set -x

bash install.sh

CURRDIR=$(pwd)
export PATH="${CURRDIR}/prism/prism/bin:$PATH"
echo $PATH

mkdir results

prism model.prism prop.props -const mu=50,tau=0:5:50,pe=0.5,pm=0.5,tgen=5,tbsm=10 -exportresults results/tau.csv:csv -ptamethod digital

prism model.prism prop.props -const mu=10:10:100,tau=15,pe=0.5,pm=0.5,tgen=5,tbsm=10 -exportresults results/mu.csv:csv -ptamethod digital

prism model.prism prop.props -const mu=50,tau=15,pe=0:0.1:1,pm=0.5,tgen=5,tbsm=10 -exportresults results/pe.csv:csv -ptamethod digital

prism model.prism prop.props -const mu=50,tau=15,pe=0.5,pm=0:0.1:1,tgen=5,tbsm=10 -exportresults results/pm.csv:csv -ptamethod digital

prism model.prism prop.props -const mu=100,tau=50,pe=0.5,pm=0.5,tgen=0:5:50,tbsm=50 -exportresults results/tgen.csv:csv -ptamethod digital

prism model.prism prop.props -const mu=100,tau=50,pe=0.5,pm=0.5,tgen=10,tbsm=0:5:50 -exportresults results/tbsm.csv:csv -ptamethod digital
