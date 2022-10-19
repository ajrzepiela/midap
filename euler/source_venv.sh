#!/bin/bash

# swap to the right module /env2lmod is an alias for this
# space between the dot is because it is an absolute path
. /cluster/apps/local/env2lmod.sh

# actiavte the right modules
# gcc 8.2 stack
module load gcc/8.2.0
# python 3.9.9 and the proxy
module load python_gpu/3.9.9 eth_proxy

# activate the env, use realpath and dir name to make it sourable from everywhere
source $(realpath $BASH_SOURCE | xargs dirname)/midap/bin/activate
