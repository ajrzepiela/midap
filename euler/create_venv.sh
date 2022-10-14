#!/bin/bash

# swap to the right module /env2lmod is an alias for this
# space between the dot is because it is an absolute path
. /cluster/apps/local/env2lmod.sh

# actiavte the right modules
# gcc 8.2 stack
module load gcc/8.2.0
# python 3.9.9 and the proxy
module load python_gpu/3.9.9 eth_proxy

# create the env
python -m venv midap

# activate the env
source midap/bin/activate

# install the requirements whithout TF
pip --disable-pip-version-check install -U pip
sed 's/tensorflow.*//' ../requirements.txt | xargs pip install

# now we link TF and its dependencies to the venv from the cluster
# we do this because the cluster installation has been optimized
# but we don't want all (--system-site-package) because we also
# want to overwrite stuff with our requirements
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/tensorboard ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/tensorflow* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/absl ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/google* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/wrapt* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/typing_extensions* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/tf_estimator_nightly* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/opt_einsum* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/gast* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/astunparse* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/termcolor* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/flatbuffers* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/grpc* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/h5py* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/keras* ./midap/lib/python3.9/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.9.9/x86_64/lib64/python3.9/site-packages/libclang* ./midap/lib/python3.9/site-packages/

# install the package
pip install -e ..
