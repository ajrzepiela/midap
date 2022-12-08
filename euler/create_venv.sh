#!/bin/bash

# swap to the right module /env2lmod is an alias for this
# space between the dot is because it is an absolute path
. /cluster/apps/local/env2lmod.sh

# actiavte the right modules
# gcc 8.2 stack
module load gcc/8.2.0
# python 3.8.5 and the proxy
module load python_gpu/3.8.5 eth_proxy

# create the env
python -m venv midap

# activate the env
source midap/bin/activate

# install the requirements whithout TF
pip --disable-pip-version-check install -U pip
sed '/tensorflow.*/d' ../requirements.txt | sed '/\-e \./d' | sed '/omnipose*/d' | xargs pip install

# now we link TF and its dependencies to the venv from the cluster
# we do this because the cluster installation has been optimized
# but we don't want all (--system-site-package) because we also
# want to overwrite stuff with our requirements
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/tensorboard ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/tensorflow* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/absl ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/google* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/wrapt* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/typing_extensions* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/tf_estimator_nightly* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/opt_einsum* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/gast* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/astunparse* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/termcolor* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/flatbuffers* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/grpc* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/h5py* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/keras* ./midap/lib/python3.8/site-packages/
ln -s /cluster/apps/nss/gcc-8.2.0/python/3.8.5/x86_64/lib64/python3.8/site-packages/libclang* ./midap/lib/python3.8/site-packages/

# install the package
pip install -e ..

while true; do
    read -p "Do you want to add the source script to your .bash_profile? Y/N" yn
    case $yn in
        [Yy]* ) printf '%s\n' '' '# midap env' "source $(pwd)/source_venv.sh" >> ${HOME}/.bash_profile; break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done