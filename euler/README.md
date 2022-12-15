# MIDAP on Euler

This directory contains instructions to run the pipeline on Euler. It also contains a scripts that allow you to run 
jupyter notebooks on Euler. However, note that using jupyter on Euler is for **testing only** and is 
interactive. It is bad practice to leave jupyter notebooks running unattended.

## Installation of the pipeline

1. **[Mac only]**: Install [XQuartz](https://www.xquartz.org/) for X11 support (GUI forwarding from Euler) and start the software.

2. Log into Euler with activated X11 forwarding: `ssh -X <username>@euler.ethz.ch`

3. Clone the repo, navigate to the directory containing the pipeline `cd midap` and download model weights and example files from polybox `./download_files.sh`.

4. Navigate to the Euler directory in the repo `cd ./euler` and create the virtual environment
```
./create_venv.sh
```
At the end of script it will ask you if you want to add the source script to your `.bash_profile` if you choose to do 
so, you won't need to do it manually in the next step.

5. Source the environment

```
source source_venv.sh
```
If you did not choose to add the source script to your `.bash_profile` in the previous step 
then this step has to be **repeated everytime you log into Euler before starting the 
pipeline**. If you want this to happen automatically add the following line 
to your `$HOME/.bash_profile`:
```
source <path/to/your>/source_venv.sh
```
where you fill in the absolute path to your source file.

6. Navigate to the bin and start an interactive job with X11 forwarding
```
cd ../bin/
bsub -XF -n 8 -R "rusage[ngpus_excl_p=1]" -Is bash
```

7. After the job starts you can run the pipeline in the same way as on your local machine (see main [README.md](../README.md)).

## Jupyter on Euler

For each step it is indicated if you should do this on your local machine [local] or on Euler [euler]. 

### 1. Passwordless Login [local]

In a first step you will need to set up passwordless login. Details for this procedure can be found on the [scicomp wiki](https://scicomp.ethz.ch/wiki/Accessing_the_clusters#How_to_use_keys_with_non-default_names). First you need to create an SSH key:

```
ssh-keygen -t ed25519 -f $HOME/.ssh/id_ed25519_euler
```

You can leave the passphrase emtpy by pressing Enter twice (even though, strictly speaking this is not recommended).

Then you copy the key to euler via:

```
ssh-copy-id -i ~/.ssh/id_ed25519_euler.pub <username>@euler.ethz.ch
```

where you replace `<username>` with you username on Euler.

Now you can change into your SSH directory and create a config file:

```
cd $HOME/.ssh
echo -e "Host euler\nHostName euler.ethz.ch\nUser <username>\nIdentityFile ~/.ssh/id_ed25519_euler" > config 
```

again replacing `<username>` with your actual username.

At this point the passwordless login should work, you can try it via

```
ssh <username>@euler.ethz.ch
```

### 2. Euler setup [euler]

Log into Euler as shown above. Then clone the repo and follow the installation [instructions above](#installation-of-the-pipeline). Once you have sourced `source_venv.sh`, there should be a `(midap)` on the line of the curser like this

```
(midap) [<username>@eu-....
```  

The next step is to register the Jupyter kernel of the virtual environment via

```
python -m ipykernel install --name midap --user
```

Now you need to configure the new software stack as default stack. Depending on the account creation this might already be the case and the command below will do nothing. To do this run

```
set_software_stack.sh new
```

**Optionally:**

Most likely, everything above happened in your `$HOME` directory. The storage there is permanent but small. If you want to be able to access your scratch directory later from the Jupyter notebook (please be aware that files there are deleted after 2-4 weeks automatically), you can do this by creating a symlink to your scratch. For this run

```
cd $HOME
ln -s $SCRATCH/ scratch
```

### 3. Jupyter Setup [local]

Finally, we can proceed to the jupyter setup. In the `euler` directory of the repo, open the `jnb_euler.config` file, it should look something like this:

```
JNB_USERNAME="<username>"   # ETH username for SSH connection to Euler
JNB_NUM_CPU=8               # Number of CPU cores to be used on the cluster
JNB_NUM_GPU=0               # Number of GPUs to be used on the cluster
JNB_RUN_TIME="04:00"        # Run time limit for the jupyter notebook in hours and minutes HH:MM
JNB_MEM_PER_CPU_CORE=2048   # Memory limit in MB per core
JNB_WAITING_INTERVAL=60     # Time interval to check if the job on the cluster already started
JNB_SSH_KEY_PATH=""         # Path to SSH key with non-standard name
JNB_SOFTWARE_STACK="new"    # Software stack to be used (old, new)
JNB_WORKING_DIR=""          # Working directory for the jupyter notebook
JNB_ENV="<path/to/your/env/midap>"  # Path to virtual environment
```

You **have to** replace `<username>` with your username and `<path/to/your/env/midap>` to the absolute path of the virtual environment that you created in the previous step. Per default, this will be located in the `euler` directory in the repo on Euler. Note that the path has to end in `midap`. Additionally, **you can change** the runtime, number of CPUs and memory. If you have GPU access, you can also increase the number of GPUs. The other parameters should be left to their defaults.

To start the jupyter notebook run:

```
./start_jupyter_nb.sh -c jnb_euler.config
```

This will automatically log into Euler and create a job for you. Then it will check if the job started in a regular interval (60 seconds per default). Once the job starts, it will create an ssh-tunnel to Euler to connect to Jupyter. The output should look something like this:

```
Receiving ip, port and token from jupyter notebook
Remote IP address: 10.205.176.185
Remote port: 8888
Jupyter token: 58d83de3d3f566532d790256a30d375e8b1b934a9afb0592
Determining free port on local computer
Using local port: 64184
Setting up SSH tunnel for connecting the browser to the jupyter notebook
Starting browser and connecting it to jupyter notebook
Connecting to url http://localhost:64184/?token=58d83de3d3f566532d790256a30d375e8b1b934a9afb0592
```

Copy the link after `Connecting to url ` into a browser window and you should see the standard jupyter environment. If you accidentally closed the terminal, you also find the link in the `reconnect_info` file, if the job on Euler is still running, it should reconnect you. 