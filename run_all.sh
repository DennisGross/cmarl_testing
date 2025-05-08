#!/bin/bash
# #srun python3.10 -m venv venv
# sinfo to show all partitions
# sbatch SCRIPNAME.sh # Run script
# squeue -u * # Check status of job
# 

#SBATCH --account=*
#SBATCH --job-name=cmarl_testing
#SBATCH --output=cmarl_testing.log   
#SBATCH --partition=a100q # a100q dgx2q hgx2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

# Output is visible in stdout
echo "Starting job at time:" && date +%Y-%m-%d_%H:%M:%S

# For cuda
export LD_LIBRARY_PATH=/cm/shared/apps/cuda12.1/toolkit/12.1.1/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# Initialize Virtual Environment
source /home/*/D1/cmarl_testing/venv/bin/activate


echo $CUDA_VISIBLE_DEVICES
# Set your variables here
training_steps=100000000
num_cpus=128
num_instances=12000


test_samples=25
test_budget=1000
calibration_number=25

##srun python setup_project.py

# CENTRAL
##srun python central_training.py --env_name="simple_speaker_listener_v4" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_speaker_listener_v4/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_speaker_listener_v4" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_speaker_listener_v4/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=3 --local_ratio=0 --model_path="central_policies/simple_spread_v3_3/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=3 --local_ratio=0 --model_path="central_policies/simple_spread_v3_3/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=4 --local_ratio=0 --model_path="central_policies/simple_spread_v3_4/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=4 --local_ratio=0 --model_path="central_policies/simple_spread_v3_4/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=5 --local_ratio=0 --model_path="central_policies/simple_spread_v3_5/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=5 --local_ratio=0 --model_path="central_policies/simple_spread_v3_5/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=6 --local_ratio=0 --model_path="central_policies/simple_spread_v3_6/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=6 --local_ratio=0 --model_path="central_policies/simple_spread_v3_6/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python central_training.py --env_name="coin_collector" --N_agents=2 --local_ratio=0 --model_path="central_policies/coin_collector/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="coin_collector" --N_agents=2 --local_ratio=0 --model_path="central_policies/coin_collector/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="knight_wizard_zombies" --N_agents=2 --local_ratio=0 --model_path="central_policies/knight_wizard_zombies/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="knight_wizard_zombies" --N_agents=2 --local_ratio=0 --model_path="central_policies/knight_wizard_zombies/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

# local ratio
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.1 --model_path="central_policies/simple_spread_v3_2_01/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.1 --model_path="central_policies/simple_spread_v3_2_01/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.2 --model_path="central_policies/simple_spread_v3_2_02/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.2 --model_path="central_policies/simple_spread_v3_2_02/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.3 --model_path="central_policies/simple_spread_v3_2_03/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.3 --model_path="central_policies/simple_spread_v3_2_03/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.4 --model_path="central_policies/simple_spread_v3_2_04/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.4 --model_path="central_policies/simple_spread_v3_2_04/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.5 --model_path="central_policies/simple_spread_v3_2_05/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.5 --model_path="central_policies/simple_spread_v3_2_05/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.6 --model_path="central_policies/simple_spread_v3_2_06/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.6 --model_path="central_policies/simple_spread_v3_2_06/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.7 --model_path="central_policies/simple_spread_v3_2_07/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.7 --model_path="central_policies/simple_spread_v3_2_07/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.8 --model_path="central_policies/simple_spread_v3_2_08/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.8 --model_path="central_policies/simple_spread_v3_2_08/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.9 --model_path="central_policies/simple_spread_v3_2_09/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.9 --model_path="central_policies/simple_spread_v3_2_09/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=1.0 --model_path="central_policies/simple_spread_v3_2_10/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
##srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=1.0 --model_path="central_policies/simple_spread_v3_2_10/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9


# Different p-percentages
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.0
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.1
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.2
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.3
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.4
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.5
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.6
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.7
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.8

# Different training episodes:
#training_steps=100000000
#training_steps=100
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t100/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t100/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=1000
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t1000/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t1000/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=10000
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t10000/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t10000/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=100000
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t100000/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t100000/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=1000000
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t1000000/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t1000000/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=10000000
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t10000000/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t10000000/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=100000000
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t100000000/model.zip" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python central_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="central_policies/simple_spread_v3_2_t100000000/model.zip" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

# DECENTRAL
#training_steps=100000
#srun python decentral_training.py --env_name="simple_speaker_listener_v4" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_speaker_listener_v4" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_speaker_listener_v4" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_speaker_listener_v4" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=3 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_3" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=3 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_3" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=4 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_4" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=4 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_4" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=5 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_5" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=5 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_5" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=6 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_6" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=6 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_6" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="coin_collector" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/coin_collector" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="coin_collector" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/coin_collector" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="knight_wizard_zombies" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/knight_wizard_zombies" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="knight_wizard_zombies" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/knight_wizard_zombies" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9


# Local ratio
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.1 --model_path="decentral_policies/simple_spread_v3_2_01" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.1 --model_path="decentral_policies/simple_spread_v3_2_01" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.2 --model_path="decentral_policies/simple_spread_v3_2_02" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.2 --model_path="decentral_policies/simple_spread_v3_2_02" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.3 --model_path="decentral_policies/simple_spread_v3_2_03" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.3 --model_path="decentral_policies/simple_spread_v3_2_03" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.4 --model_path="decentral_policies/simple_spread_v3_2_04" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.4 --model_path="decentral_policies/simple_spread_v3_2_04" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.5 --model_path="decentral_policies/simple_spread_v3_2_05" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.5 --model_path="decentral_policies/simple_spread_v3_2_05" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.6 --model_path="decentral_policies/simple_spread_v3_2_06" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.6 --model_path="decentral_policies/simple_spread_v3_2_06" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.7 --model_path="decentral_policies/simple_spread_v3_2_07" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.7 --model_path="decentral_policies/simple_spread_v3_2_07" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.8 --model_path="decentral_policies/simple_spread_v3_2_08" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.8 --model_path="decentral_policies/simple_spread_v3_2_08" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.9 --model_path="decentral_policies/simple_spread_v3_2_09" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0.9 --model_path="decentral_policies/simple_spread_v3_2_09" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=1.0 --model_path="decentral_policies/simple_spread_v3_2_10" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=1.0 --model_path="decentral_policies/simple_spread_v3_2_10" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9

# Different p-percentages
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.0
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.1
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.2
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.3
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.4
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.5
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.6
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.7
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.8


# Different training episodes:
#training_steps=2000
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t2000" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t2000" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=4000
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t4000" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t4000" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=8000
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t8000" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t8000" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=16000
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t16000" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t16000" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=32000
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t32000" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t32000" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=64000
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t64000" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t64000" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9
#training_steps=128000
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t128000" --train=1 --training_steps=$training_steps --num_cpus=$num_cpus --num_instances=$num_instances
#srun python decentral_training.py --env_name="simple_spread_v3" --N_agents=2 --local_ratio=0 --model_path="decentral_policies/simple_spread_v3_2_t128000" --train=0 --test_samples=$test_samples --test_budget=$test_budget --calibration_number=$calibration_number --top_percentage=0.9



# Plot all results
#srun python plot_box_plots.py
#srun python plot_testing_improvements.py
#srun python plot_different_thresholds.py