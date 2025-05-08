# Coordination Failures for Parallel Cooperative MARL Testing
Deep parallel cooperative multi-agent reinforcement learning (CMARL) enables multiple agents to learn policies that optimize a common objective through parallel interactions with the environment.
CMARL, like single-agent RL, faces challenges such as the potential for unsafe behaviors (e.g., collisions) and difficulties in explaining these behaviors.
However, research on testing CMARL policies for safety remains relatively underexplored compared to the safety testing of single-agent RL policies.
In this paper, we propose DisQ, a test case selection method that extends established single-agent RL testing methods to more effectively identify failures in CMARL settings by taking advantage of the unique dynamics of trained CMARL systems.
We achieve this by defining \emph{coordination failure} as an explainable CMARL metric, where agents' future perspectives diverge significantly from each other, warranting potential miscoordination that requires further testing while disregarding test cases where this coordination failure does not occur.
Our experiments in various CMARL benchmark environments demonstrate that DisQ improves failure detection compared to traditional single-agent RL testing methods applied to CMARL system testing, such as random and search-based testing with genetic algorithms.
Our main contributions include an explainable metric for CMARL miscoordination, defined as coordination failure, and a test case selection method that enhances state-of-the-art single-agent testing methods for CMARL~testing.

## Paper Data
The results for the paper are stored in the `logs` directory.
There is a seperate log file for centralized and decentralized policies.
The paper plots can be found in plots.

## Experiments
To run the experiments, install the requirements in `requirements.txt` and execute `run_all.sh` in the root directory.
Comment out the experiments you do not want to run in the script (also remove the srun if you are not using a cluster).