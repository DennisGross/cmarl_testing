import os
class Logger:

    def __init__(self, log_file, env_name, N_agents, local_ratio, model_path, sample_size, test_budget,top_percentage):
        self.log_file = log_file
        self.env_name = env_name
        self.model_path = model_path
        self.sample_size = sample_size
        self.test_budget = test_budget
        self.N_agents = N_agents
        self.local_ratio = local_ratio
        self.top_percentage = top_percentage

    def log(self, test_method, avg_reward, avg_time_taken):
        if os.path.exists(self.log_file) == False:
            with open(self.log_file, 'w') as f:
                    f.write("env_name,N_agents,local_ratio,model_path,test_method,sample_size,test_budget,avg_reward,avg_time_taken,top_percentage\n")

        with open(self.log_file, 'a') as f:
            f.write(f"{self.env_name},{self.N_agents},{self.local_ratio},{self.model_path},{test_method},{self.sample_size},{self.test_budget},{avg_reward},{avg_time_taken},{self.top_percentage}\n")

    
    def store_collected_rewards(self, test_method, rewards, path="all_rewards.csv"):
        # Store the rewards line by line in a file
        with open(path, 'a') as f:
            for reward in rewards:
                f.write(f"{self.model_path},{test_method},{reward}\n")