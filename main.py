import numpy as np
import torch
import gymnasium as gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG

from td3_plot import plot_results

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=20, render_mode=None, max_eval=0., discount=0.99):
	eval_env = gym.make(env_name, render_mode=render_mode)
	eval_env.reset(seed=seed + 100)

	avg_reward = 0.
	disc_reward = 0.
	
	for _ in range(eval_episodes):
		state, done, truncated = eval_env.reset(), False, False
		state = np.array(state[0], dtype=np.float32)
		ep_steps = 0
		while not done and not truncated:
			action = policy.select_action(np.array(state))
			state, reward, done, truncated, _ = eval_env.step(action)
			avg_reward += reward
			disc_reward += reward * pow(discount, ep_steps)
			ep_steps += 1
	
	avg_reward /= eval_episodes
	disc_reward /= eval_episodes
	max_eval = max(max_eval, disc_reward)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} ({disc_reward:.3f}) -- Max Disc: {max_eval:.3f}")
	print("---------------------------------------")
	return [disc_reward, avg_reward]


if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="LunarLanderContinuous-v3")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99, type=float)     # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
	parser.add_argument("--demo", action="store_true")              # Run a visual demo of the model
	parser.add_argument("--plot_results", action="store_true")      # Generate a simple plot of the latest raw training results
	parser.add_argument("--plot_ave", action="store_true")          # Generate a simple plot of the latest average training results
	parser.add_argument("--plot_all", action="store_true")          # Generate a simple plot of the latest raw and average training results
	parser.add_argument("--max_buffer_size", default=1e5, type=int) # Set the maximum size of the replay buffer
	parser.add_argument("--no_replacement", action="store_true")    # Prevent batch replacement in the replay buffer samples
	parser.add_argument("--dev", action="store_true")               # Development mode
	args = parser.parse_args()

	if args.dev:
		args.policy = "TD3-DEV"

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.plot_all:
		plot_results(f"{args.env}_{args.seed}", policy_name=args.policy, eval_freq=args.eval_freq)
		exit(0)

	if args.plot_results:
		plot_results(f"{args.env}_{args.seed}", policy_name=args.policy, eval_freq=args.eval_freq, show_ave=False)
		exit(0)

	if args.plot_ave:
		plot_results(f"{args.env}_{args.seed}", policy_name=args.policy, eval_freq=args.eval_freq, show_train=False)
		exit(0)

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.reset(seed=args.seed)
	env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])


	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy.startswith("TD3"):
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		kwargs["dev_mode"] = args.dev
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim, max_size=int(args.max_buffer_size), gamma=args.discount, prevent_replacement=args.no_replacement)
	
	# Run the demo if required
	if args.demo:
		eval_policy(policy, args.env, args.seed, eval_episodes=1, render_mode="human", discount=args.discount)
		exit(0)

	# Evaluate untrained policy
	eval = eval_policy(policy, args.env, args.seed, discount=args.discount)
	evaluations = [eval[0]]
	max_eval = evaluations[-1]

	state, done, truncated = env.reset(), False, False
	state = np.array(state[0], dtype=np.float32)
	episode_reward = 0
	episode_disc = 0
	episode_timesteps = 0
	episode_num = 0

	for t in range(int(args.max_timesteps)):

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, truncated, _ = env.step(action)

		# Store data in replay buffer
		replay_buffer.push(state, action, next_state, reward, done, truncated)

		state = next_state
		episode_reward += reward
		episode_disc += reward * pow(args.discount, episode_timesteps)
		episode_timesteps += 1

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size, max_eval)

		if done or truncated:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} ({episode_disc:.3f}) RPS: {episode_reward/episode_timesteps:.3f}")
			# Reset environment
			state, done, truncated = env.reset(), False, False
			state = np.array(state[0], dtype=np.float32)
			episode_reward = 0
			episode_disc = 0
			episode_timesteps = 0
			episode_num += 1 

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			eval = eval_policy(policy, args.env, args.seed, max_eval=max_eval, discount=args.discount)
			evaluations.append(eval[0])
			max_eval = max(evaluations[-1], max_eval)
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")
