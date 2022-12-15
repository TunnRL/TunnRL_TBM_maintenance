# quick runs of repository cleaning and other automatizations
# USAGE: make delete_optimization_dirs algorithm=DDPG

algorithm ?= PPO #set on input to make

delete_optimization_dirs:
	@echo deleting optimization directories for "${algorithm}"
	rm -rf optimization/"${algorithm}"*

delete_checkpoints:
	@echo deleting checkpoint directories for "${algorithm}"
	rm -rf checkpoints/"${algorithm}"*

delete_results:
	@echo deleting result files for "${algorithm}"
	rm results/"${algorithm}"*

delete_graphics:
	@echo deleting svg graphics
	rm graphics/*.svg

delete_all_experiment_data:
	@echo delete all experiments, checkpoints, graphics, optimization files
	rm -rf optimization/*
	rm -rf graphics/*.svg
	sudo rm -rf checkpoints/PPO*
	rm -rf checkpoints/DDPG*
	rm -rf checkpoints/TD3*
	rm -rf checkpoints/A2C*
	rm -rf checkpoints/SAC*
	rm -rf checkpoints/_sample/*
	rm -rf experiments/*
	rm -rf results/*.db
	rm -rf results/*.yaml

copy_results_and_graphics_from_az_to_odin:
	@echo copying graphics and results from az-cluster to odin
	scp -r az-cluster.oslo.ngi.no:~/projects/TunnRL_TBM_maintenance/results/"${algorithm}"* ./results
	scp -r az-cluster.oslo.ngi.no:~/projects/TunnRL_TBM_maintenance/optimization/"${algorithm}"* ./optimization

copy_experiments_from_az_to_odin:
	scp -r az-cluster.oslo.ngi.no:~/projects/TunnRL_TBM_maintenance/experiments/mlruns/0/* ./experiments/mlruns/3