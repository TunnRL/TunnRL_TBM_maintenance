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

# Run from dir where the files should be moved
move_results:
	@echo move study.db's, trained agents, graphics to common storage
	scp tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/graphics/PPO_2022_08_15_study_optimization_progress.svg .
	scp tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/results/*.db .
	
delete_all:
	@echo delete all experiment and result files
	rm -rf optimization/*
	rm -rf results/*.db
	rm -rf graphics/*.svg
	rm -rf checkpoints/*

# init:
# 	@echo initialize the environment and tab completion in hydra
# 	poetry shell

# hydra:
# 	eval "python src/A_main_hydra.py -sc install=bash"