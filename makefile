# quick runs of repository cleaning and other automatizations
# USAGE: make delete_optimization_dirs algorithm=DDPG

algorithm ?= PPO #set on input to make, PPO, A2C, DDPG
db_file ?= A2C_2022-09-13_study.db

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
move_results_from_odin:
	@echo move study.db's, trained agents, graphics to common storage
	scp tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/graphics/PPO_2022_08_15_study_optimization_progress.svg .
	scp tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/results/*.db .

move_db_to_odin:
	@echo move db for "${db_file}" to odin
	scp results/"${db_file}" tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/results
	
delete_all:
	@echo delete all experiments, checkpoints, graphics, optimization files
	rm -rf optimization/*
	rm -rf graphics/*.svg
	rm -rf checkpoints/PPO*
	rm -rf checkpoints/DDPG*
	rm -rf checkpoints/TD3*
	rm -rf checkpoints/A2C*
	rm -rf checkpoints/SAC*
	rm -rf experiments/*
	rm -rf results/*.db
	rm -rf results/*.yaml


init:
	@echo checks and initialize the environment
	poetry check
	poetry install
	poetry shell

setup_dirs:
	@echo setup directory structure
	mkdir checkpoints optimization results graphics experiments

# hydra:
# 	eval "python src/A_main_hydra.py -sc install=bash"
