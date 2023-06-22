# quick runs of repository cleaning and other automatizations
# USAGE: make delete_optimization_dirs algorithm=DDPG

algorithm ?=PPO#set on input to make, PPO, A2C, DDPG
db_file ?= A2C_2022-09-13_study.db
experiment_data ?= /mnt/P/2022/00/20220043/Calculations/

.PHONY: help

help:
	@echo "Makefile Help:"
	@echo "--------------"
	@echo "Usage: make [target] [optional] algorithm=PPO" experiment_data=results
	@echo ""
	@echo "Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

plot_optimization_plots: ## Plots optimization plots for all algorithms
	@echo Plots optimization plots for all algorithms
	python src/B_optimization_analyzer.py --study_name DDPG_2022_10_03_study --study_dirpath "${experiment_data}"
	python src/B_optimization_analyzer.py --study_name TD3_2022_09_27_study --study_dirpath "${experiment_data}"
	python src/B_optimization_analyzer.py --study_name SAC_2022_10_05_study --study_dirpath "${experiment_data}"
	python src/B_optimization_analyzer.py --study_name PPO_2022_09_27_study --study_dirpath "${experiment_data}"
	python src/B_optimization_analyzer.py --study_name A2C_2022_11_30_study --study_dirpath "${experiment_data}"

delete_optimization_dirs: ## Delete all directories in the optimization directory for a certain algorithm
	@echo deleting optimization directories for "${algorithm}"
	rm -rf optimization/"${algorithm}"*

delete_checkpoints: ## Delete all directories in the checkpoint directory for a certain algorithm
	@echo deleting checkpoint directories for "${algorithm}"
	rm -rf checkpoints/"${algorithm}"*

delete_results: ## Delete all directories in the result directory for a certain algorithm
	@echo deleting result files for "${algorithm}"
	rm results/"${algorithm}"*

delete_graphics: ## Delete all directories in the graphics directory
	@echo deleting svg graphics
	rm graphics/*.svg

# Run from root in project locally
move_results_from_odin: ##
	@echo move studys, trained agents, graphics to common storage
	scp -r tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/results/*.db ./results
	scp -r tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/experiments/mlruns ./experiments

# Run from root in project locally
move_optimization_from_odin: ##
	@echo move optimization directories from odin for "${algorithm}"
	scp -r tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/optimization/"${algorithm}"_* ./optimization

# Run from root in project locally
move_db_to_odin: ##
	@echo move db for "${db_file}" to odin
	scp results/"${db_file}" tfh@odin.oslo.ngi.no:/home/tfh/data/projects/TunnRL_TBM_maintenance/results

delete_all: ##
	@echo delete all experiments, checkpoints, graphics, optimization files
	rm -rf optimization/*
	rm -rf graphics/*.svg
	rm -rf checkpoints/PPO*
	rm -rf checkpoints/DDPG*
	rm -rf checkpoints/TD3*
	rm -rf checkpoints/A2C*
	rm -rf checkpoints/SAC*
	rm -rf checkpoints/_sample/*
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
