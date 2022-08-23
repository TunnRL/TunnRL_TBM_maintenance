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

# include som scp from odin, just to remember the code
move_results:
	@echo move study.db's, trained agents, graphics to common storage


# all_delete:
# 	delete_checkpoints
# 	delete_optimization_dirs
# 	delete_results
# 	delete_graphics