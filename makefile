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

copy_from_az_to_odin:
	@echo copying graphics and results from az-cluster to odin
	scp -r az-cluster.oslo.ngi.no:~/projects/TunnRL_TBM_maintenance/results/"${algorithm}"* ./results
	scp -r az-cluster.oslo.ngi.no:~/projects/TunnRL_TBM_maintenance/optimization/"${algorithm}"* ./optimization
	
# scp -r az-cluster.oslo.ngi.no:~/projects/TunnRL_TBM_maintenance/graphics/"${algorithm}"* ./graphics