UV := uv --no-cache
RUN := $(UV) run
SRC := midit

define PRINT_HELP_PYSCRIPT
import re, sys

BOLD = '\033[1m'
BLUE = '\033[94m'
END = '\033[0m'

print("Usage: make <target>\n")
print(BOLD + "%-20s%s" % ("target", "description") + END)
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print( BLUE + "%-20s" % (target) + END + "%s" % (help))
endef
export PRINT_HELP_PYSCRIPT

help: ## Show this help.
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

infer: ## Submit slurm/infer.sh (template launcher, not used for thesis runs).
	sbatch slurm/infer.sh

install: ## Install the project dependencies with uv
	$(UV) sync

train: ## Submit slurm/train.sh (template launcher, not used for thesis runs).
	sbatch slurm/train.sh

.PHONY: help install infer train
