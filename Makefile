.PHONY: check test lint validate-single validate-merged data-smoke

check: lint test validate-single validate-merged data-smoke

test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest

lint:
	.venv/bin/ruff check .

validate-single:
	.venv/bin/vggt-bev-validate --target-mode single --extent-key bev_5m

validate-merged:
	.venv/bin/vggt-bev-validate --target-mode merged --extent-key bev_5m

data-smoke:
	.venv/bin/vggt-bev-train --config configs/method2_observed.toml --data-only

