"""FMPProvider — SKELETON for Financial Modeling Prep (future premium source).

Not used in Step 2. Exists so that if we upgrade to FMP Premium, only this one
class gets filled in — analysis code already speaks the FinancialProvider
contract. The API key is read from the environment / a gitignored .env file
(slot: FMP_API_KEY); no key ships with the repo.
"""

from __future__ import annotations

import os

from .. import config
from .financial_base import CompanyFinancials, FinancialProvider


def _load_api_key() -> str | None:
    """Env var first, then a simple KEY=value scan of repo-root .env."""
    key = os.environ.get(config.FMP_API_KEY_ENV_VAR)
    if key:
        return key
    env_path = os.path.join(config.REPO_ROOT, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{config.FMP_API_KEY_ENV_VAR}="):
                    return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


class FMPProvider(FinancialProvider):
    """Placeholder. Fill in get_company_financials when/if we go Premium."""

    def __init__(self):
        self.api_key = _load_api_key()

    def get_company_financials(self, ticker: str) -> CompanyFinancials:
        raise NotImplementedError(
            "FMPProvider is a skeleton (Step 2). Set FMP_API_KEY in .env and "
            "implement income-statement extraction here if we adopt FMP Premium.")
