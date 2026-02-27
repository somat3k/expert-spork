"""
On-chain metrics client.

Fetches Ethereum on-chain data from public APIs such as
`beaconcha.in`, `ultrasound.money`, and `L2BEAT`.  Returns zero /
empty DataFrames gracefully when network access is unavailable.
"""

from __future__ import annotations

from typing import Dict

import requests


_TIMEOUT = 10  # seconds


class OnChainMetrics:
    """
    Lightweight HTTP client for Ethereum on-chain metrics.

    All methods return ``float`` scalars or ``dict`` objects that can be
    assembled into feature vectors for ML models.
    """

    # ------------------------------------------------------------------
    # Staking / beacon chain
    # ------------------------------------------------------------------

    def get_validator_count(self) -> int:
        """Return the approximate number of active validators."""
        try:
            resp = requests.get(
                "https://beaconcha.in/api/v1/epoch/latest",
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return int(data.get("data", {}).get("validatorscount", 0))
        except Exception:
            return 0

    def get_staking_metrics(self) -> Dict[str, float]:
        """
        Return a dict with staking snapshot metrics.

        Keys: ``validator_count``, ``participation_rate``.
        """
        try:
            resp = requests.get(
                "https://beaconcha.in/api/v1/epoch/latest",
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            d = resp.json().get("data", {})
            return {
                "validator_count": float(d.get("validatorscount", 0)),
                "participation_rate": float(d.get("globalparticipationrate", 0)),
            }
        except Exception:
            return {"validator_count": 0.0, "participation_rate": 0.0}

    # ------------------------------------------------------------------
    # Supply / burn (ultrasound.money)
    # ------------------------------------------------------------------

    def get_eth_supply_metrics(self) -> Dict[str, float]:
        """
        Return circulating supply and burn-rate approximations.

        Note: ultrasound.money does not expose a public JSON API; this
        method is a stub that returns placeholder values until a suitable
        data source is configured.
        """
        return {"circulating_supply": 120_000_000.0, "burn_rate_eth_per_day": 0.0}

    # ------------------------------------------------------------------
    # Layer-2 activity (L2BEAT)
    # ------------------------------------------------------------------

    def get_l2_tvl(self) -> Dict[str, float]:
        """
        Return total value locked (USD) across major Ethereum L2 networks.

        Source: L2BEAT public API.
        """
        try:
            resp = requests.get(
                "https://l2beat.com/api/tvl",
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            # The L2BEAT API returns a list of projects; sum total USD TVL.
            total = sum(float(p.get("tvl", 0)) for p in data.get("projects", []))
            return {"l2_total_tvl_usd": total}
        except Exception:
            return {"l2_total_tvl_usd": 0.0}
