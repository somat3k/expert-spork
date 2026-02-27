"""
Reinforcement learning trading agent.

Implements a Deep Q-Network (DQN) that learns to produce buy / sell / hold
signals from **multi-timeframe OHLCV data**, with special emphasis on the
high-low price *range* as a volatility proxy.

Hyperparameters are defined as constants in
:class:`~eth_algo_trading.config.RLConfig` and can be updated dynamically at
runtime via a :class:`~eth_algo_trading.db.hyperparams.HyperparamDB` connection.

Usage
-----
>>> from eth_algo_trading.models.rl_agent import RLTradingAgent
>>> agent = RLTradingAgent()
>>> signal = agent.predict({"1h": ohlcv_1h, "4h": ohlcv_4h})
"""

from __future__ import annotations

import io
import random
from collections import deque
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eth_algo_trading.config import RLConfig
from eth_algo_trading.db.hyperparams import HyperparamDB
from eth_algo_trading.strategies.base import Signal

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

_ACTION_FLAT = 0
_ACTION_LONG = 1
_ACTION_SHORT = 2
_ACTION_TO_DIRECTION = {_ACTION_FLAT: "flat", _ACTION_LONG: "long", _ACTION_SHORT: "short"}
N_ACTIONS = 3

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def _build_state(
    ohlcv_by_tf: Dict[str, pd.DataFrame],
    lookback: int,
) -> np.ndarray:
    """
    Build a flat feature vector from multi-timeframe OHLCV data.

    For each timeframe three normalised feature sequences are extracted:

    * **hl_range** – ``(high - low) / close``  (volatility proxy)
    * **returns**  – ``close.pct_change()``
    * **log_vol**  – ``log1p(volume)`` normalised by its own maximum

    The sequences are truncated / zero-padded to *lookback* bars and
    concatenated into a single ``float32`` array.

    Parameters
    ----------
    ohlcv_by_tf:
        Mapping of timeframe label → OHLCV DataFrame.
    lookback:
        Number of bars to retain per timeframe.

    Returns
    -------
    np.ndarray, shape ``(len(ohlcv_by_tf) * 3 * lookback,)``
    """
    parts: List[np.ndarray] = []

    for ohlcv in ohlcv_by_tf.values():
        n = len(ohlcv)

        if n == 0:
            # Empty DataFrame — fill with zeros
            parts.append(np.zeros(3 * lookback, dtype=np.float32))
            continue

        # Align to lookback window
        if n >= lookback:
            sub = ohlcv.iloc[-lookback:]
        else:
            # Zero-pad at the front
            pad_rows = lookback - n
            pad = pd.DataFrame(
                {c: np.zeros(pad_rows) for c in ohlcv.columns},
                index=pd.RangeIndex(pad_rows),
            )
            sub = pd.concat([pad, ohlcv], ignore_index=True)

        close = sub["close"].values.astype(float)
        high = sub["high"].values.astype(float)
        low = sub["low"].values.astype(float)
        volume = sub["volume"].values.astype(float)

        denom = np.where(close != 0, close, 1.0)
        hl_range = (high - low) / denom

        prev_close = np.concatenate([[close[0]], close[:-1]])
        prev_denom = np.where(prev_close != 0, prev_close, 1.0)
        returns = (close - prev_close) / prev_denom

        log_vol = np.log1p(volume)
        max_lv = log_vol.max()
        log_vol = log_vol / max_lv if max_lv > 0 else log_vol

        parts.append(np.concatenate([hl_range, returns, log_vol]).astype(np.float32))

    return np.concatenate(parts)


def _state_dim(n_timeframes: int, lookback: int) -> int:
    return n_timeframes * 3 * lookback


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class _DQNetwork:
    """
    Two-layer MLP Q-network implemented with PyTorch.

    Falls back gracefully when ``torch`` is unavailable (inference returns
    random Q-values so the agent still runs, just untrained).
    """

    def __init__(self, input_dim: int, hidden_size: int, n_actions: int) -> None:
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_actions = n_actions
        self._net = None
        self._opt = None
        self._loss_fn = None

        try:
            import torch
            import torch.nn as nn

            self._net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, n_actions),
            )
            self._opt = torch.optim.Adam(self._net.parameters(), lr=1e-3)
            self._loss_fn = nn.MSELoss()
        except ImportError:
            pass  # Torch not available — random fallback

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Return Q-values for *state* as a (n_actions,) float array."""
        if self._net is None:
            return np.random.rand(self.n_actions).astype(np.float32)

        import torch

        with torch.no_grad():
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self._net(t).squeeze(0).numpy()

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Return Q-values for a batch of states as a ``(n, n_actions)`` array.

        Parameters
        ----------
        states:
            2-D float32 array of shape ``(n, input_dim)``.

        Returns
        -------
        np.ndarray, shape ``(n, n_actions)``
        """
        if self._net is None:
            return np.random.rand(len(states), self.n_actions).astype(np.float32)

        import torch

        with torch.no_grad():
            t = torch.tensor(states, dtype=torch.float32)
            return self._net(t).numpy()

    def train_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
    ) -> float:
        """Perform one gradient update; returns the scalar loss."""
        if self._net is None:
            return 0.0

        import torch

        # Update learning rate in-place
        for pg in self._opt.param_groups:
            pg["lr"] = learning_rate

        s = torch.tensor(states, dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.long)
        tgt = torch.tensor(targets, dtype=torch.float32)

        q_vals = self._net(s)
        q_pred = q_vals.gather(1, a.unsqueeze(1)).squeeze(1)

        loss = self._loss_fn(q_pred, tgt)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return float(loss.item())

    def copy_weights_from(self, other: "_DQNetwork") -> None:
        """Copy weights from *other* into this network (target-net sync)."""
        if self._net is None or other._net is None:
            return
        import torch

        self._net.load_state_dict(other._net.state_dict())

    def state_dict_bytes(self) -> bytes:
        """Serialise network weights to bytes for database storage."""
        if self._net is None:
            return b""
        import torch

        buf = io.BytesIO()
        torch.save(self._net.state_dict(), buf)
        return buf.getvalue()

    def load_state_dict_bytes(self, data: bytes) -> None:
        """Restore network weights from bytes previously produced by
        :meth:`state_dict_bytes`."""
        if self._net is None or not data:
            return
        import torch

        buf = io.BytesIO(data)
        state = torch.load(buf, weights_only=True)
        self._net.load_state_dict(state)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

_Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class _ReplayBuffer:
    """Circular experience-replay buffer."""

    def __init__(self, capacity: int) -> None:
        self._buf: deque[_Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._buf.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[_Transition]:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class RLTradingAgent:
    """
    DQN-based reinforcement learning agent for multi-timeframe ETH trading.

    Hyperparameters are seeded from :class:`~eth_algo_trading.config.RLConfig`
    (which acts as the source of *constants*) and can be refreshed at any time
    from a :class:`~eth_algo_trading.db.hyperparams.HyperparamDB` connection.

    Parameters
    ----------
    config:
        RL hyperparameter configuration.  Defaults to ``RLConfig()`` if not
        supplied.
    db:
        Optional database handle.  When provided, hyperparameters are loaded
        from the DB on construction (DB values override *config*).
    """

    def __init__(
        self,
        config: Optional[RLConfig] = None,
        db: Optional[HyperparamDB] = None,
    ) -> None:
        self._cfg = config or RLConfig()
        self._db = db

        # Optionally refresh hyperparameters from the database.
        # Structural params (lookback_bars, hidden_size, timeframes) ARE applied
        # here because the networks have not been built yet.
        if self._db is not None:
            self._apply_db_hyperparams(skip_structural=False)

        self._step = 0
        self._is_fitted = False

        input_dim = _state_dim(len(self._cfg.timeframes), self._cfg.lookback_bars)
        self._online_net = _DQNetwork(input_dim, self._cfg.hidden_size, N_ACTIONS)
        self._target_net = _DQNetwork(input_dim, self._cfg.hidden_size, N_ACTIONS)
        self._target_net.copy_weights_from(self._online_net)

        self._replay = _ReplayBuffer(self._cfg.replay_capacity)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _aligned_tfs(self, ohlcv_by_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Return an ordered dict covering exactly the configured timeframes.

        Any timeframe not present in *ohlcv_by_tf* is filled with an empty
        DataFrame so that the resulting state vector has the correct dimension.
        """
        _empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        return {tf: ohlcv_by_tf.get(tf, _empty) for tf in self._cfg.timeframes}

    # ------------------------------------------------------------------
    # Hyperparameter management
    # ------------------------------------------------------------------

    # Parameters that determine network input/hidden dimensions; changing them
    # after the networks are built would silently corrupt the agent.
    _STRUCTURAL_PARAMS: frozenset = frozenset({"lookback_bars", "hidden_size", "timeframes"})

    def _apply_db_hyperparams(self, *, skip_structural: bool = True) -> None:
        """
        Override *_cfg* fields with any values stored in the database.

        Parameters
        ----------
        skip_structural:
            When ``True`` (the default, used by :meth:`refresh_hyperparams`),
            structural parameters (``lookback_bars``, ``hidden_size``,
            ``timeframes``) are skipped to prevent the already-initialised
            network weight tensors from becoming inconsistent with the config.
            Pass ``False`` only during :meth:`__init__`, before the networks
            are built, to allow DB values to fully override *config*.
        """
        if self._db is None:
            return
        stored = self._db.load_all()
        cfg_dict = asdict(self._cfg)
        for key, val in stored.items():
            if key in cfg_dict:
                if skip_structural and key in self._STRUCTURAL_PARAMS:
                    continue
                setattr(self._cfg, key, val)

    def refresh_hyperparams(self) -> None:
        """
        Pull updated hyperparameters from the database.

        Call this periodically (e.g. after each episode) to pick up values
        that were written to the DB by an external tuning process.
        """
        self._apply_db_hyperparams()

    # ------------------------------------------------------------------
    # Epsilon-greedy exploration
    # ------------------------------------------------------------------

    def _epsilon(self) -> float:
        """Linearly decayed exploration rate."""
        progress = min(1.0, self._step / max(1, self._cfg.epsilon_decay))
        return self._cfg.epsilon_start + progress * (
            self._cfg.epsilon_end - self._cfg.epsilon_start
        )

    # ------------------------------------------------------------------
    # Core RL methods
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Parameters
        ----------
        state:
            Flat feature vector from :func:`_build_state`.
        greedy:
            When ``True``, always select the highest Q-value action (no
            exploration).  Use this during inference.

        Returns
        -------
        int
            0 = flat, 1 = long, 2 = short.
        """
        if not greedy and random.random() < self._epsilon():
            return random.randrange(N_ACTIONS)
        q_vals = self._online_net.predict(state)
        return int(np.argmax(q_vals))

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Add a transition to the replay buffer."""
        self._replay.push(state, action, reward, next_state, done)

    def update(self) -> float:
        """
        Sample a mini-batch and perform one gradient step.

        Returns
        -------
        float
            Training loss (0.0 if the buffer is too small to sample).
        """
        if len(self._replay) < self._cfg.batch_size:
            return 0.0

        batch = self._replay.sample(self._cfg.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_arr = np.array(states, dtype=np.float32)
        next_states_arr = np.array(next_states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int64)
        rewards_arr = np.array(rewards, dtype=np.float32)
        dones_arr = np.array(dones, dtype=np.float32)

        # Compute TD targets using the target network (batched for efficiency)
        next_q = np.max(self._target_net.predict_batch(next_states_arr), axis=1)
        targets = rewards_arr + self._cfg.gamma * next_q * (1.0 - dones_arr)

        loss = self._online_net.train_step(
            states_arr, actions_arr, targets, self._cfg.learning_rate
        )

        self._step += 1
        if self._step % self._cfg.target_update_freq == 0:
            self._target_net.copy_weights_from(self._online_net)

        return loss

    # ------------------------------------------------------------------
    # Training on historical data
    # ------------------------------------------------------------------

    def fit(
        self,
        ohlcv_by_tf: Dict[str, pd.DataFrame],
        n_episodes: int = 5,
    ) -> "RLTradingAgent":
        """
        Train the agent by simulating trading episodes on historical data.

        For each episode the agent walks forward through the price history,
        selects actions with epsilon-greedy policy, receives a reward equal
        to the signed next-bar return (positive for correct directional bets,
        negative otherwise), and stores the transitions for batch updates.

        .. note::
            Training uses only the primary timeframe (``config.timeframes[0]``)
            for state construction.  Secondary timeframes in *ohlcv_by_tf* are
            ignored during training because their integer indices do not
            correspond to the same timestamps as the primary timeframe.
            Multi-timeframe state vectors are only meaningful at inference time
            when the caller provides pre-aligned DataFrames.

        Parameters
        ----------
        ohlcv_by_tf:
            Dict mapping timeframe label → OHLCV DataFrame.  The primary
            timeframe (``config.timeframes[0]``) must be present and have at
            least ``lookback_bars + 1`` rows.
        n_episodes:
            Number of passes through the dataset.

        Returns
        -------
        RLTradingAgent
            self (for chaining).
        """
        # Use the shortest timeframe as the step driver
        primary_tf = self._cfg.timeframes[0]
        primary = ohlcv_by_tf.get(primary_tf)
        if primary is None or len(primary) < self._cfg.lookback_bars + 1:
            return self

        close_arr = primary["close"].values

        for _ in range(n_episodes):
            # Optionally refresh hyperparameters between episodes
            if self._db is not None:
                self.refresh_hyperparams()

            for i in range(self._cfg.lookback_bars, len(close_arr) - 1):
                # Build state from the primary TF window ending at bar i.
                # Secondary TFs are excluded from training to avoid index
                # misalignment (they have different bar counts for the same
                # wall-clock period).
                start = max(0, i - self._cfg.lookback_bars + 1)
                window = self._aligned_tfs({
                    primary_tf: primary.iloc[start: i + 1]
                })
                state = _build_state(window, self._cfg.lookback_bars)

                action = self.select_action(state)

                # Reward: signed bar return aligned with action
                ret = (close_arr[i + 1] - close_arr[i]) / (close_arr[i] + 1e-8)
                if action == _ACTION_LONG:
                    reward = float(ret)
                elif action == _ACTION_SHORT:
                    reward = float(-ret)
                else:
                    reward = 0.0

                # Next state: one step further in the primary TF
                next_start = max(0, i - self._cfg.lookback_bars + 2)
                next_window = self._aligned_tfs({
                    primary_tf: primary.iloc[next_start: i + 2]
                })
                next_state = _build_state(next_window, self._cfg.lookback_bars)
                done = i + 2 >= len(close_arr)

                self.store_experience(state, action, reward, next_state, done)
                self.update()

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        ohlcv_by_tf: Dict[str, pd.DataFrame],
    ) -> Signal:
        """
        Generate a trading signal from the latest multi-timeframe bars.

        Parameters
        ----------
        ohlcv_by_tf:
            Dict mapping timeframe label → OHLCV DataFrame (most-recent bar
            last).  Any configured timeframe missing from the dict is treated
            as empty (zero-padded state features).

        Returns
        -------
        Signal
            Direction derived from the greedy Q-value action.  *strength* is
            the softmax-normalised probability of the chosen action.
        """
        full_tf = self._aligned_tfs(ohlcv_by_tf)
        state = _build_state(full_tf, self._cfg.lookback_bars)
        q_vals = self._online_net.predict(state)

        # Softmax for a calibrated strength score
        exp_q = np.exp(q_vals - q_vals.max())
        probs = exp_q / exp_q.sum()

        action = int(np.argmax(q_vals))
        direction = _ACTION_TO_DIRECTION[action]
        strength = round(float(probs[action]), 4)

        return Signal(
            direction=direction,
            strength=strength,
            strategy="rl_agent",
            metadata={
                "q_values": {
                    "flat": round(float(q_vals[_ACTION_FLAT]), 4),
                    "long": round(float(q_vals[_ACTION_LONG]), 4),
                    "short": round(float(q_vals[_ACTION_SHORT]), 4),
                },
                "is_fitted": self._is_fitted,
            },
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, db: Optional[HyperparamDB] = None) -> None:
        """
        Save model weights and current hyperparameters to the database.

        Parameters
        ----------
        db:
            Target database.  Falls back to ``self._db`` if not supplied.
        """
        target_db = db or self._db
        if target_db is None:
            return

        target_db.save_blob("rl_online_weights", self._online_net.state_dict_bytes())
        target_db.save_blob("rl_target_weights", self._target_net.state_dict_bytes())

        # Persist architecture metadata alongside weights so that
        # load_checkpoint() can verify compatibility.
        target_db.set("rl_arch_input_dim", self._online_net.input_dim)
        target_db.set("rl_arch_hidden_size", self._online_net.hidden_size)

        # Persist current scalar hyperparameters
        cfg_dict = asdict(self._cfg)
        for key, val in cfg_dict.items():
            target_db.set(key, val)

    def load_checkpoint(self, db: Optional[HyperparamDB] = None) -> None:
        """
        Restore model weights and hyperparameters from the database.

        Raises
        ------
        ValueError
            If the saved network architecture (``input_dim`` or
            ``hidden_size``) does not match the current agent's configuration.

        Parameters
        ----------
        db:
            Source database.  Falls back to ``self._db`` if not supplied.
        """
        source_db = db or self._db
        if source_db is None:
            return

        # Validate architecture compatibility before loading weights.
        saved_input_dim = source_db.get("rl_arch_input_dim")
        saved_hidden_size = source_db.get("rl_arch_hidden_size")

        if saved_input_dim is not None and saved_input_dim != self._online_net.input_dim:
            raise ValueError(
                f"Checkpoint input_dim {saved_input_dim} does not match current "
                f"agent input_dim {self._online_net.input_dim}. "
                "Ensure lookback_bars and timeframes match the saved configuration."
            )
        if saved_hidden_size is not None and saved_hidden_size != self._online_net.hidden_size:
            raise ValueError(
                f"Checkpoint hidden_size {saved_hidden_size} does not match current "
                f"agent hidden_size {self._online_net.hidden_size}."
            )

        self._apply_db_hyperparams()

        online_bytes = source_db.load_blob("rl_online_weights")
        if online_bytes:
            self._online_net.load_state_dict_bytes(online_bytes)
        target_bytes = source_db.load_blob("rl_target_weights")
        if target_bytes:
            self._target_net.load_state_dict_bytes(target_bytes)

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def config(self) -> RLConfig:
        return self._cfg
