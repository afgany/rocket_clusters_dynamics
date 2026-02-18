"""Tests for Engine, Environment, and DampingParameters models."""

import json

import pytest
from pydantic import ValidationError

from cre.models.engine import Engine
from cre.models.environment import DampingParameters, Environment


# --- Engine model tests ---


def _make_merlin() -> Engine:
    return Engine(
        name="Merlin 1D",
        thrust_sl=845_000.0,
        thrust_vac=914_000.0,
        chamber_pressure=97e5,
        chamber_diameter=0.36,
        nozzle_exit_diameter=0.92,
        expansion_ratio=16.0,
        mass=470.0,
        isp_sl=282.0,
        isp_vac=311.0,
        gamma=1.25,
        sound_speed=1240.0,
        injector_type="pintle",
        cycle="gas_generator",
        n_range=(0.5, 3.0),
        tau_range=(0.5e-3, 5.0e-3),
    )


class TestEngine:
    def test_instantiation(self):
        engine = _make_merlin()
        assert engine.name == "Merlin 1D"
        assert engine.thrust_sl == 845_000.0
        assert engine.chamber_pressure == 97e5

    def test_optional_fields_none(self):
        """RVac has no sea-level thrust."""
        engine = Engine(
            name="RVac 2",
            thrust_sl=None,
            thrust_vac=2_530_000.0,
            chamber_pressure=300e5,
            chamber_diameter=0.42,
            nozzle_exit_diameter=2.4,
            expansion_ratio=85.0,
            mass=1700.0,
            isp_sl=None,
            isp_vac=363.0,
            sound_speed=1310.0,
            injector_type="coaxial_swirl",
            cycle="ffscc",
            n_range=(0.3, 2.0),
            tau_range=(0.2e-3, 2.0e-3),
        )
        assert engine.thrust_sl is None
        assert engine.isp_sl is None

    def test_json_round_trip(self):
        engine = _make_merlin()
        json_str = engine.model_dump_json()
        restored = Engine.model_validate_json(json_str)
        assert restored == engine

    def test_dict_round_trip(self):
        engine = _make_merlin()
        d = engine.model_dump()
        restored = Engine.model_validate(d)
        assert restored == engine

    def test_frozen(self):
        engine = _make_merlin()
        with pytest.raises(ValidationError):
            engine.name = "Modified"  # type: ignore[misc]

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            Engine(name="Bad")  # type: ignore[call-arg]

    def test_default_gamma(self):
        engine = _make_merlin()
        assert engine.gamma == 1.25


# --- Environment model tests ---


class TestEnvironment:
    def test_earth(self):
        env = Environment(
            name="earth_sl",
            ambient_pressure=101_325.0,
            acoustic_impedance=420.0,
            zeta_atmospheric=0.028,
        )
        assert env.name == "earth_sl"
        assert env.ambient_pressure == 101_325.0

    def test_vacuum(self):
        env = Environment(
            name="lunar_vacuum",
            ambient_pressure=0.0,
            acoustic_impedance=0.0,
            zeta_atmospheric=0.0,
        )
        assert env.zeta_atmospheric == 0.0

    def test_json_round_trip(self):
        env = Environment(
            name="earth_sl",
            ambient_pressure=101_325.0,
            acoustic_impedance=420.0,
            zeta_atmospheric=0.028,
        )
        restored = Environment.model_validate_json(env.model_dump_json())
        assert restored == env

    def test_frozen(self):
        env = Environment(
            name="earth_sl",
            ambient_pressure=101_325.0,
            acoustic_impedance=420.0,
            zeta_atmospheric=0.028,
        )
        with pytest.raises(ValidationError):
            env.name = "modified"  # type: ignore[misc]


# --- DampingParameters tests ---


class TestDampingParameters:
    def test_defaults(self):
        dp = DampingParameters()
        assert dp.zeta_internal == 0.015
        assert dp.zeta_nozzle == 0.020
        assert dp.zeta_feed == 0.005
        assert dp.zeta_coupling_max == 0.022
        assert dp.zeta_atmospheric == 0.028

    def test_override(self):
        dp = DampingParameters(zeta_internal=0.025)
        assert dp.zeta_internal == 0.025
        assert dp.zeta_nozzle == 0.020  # other defaults preserved

    def test_mutable(self):
        dp = DampingParameters()
        dp.zeta_internal = 0.030
        assert dp.zeta_internal == 0.030

    def test_json_round_trip(self):
        dp = DampingParameters(zeta_feed=0.010)
        restored = DampingParameters.model_validate_json(dp.model_dump_json())
        assert restored == dp
