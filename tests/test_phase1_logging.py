import pytest
import numpy as np  # Added import for state extractor dtype/shape assertions
from src.logging.run_logger import RunLogger
from tests.factory import create_test_model

pytestmark = pytest.mark.phase1

# Assumptions for test environment (toy config) override for speed
TEST_STEPS = 3
# Larger step count for compression effectiveness
COMPRESSION_STEPS = 30


# Helper to create a small model using existing factory with safe area dimensions
def _model():
    model, _ = create_test_model(num_agents=10, num_personalities=3, height=10,
                                 width=8, num_areas=2, av_area_width=2,
                                 av_area_height=2)
    return model


def test_step_level_parquet_schema_minimal(tmp_path):
    model = _model()
    rl = RunLogger(tmp_path, run_id=0, preset='minimal', agent_logging=False)
    for s in range(TEST_STEPS):
        model.step(); rl.log_step(s, model)
    rl.finalize()
    steps_parquet = tmp_path / 'steps.parquet'
    assert steps_parquet.exists(), 'steps.parquet not written'
    # Placeholder: schema validation will be added after implementation
    # Expected columns minimal preset
    expected_cols = {'run_id','step','collective_assets','gini_index','turnout'}
    import pyarrow.parquet as pq
    table = pq.read_table(steps_parquet)
    cols = set(table.schema.names)
    assert expected_cols.issubset(cols), f'Missing columns: {expected_cols - cols}'


def test_logging_presets_column_differences(tmp_path):
    presets = {}
    for preset in ['minimal','standard','full']:
        model = _model()
        out_dir = tmp_path / preset
        out_dir.mkdir()
        rl = RunLogger(out_dir, run_id=0, preset=preset, agent_logging=False)
        for s in range(2):
            model.step(); rl.log_step(s, model)
        rl.finalize()
        import pyarrow.parquet as pq
        table = pq.read_table(out_dir / 'steps.parquet')
        presets[preset] = set(table.schema.names)
    assert presets['minimal'] < presets['standard'] < presets['full'], 'Preset column sets not strictly increasing'


def test_agent_logging_toggle(tmp_path):
    model = _model()
    # Without agent logging
    rl_no = RunLogger(tmp_path / 'no', run_id=0, preset='minimal', agent_logging=False)
    rl_no.log_step(0, model); rl_no.finalize()
    assert not (tmp_path / 'no' / 'agents.parquet').exists(), 'agents.parquet should not exist when agent_logging=False'
    # With agent logging
    model2 = _model()
    rl_yes = RunLogger(tmp_path / 'yes', run_id=0, preset='minimal', agent_logging=True)
    rl_yes.log_step(0, model2); rl_yes.finalize()
    assert (tmp_path / 'yes' / 'agents.parquet').exists(), 'agents.parquet missing with agent_logging=True'


def test_compression_smaller_file(tmp_path):
    """Ensure snappy compression yields a smaller parquet for a larger dataset.
    Using small datasets can invert this expectation due to metadata overhead."""
    model_plain = _model()
    rl_plain = RunLogger(tmp_path / 'plain', run_id=0, preset='standard', agent_logging=False, compression=None)
    for s in range(COMPRESSION_STEPS):
        model_plain.step(); rl_plain.log_step(s, model_plain)
    rl_plain.finalize()
    size_plain = (tmp_path / 'plain' / 'steps.parquet').stat().st_size

    model_comp = _model()
    rl_comp = RunLogger(tmp_path / 'comp', run_id=0, preset='standard', agent_logging=False, compression='snappy')
    for s in range(COMPRESSION_STEPS):
        model_comp.step(); rl_comp.log_step(s, model_comp)
    rl_comp.finalize()
    size_comp = (tmp_path / 'comp' / 'steps.parquet').stat().st_size
    assert size_comp < size_plain, f'Compressed file ({size_comp}) not smaller than plain ({size_plain})'


def test_state_extractor_shapes(tmp_path):
    from src.logging.state_extractor import extract_state
    model = _model(); model.step()
    state = extract_state(model)
    assert 'agent_features' in state and 'area_features' in state and 'grid_tensor' in state
    af = state['agent_features']; ar = state['area_features']; gt = state['grid_tensor']
    assert af.dtype == np.float32 and ar.dtype == np.float32
    assert gt.dtype in (np.uint8, np.int32)
    assert af.shape[1] == 6  # columns
    assert ar.shape[1] == 3
    assert gt.shape == (model.height, model.width)


def test_presets_column_specifics(tmp_path):
    preset_info = {}
    for preset in ['minimal','standard','full']:
        model = _model()
        out_dir = tmp_path / f'pi_{preset}'
        out_dir.mkdir()
        rl = RunLogger(out_dir, run_id=1, preset=preset, agent_logging=False)
        for s in range(2):
            model.step(); rl.log_step(s, model)
        rl.finalize()
        import pyarrow.parquet as pq
        table = pq.read_table(out_dir / 'steps.parquet')
        preset_info[preset] = set(table.schema.names)
    # Color columns only in standard/full
    assert not any(c.startswith('color_') for c in preset_info['minimal'])
    assert any(c.startswith('color_') for c in preset_info['standard'])
    assert any(c.startswith('color_') for c in preset_info['full'])
    # grid_hash only in full
    assert 'grid_hash' not in preset_info['minimal']
    assert 'grid_hash' not in preset_info['standard']
    assert 'grid_hash' in preset_info['full']
    # Basic dtype checks
    import pyarrow.parquet as pq
    full_table = pq.read_table(tmp_path / 'pi_full' / 'steps.parquet')
    assert 'run_id' in full_table.schema.names and 'step' in full_table.schema.names
