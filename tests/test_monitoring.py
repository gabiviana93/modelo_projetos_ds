from src.monitoring import population_stability_index

def test_psi_returns_float(psi_series_expected, psi_series_actual):
    psi = population_stability_index(psi_series_expected, psi_series_actual)

    assert isinstance(psi, float)
