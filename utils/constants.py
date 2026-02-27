confounder_list = ['common_input', 'indirect_connections', 'volume_conduction']

undirected_methods = ['coh', 'ciplv', 'wpli2_debiased', 'imaginary_coherence']

directed_methods = [
    'generalized_partial_directed_coherence',
    'direct_directed_transfer_function',
    'pairwise_spectral_granger_prediction',
]

all_methods = undirected_methods + directed_methods

comparison_pairs = [
    ('coh', 'ciplv'),
    ('coh', 'imaginary_coherence'),
    ('coh', 'wpli2_debiased'),
    ('ciplv', 'imaginary_coherence'),
    ('ciplv', 'wpli2_debiased'),
    ('imaginary_coherence', 'wpli2_debiased'),
    ('direct_directed_transfer_function', 'generalized_partial_directed_coherence'),
    ('direct_directed_transfer_function', 'pairwise_spectral_granger_prediction'),
    ('generalized_partial_directed_coherence', 'pairwise_spectral_granger_prediction'),
]
