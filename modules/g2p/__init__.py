from typing import Union
from modules.g2p.main import G2p_en, G2p_vi
from modules.g2p.symbols import symbols
# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def _symbols_to_sequence(symbols: Union[list, str]):
    if isinstance(symbols, str): symbols = symbols.split()

    return [_symbol_to_id[s[:-1] if s.startswith("@") and s[-1].isdigit() else s] for s in symbols]


def _sequence_to_symbols(sequence: Union[list, str]):
    if isinstance(sequence, str): sequence = sequence.split()
    
    return '_'.join([_id_to_symbol[int(s)] for s in sequence])
