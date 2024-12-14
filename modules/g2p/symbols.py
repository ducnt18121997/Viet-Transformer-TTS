"""
[*] Defines the set of phonemes used in text input to the model.
+ Format: C1 w V C2 T
_letters = _consonants + _medial + _vowels + _coda + _tone

+ Format: C1 w V_T C2
_letters = _consonants + _medial + [f"{v}_{t}" for v in _vowels for t in _tone] + _coda

[*]All CMU phonemes: 
_cmu = ["AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
        "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
        "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1",
        "EY2", "F", "G", "HH",
        "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L",
        "M", "N", "NG", "OW0", "OW1",
        "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
        "UH0", "UH1", "UH2", "UW",
        "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"]
"""

# special symbols
_pad = ["<pad>"]
_silent = ["<silent>"]
_eos = ["<space>", "</s>"]

# vietnamese symbols
_consonants = ["b", "ch", "d", "dd", "g", "h", "k", "kh", "kw", "l", "m", "n", "ng", "nh", "p", \
    "ph", "r", "s", "t", "th", "tr", "v", "x"]  # 23
_medial = ["wo", "wu"]  # 2
_vowels = ["a", "aa", "aw", "e", "ee", "i", "o", "oa", "oo", "ow", "u", "uw"] + ["ie", "uo", "wa"] # 12 monophthongs + 3 diphthongs
_coda = ["cz", "iz", "kz", "mz", "ngz", "nhz", "nz", "oz", "pz", "tz", "uz", "yz"]  # 12
_tone = ["0", "1", "2", "3", "4", "5"]

_letters = _consonants + _medial + [f"{v}_{t}" for v in _vowels for t in _tone] + _coda

# english symbols
_cmu = ["AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0",
        "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH",
        "EH0", "EH1", "EH2", "ER0", "ER1", "ER2", "EY0", "EY1",
        "EY2", "F", "G", "HH",
        "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "JH", "K", "L",
        "M", "N", "NG", "OW0", "OW1",
        "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T", "TH",
        "UH0", "UH1", "UH2", "UW",
        "UW0", "UW1", "UW2", "V", "W", "Y", "Z", "ZH"]
_cmu = [f"@{ph}" for ph in _cmu]

# all symbols 
symbols = _pad + _silent + _eos + _letters # + _cmu
symbols = [_.upper() for _ in symbols]
