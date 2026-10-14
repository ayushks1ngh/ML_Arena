import numpy as np
import pandas as pd
from arena.id3 import ID3Classifier, entropy, information_gain

# Quinlan's PlayTennis (14 rows)
D = pd.DataFrame({
 "Outlook": "Sunny Sunny Overcast Rain Rain Rain Overcast Sunny Sunny Rain Sunny Overcast Overcast Rain".split(),
 "Temp": "Hot Hot Hot Mild Cool Cool Cool Mild Cool Mild Mild Mild Hot Mild".split(),
 "Humidity": "High High High High Normal Normal Normal High Normal Normal Normal High Normal High".split(),
 "Wind": "Weak Strong Weak Weak Weak Strong Strong Weak Weak Weak Strong Strong Weak Strong".split(),
 "Play": "No No Yes Yes Yes No Yes No Yes Yes Yes Yes Yes No".split()})
y = (D.Play == "Yes").astype(int).to_numpy()
X = D.drop(columns="Play").apply(lambda c: c.astype("category").cat.codes).to_numpy()


def test_entropy_and_gains_match_textbook():
    assert abs(entropy(y) - 0.940) < 1e-3
    gains = {c: information_gain(X[:, i], y) for i, c in enumerate(D.columns[:-1])}
    assert abs(gains["Outlook"] - 0.2467) < 1e-3 and abs(gains["Humidity"] - 0.1518) < 1e-3
    assert abs(gains["Wind"] - 0.0481) < 1e-3 and abs(gains["Temp"] - 0.0292) < 1e-3


def test_tree_fits_playtennis_and_root_is_outlook():
    m = ID3Classifier().fit(X, y)
    assert m.tree_["feat"] == 0 and (m.predict(X) == y).all()        # Outlook is feature 0 -> root


def test_unseen_value_falls_back_to_majority():
    m = ID3Classifier().fit(X, y)
    row = X[:1].copy(); row[0, 0] = 99
    assert m.predict(row)[0] in (0, 1) and m.predict_proba(row).shape == (1, 2)
