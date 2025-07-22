from __future__ import annotations

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

from data_loader_icsv import load_all, SEG_SAMPLES
from main_statex_featex import model_emb_cnn, mixupLoss  # reuse architecture & loss

# ---------------------------------------
# 0. Config
# ---------------------------------------
RAW_DIM = SEG_SAMPLES               # 32 000
BATCH_SIZE = 32
EPOCHS = 1                          # first smoke‑test; increase later
N_SUBCLUSTERS = 8                   # reduce VRAM; 18×8=144 sub‑centers
OUT_DIR = "outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------
# 1. Data
# ---------------------------------------
(train_X, train_y), (eval_X, eval_y), (test_X, _, test_ids), encoder = load_all()
num_classes = len(encoder.classes_)

y_train_cat = tf.keras.utils.to_categorical(train_y, num_classes)
y_eval_cat  = tf.keras.utils.to_categorical(eval_y,  num_classes)

print("Data loaded:")
print("  Train:", train_X.shape)
print("  Eval :", eval_X.shape)
print("  Test :", test_X.shape)

# ---------------------------------------
# 2. Model
# ---------------------------------------
data_in, label_in, out_mix, out_ssl, out_ssl2 = model_emb_cnn(
    num_classes=num_classes,
    raw_dim=RAW_DIM,
    n_subclusters=N_SUBCLUSTERS,
    use_bias=False,
)

model = tf.keras.Model([data_in, label_in], [out_mix, out_ssl, out_ssl2])
model.compile(
    loss=[mixupLoss, mixupLoss, mixupLoss],
    optimizer=tf.keras.optimizers.Adam(),
    loss_weights=[1.0, 0.0, 1.0],
)
model.summary()

# ---------------------------------------
# 3. Train (only 1 epoch now)
# ---------------------------------------
model.fit(
    [train_X, y_train_cat],
    [y_train_cat, y_train_cat, y_train_cat],
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=([eval_X, y_eval_cat], [y_eval_cat, y_eval_cat, y_eval_cat]),
)

# ---------------------------------------
# 4. Embedding extraction
# ---------------------------------------
emb_model = tf.keras.Model(model.input, model.layers[-8].output)  # last shared Dense(128)

train_emb = emb_model.predict([train_X, np.zeros_like(y_train_cat)], batch_size=BATCH_SIZE)
eval_emb  = emb_model.predict([eval_X,  np.zeros_like(y_eval_cat )], batch_size=BATCH_SIZE)
test_emb  = emb_model.predict([test_X,  np.zeros((test_X.shape[0], num_classes))], batch_size=BATCH_SIZE)

# length normalise
train_emb /= np.linalg.norm(train_emb, axis=1, keepdims=True) + 1e-9
eval_emb  /= np.linalg.norm(eval_emb,  axis=1, keepdims=True) + 1e-9
test_emb  /= np.linalg.norm(test_emb,  axis=1, keepdims=True) + 1e-9

# ---------------------------------------
# 5. Simple anomaly score (global center cosine distance)
# ---------------------------------------
center = train_emb.mean(axis=0, keepdims=True)  # (1, 128)

score_eval = 1.0 - (eval_emb @ center.T).ravel()
score_test = 1.0 - (test_emb @ center.T).ravel()

# ---------------------------------------
# 6. Write submission CSV
# ---------------------------------------
np.savetxt(os.path.join(OUT_DIR, "eval_score.csv"), score_eval, fmt="%.6f")
np.savetxt(os.path.join(OUT_DIR, "test_score.csv"), score_test, fmt="%.6f")

print("\nSaved eval_score.csv & test_score.csv in", OUT_DIR)

# ---------------------------------------
# 7. (Optional) Quick sanity stats
# ---------------------------------------
print("Eval score stats — min/mean/max: {:.4f}/{:.4f}/{:.4f}".format(
    score_eval.min(), score_eval.mean(), score_eval.max()))
print("Test score stats — min/mean/max: {:.4f}/{:.4f}/{:.4f}".format(
    score_test.min(), score_test.mean(), score_test.max()))
