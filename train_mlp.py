
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Charger dataset & scaler
DATASET='microgestes_dataset.csv'
SCALER='scaler_params.json'
assert os.path.exists(DATASET), f"Fichier manquant: {DATASET} (générez-le via generate_dataset.py)"
assert os.path.exists(SCALER), f"Fichier manquant: {SCALER} (générez-le via generate_dataset.py)"

df = pd.read_csv(DATASET)
with open(SCALER,'r', encoding='utf-8') as f:
    scaler = json.load(f)
cols = scaler['feature_names']
X = df[cols].values
y = df['label'].values

X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

Xmin = np.array(scaler['Xmin']); Xmax = np.array(scaler['Xmax'])

def minmax_scale(X):
    return (X - Xmin) / np.maximum((Xmax - Xmin), 1e-6)

X_train_n = minmax_scale(X_train).astype(np.float32)
X_val_n   = minmax_scale(X_val).astype(np.float32)
X_test_n  = minmax_scale(X_test).astype(np.float32)

num_features = len(cols)
num_classes  = len(np.unique(y))

# Modèle MLP minimal (16 neurones)
def build_mlp(hidden=16):
    model = keras.Sequential([
        layers.Input(shape=(num_features,)),
        layers.Dense(hidden, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_mlp(16)
model.fit(X_train_n, y_train, validation_data=(X_val_n, y_val), epochs=25, batch_size=64, verbose=2)
print('\nTest float:')
res = model.evaluate(X_test_n, y_test, verbose=0)
print(dict(zip(model.metrics_names, res)))
print('\nReport:\n', classification_report(y_test, np.argmax(model.predict(X_test_n), axis=1)))
model.save('model_float.h5')

# PTQ int8
print('\nConversion PTQ int8...')

def representative_data_gen():
    for i in range(min(200, len(X_train_n))):
        yield [X_train_n[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
with open('model_int8.tflite', 'wb') as f:
    f.write(tflite_quant_model)
print('OK -> model_int8.tflite (taille:', len(tflite_quant_model), 'octets)')

# Optionnel: QAT + Pruning
try:
    import tensorflow_model_optimization as tfmot
    print('\nQAT + Pruning...')
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    end_epoch = 25
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0,
                                                                 final_sparsity=0.5,
                                                                 begin_step=0,
                                                                 end_step=end_epoch * (len(X_train_n)//64 + 1))
    }
    model_pruned = build_mlp(16)
    model_pruned = prune_low_magnitude(model_pruned, **pruning_params)
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    model_pruned.fit(X_train_n, y_train, validation_data=(X_val_n, y_val),
                     epochs=end_epoch, batch_size=64, callbacks=callbacks, verbose=2)
    model_pruned = tfmot.sparsity.keras.strip_pruning(model_pruned)

    # QAT
    qat_model = tfmot.quantization.keras.quantize_model(model_pruned)
    qat_model.compile(optimizer=keras.optimizers.Adam(1e-3),
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    qat_model.fit(X_train_n, y_train, validation_data=(X_val_n, y_val), epochs=10, batch_size=64, verbose=2)

    # Conversion int8
    converter2 = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.representative_dataset = representative_data_gen
    converter2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter2.inference_input_type = tf.int8
    converter2.inference_output_type = tf.int8
    tflite_qat_int8 = converter2.convert()
    with open('model_qat_pruned_int8.tflite', 'wb') as f:
        f.write(tflite_qat_int8)
    print('OK -> model_qat_pruned_int8.tflite (taille:', len(tflite_qat_int8), 'octets)')
except Exception as e:
    print('QAT+Pruning non exécuté (tfmot non installé?):', e)

print('\nTerminé.')
