# Preinstall: pip install datasets tensorflow scikit-learn numpy pandas

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib

# 1. Loading dataset
dataset = load_dataset("dnagpt/dna_core_promoter")
df = pd.DataFrame(dataset['train'])

# 2. preprocessing
print("Original data：")
print(df.head())

# Labelling (adjust according to datasets)
# df['label'] = df['annotation'].apply(lambda x: 1 if x == 'positive' else 0)

# Distribution of sequence length
seq_lengths = df['sequence'].apply(len)
print(f"\n statistics for sequence length：\n{seq_lengths.describe()}")

# 3. Coding
vocab = {'A': 1, 'T': 2, 'C': 3, 'G': 4, '<pad>': 0}
max_length = int(seq_lengths.quantile(0.95))  # 95 quantile for maximum length

def dna_encoder(seq):
    encoded = [vocab.get(c, 0) for c in seq]  # 0 for unknown characters
    return encoded[:max_length] + [0]*(max_length - len(encoded))

X = np.array(df['sequence'].apply(dna_encoder).tolist())
y = np.array(df['label'])

# 4. Dividing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y,
    random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5, 
    stratify=y_temp,
    random_state=42
)

# 5. Deep learning
def build_model(input_shape, vocab_size=5):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size+1, output_dim=32, input_length=input_shape),
        
        # CNN
        layers.Conv1D(64, kernel_size=8, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),
        
        # BiLSTM
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        
        # FC
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')]
    )
    return model

# Initialization
model = build_model(max_length)
model.summary()

# 6. Training
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    save_best_only=True,
    monitor='val_f1_score'
)

class F1Score(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        val_pred = (self.model.predict(X_val) > 0.5).astype("int32")
        f1 = f1_score(y_val, val_pred)
        logs['val_f1_score'] = f1
        print(f" — val_f1: {f1:.4f}")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    class_weight={0: 1., 1: len(y_train[y_train==0])/len(y_train[y_train==1])},  # 类别平衡
    callbacks=[early_stop, checkpoint, F1Score()]
)

model.save('best_model.keras')

# 8. Assessment
best_model = models.load_model('best_model.keras')
test_pred = (best_model.predict(X_test) > 0.5).astype("int32").flatten()

print("\n Assessment Results：")
print(f"Accuracy：{accuracy_score(y_test, test_pred):.4f}")
print(f"F1_score：{f1_score(y_test, test_pred):.4f}")
print(f"Precision：{history.history['val_precision'][-1]:.4f}")
print(f"Recall：{history.history['val_recall'][-1]:.4f}")
print("Confusion_Matrix：")
print(confusion_matrix(y_test, test_pred))

# 9. Pipeline saving
pipeline = {
    'encoder': dna_encoder,
    'vocab': vocab,
    'max_length': max_length
}
joblib.dump(pipeline, 'dna_preprocessor.pkl')

# 10. Predict_
def predict_promoter_dl(seq, model_path='best_model.keras', preprocessor_path='dna_preprocessor.pkl'):
    
    # Load
    model = models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Preprocess
    encoded_seq = preprocessor['encoder'](seq)
    padded_seq = np.array([encoded_seq])
    
    # Predict
    proba = model.predict(padded_seq)[0][0]
    prediction = int(proba > 0.5)
    
    return {
        'is_promoter': prediction,
        'probability': proba if prediction else 1-proba,
        'confidence': float(proba)
    }

# Example
test_seq = "TATAATGGCTAGCATCGACTAGCTAGCATCGACTAGCTAGCATCGACTAGCTAGC"
result = predict_promoter_dl(test_seq)
print("\n预测结果：")
print(result)