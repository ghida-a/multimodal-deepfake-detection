import librosa
import numpy as np
import sys
import os
from scipy.spatial.distance import cosine

def extract_embedding(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    embedding = np.mean(mfcc, axis=1)
    return embedding

def is_fake(real_path, test_path, threshold=0.3):
    real_embedding = extract_embedding(real_path)
    test_embedding = extract_embedding(test_path)
    similarity = 1 - cosine(real_embedding, test_embedding)

    print(f"\n🧠 Similarity Score: {similarity:.4f}")
    if similarity >= threshold:
        print("✅ Same Speaker (or very close)")
    else:
        print("❌ Likely Fake or Different Speaker")

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Usage: python voice_spoof_detector.py real.wav test.wav [threshold]")
    else:
        real = sys.argv[1]
        test = sys.argv[2]
        threshold = float(sys.argv[3]) if len(sys.argv) == 4 else 0.3

        if not os.path.exists(real) or not os.path.exists(test):
            print("❌ One or both audio files do not exist.")
        else:
            is_fake(real, test, threshold)
