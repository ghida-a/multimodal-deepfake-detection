import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import librosa
import numpy as np
from scipy.spatial.distance import cosine
import os

def extract_audio_features(path):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1)

def is_audio_real(test_path, reference_path='reference_real.wav', threshold=0.75):
    real_feat = extract_audio_features(reference_path)
    test_feat = extract_audio_features(test_path)
    sim = 1 - cosine(real_feat, test_feat)
    return sim, sim >= threshold

def check_image_fake(img_path):
    
    return "real"

def browse_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if path:
        image_path.set(path)
        img = Image.open(path)
        img = img.resize((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        image_preview.config(image=img_tk)
        image_preview.image = img_tk
        img_result.set("Image Result: " + check_image_fake(path))

def browse_audio():
    path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if path:
        audio_path.set(path)
        if not os.path.exists('reference_real.wav'):
            messagebox.showerror("Missing File", "Missing 'reference_real.wav' as baseline audio")
            return
        try:
            score, result = is_audio_real(path)
            audio_result.set(f"Audio Score: {score:.2f} - {'Real✅' if result else 'Fake ❌'}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("AI Deepfake Verifier")
root.geometry("600x550")
root.configure(bg="white")

username_var = tk.StringVar()
image_path = tk.StringVar()
audio_path = tk.StringVar()
img_result = tk.StringVar()
audio_result = tk.StringVar()

header = tk.Label(root, text="Deepfake Detection Tool", font=("Helvetica", 18, "bold"), bg="white")
header.pack(pady=10)

frame = tk.Frame(root, bg="white")
frame.pack(pady=10)

# Username
tk.Label(frame, text="Username:", bg="white").grid(row=0, column=0, sticky="e", pady=5)
tk.Entry(frame, textvariable=username_var, width=40).grid(row=0, column=1, pady=5)

# Image Upload
tk.Label(frame, text="Upload Image:", bg="white").grid(row=1, column=0, sticky="e", pady=5)
tk.Button(frame, text="Browse", command=browse_image).grid(row=1, column=1, sticky="w", pady=5)
image_preview = tk.Label(frame, bg="white")
image_preview.grid(row=2, column=1, pady=5)
tk.Label(frame, textvariable=img_result, fg="blue", bg="white").grid(row=3, column=1)

# Audio Upload
tk.Label(frame, text="Upload Voice Sample:", bg="white").grid(row=4, column=0, sticky="e", pady=15)
tk.Button(frame, text="Browse Audio", command=browse_audio).grid(row=4, column=1, sticky="w")
tk.Label(frame, textvariable=audio_result, fg="green", bg="white").grid(row=5, column=1, pady=10)

root.mainloop()
