import numpy as np
import pandas as pd
import yt_dlp
import os
from tqdm import tqdm

need_cols = ["track_id", "track_name", "valence",
             "energy", "danceability", "popularity"]

def open_process(CSV_FILE,SEED,N_TRACKS):
    if os.path.isfile(CSV_FILE):
        df_raw = pd.read_csv(CSV_FILE)
    else:
        # Fallback: create synthetic dataset if file not present
        n_synth = 200
        df_raw = pd.DataFrame({
            "track_id": [f"ID{i:04}" for i in range(n_synth)],
            "track_name": [f"Track_{i}" for i in range(n_synth)],
            "valence": np.random.rand(n_synth),
            "energy": np.random.rand(n_synth),
            "danceability": np.random.rand(n_synth),
            "popularity": np.random.randint(0, 101, size=n_synth)
        })

    missing = [c for c in need_cols if c not in df_raw.columns]
    if missing:
        raise KeyError(f"Columns missing in dataset: {missing}")

    df_raw = df_raw[need_cols].dropna().reset_index(drop=True)
    if len(df_raw) < N_TRACKS:
        N_TRACKS = len(df_raw)

    df = df_raw.sample(N_TRACKS, random_state=SEED).reset_index(drop=True)

    for col in ["valence", "energy", "danceability", "popularity"]:
        df[f"{col}_n"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)

    # ---------- Cosine similarity matrix ----------
    vec = df[["valence_n", "energy_n", "danceability_n"]].values
    vec /= np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
    sim_mat = np.clip(vec @ vec.T, 0, 1)   # shape (N_TRACKS, N_TRACKS)

    return df, sim_mat

def spotify_download(csv):
    df = pd.read_csv('.data/dataset.csv')

    for (_, (id, artist, track)) in tqdm(df[['track_id', 'artists', 'track_name']].iterrows()):
        track_name = f'{track} by {artist}'.replace("/","_")

        ydl_opts = {
                    'format': 'bestaudio/best',
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }],
                    'outtmpl': os.path.join('.songs', f"{id}.%(ext)s"),
                    'default_search': 'ytsearch',
                }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(track_name)
        except Exception as e:
            print(f'Error downloading {track_name}: {e}')