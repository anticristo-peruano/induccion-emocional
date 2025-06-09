import numpy as np
import pandas as pd
import yt_dlp
import os
from tqdm import tqdm
from multiprocessing import Pool

def open_process(CSV_FILE,SEED,N_TRACKS):
    need_cols = ["track_id", "track_name", "valence",
             "energy", "danceability", "popularity"]

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

def download_process(args):
    id, artist, track = args

    if os.path.exists(os.path.join('.songs',id + '.mp3')):
        pass
    else:
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
                    'quiet': True,
                    'no_warnings': True,
                    'progress_hooks': [],
                    'logger': type('NopLogger', (), {'debug': lambda *a, **k: None, 'info': lambda *a, **k: None, 'warning': lambda *a, **k: None, 'error': lambda *a, **k: None})()
                }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(track_name)
        except Exception as e:
            with open('.songs/.errors.txt','a') as f:
                f.write(f'Error downloading {id} - {track_name}: {e}\n')

def pseudospotify_downloader(csv_file,subprocesses=6):
    os.makedirs('.songs',exist_ok=True)
    df = pd.read_csv(csv_file)[['track_id', 'artists', 'track_name']].values.tolist()

    with Pool(processes=subprocesses) as p, tqdm(total=len(df)) as pbar:
        for _ in p.imap(download_process,df):
            pbar.update()
            pbar.refresh()