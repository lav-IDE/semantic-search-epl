import pandas as pd

def select_relevant_columns(df):
    
    df = df.copy()
    df.drop(columns=["summary","Stadium","Attendance","Referee", "Date"], inplace=True, errors = "ignore")
    df.dropna(subset=["events"], inplace=True)

    df["id"] = df.index.astype(int)

    df["events"] = df["events"].str.lower()

    return df


def correcting_home_away_teams(df, teams):
    
    df = df.copy()
    
    df["Home"] = None
    df["Away"] = None

    for idx, row in df.iterrows():
    
        intro = " ".join(row["events"].split()[:40])

        found = []
    
        for team in teams:
            if team.lower() in intro:
                found.append(team)
            
        if len(found)>=2:
            df.loc[idx, "Home"] = found[0]
            df.loc[idx, "Away"] = found[1]
            
        else:
            df.loc[idx, "Home"] = "unknown"
            df.loc[idx, "Away"] = "unknown"
            
    return df

def build_chunk_dataframe(df, chunk_fn):
    
    
    records = []

    for _, row in df.iterrows():
        match_id = row["id"]
        home = row["Home"]
        away = row["Away"]

        text_chunks = chunk_fn(row["events"])

        for chunk in text_chunks:
            records.append({
                "match_id": match_id,
                "home": home,
                "away": away,
                "chunk_text": chunk
            })

    return pd.DataFrame(records).reset_index(drop=True)