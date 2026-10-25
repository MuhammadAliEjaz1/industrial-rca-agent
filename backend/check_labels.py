import sqlite3
import pandas as pd

conn = sqlite3.connect("industrial_rca.db")
df = pd.read_sql("""
    SELECT sr.tool_wear, fi.failure_type
    FROM sensor_readings sr
    LEFT JOIN failure_incidents fi
      ON sr.machine_id = fi.machine_id AND sr.timestamp = fi.date
""", conn)
conn.close()

twf = df[df["failure_type"] == "Tool Wear Failure"]
normal_in_range = df[(df["tool_wear"] >= 200) & (df["tool_wear"] <= 240) & (df["failure_type"].isna())]

print("Tool wear values for actual TWF failures:")
print(twf["tool_wear"].describe())
print(f"\nCount of NORMAL (non-failure) readings with tool_wear in 200-240 range: {len(normal_in_range)}")
print(f"Count of TWF failures with tool_wear in 200-240 range: {len(twf[(twf['tool_wear']>=200)&(twf['tool_wear']<=240)])}")