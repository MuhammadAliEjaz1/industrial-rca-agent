from pathlib import Path

raw = Path(".env").read_bytes()
print("bytes:", len(raw), "| CRLF" if b"\r\n" in raw else "| LF")
for i, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
    if "=" in line:
        name = line.split("=", 1)[0]
        print(i, "name:", repr(name[:40]), "| total length:", len(line))
    else:
        print(i, "NO '=' on this line | length:", len(line))