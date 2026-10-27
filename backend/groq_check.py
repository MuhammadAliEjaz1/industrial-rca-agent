"""One tiny Groq call (about 100-300 tokens) to check: the key works, the reasoning_effort setting is accepted, and
token usage is reported. From backend/:   python groq_check.py"""
import sys
sys.path.insert(0, ".")
from graph.nodes import get_llm

llm = get_llm()
msg = llm.invoke("Reply with the single word: ready")
print("answer:", msg.content)
print("usage :", getattr(msg, "usage_metadata", None))
print("OK - the key and settings work.")
