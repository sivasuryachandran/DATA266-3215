"""Reload the saved HW4 fine-tune (model/) and run custom inference."""
import sys, torch
from transformers import BertTokenizer, BertForSequenceClassification

LABELS = ["Negative", "Neutral", "Positive"]
tok = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = BertForSequenceClassification.from_pretrained("model").cuda().eval()   # reloaded from disk

texts = sys.argv[1:] or [
    "this government has ruined the economy and everyone is suffering",
    "the meeting is scheduled for tuesday at noon",
    "what a fantastic achievement, so proud of the team",
]
enc = tok(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to("cuda")
with torch.no_grad():
    probs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits.softmax(-1).cpu()
for t, p in zip(texts, probs):
    print(f"{LABELS[int(p.argmax())]:9s} {p.max():.2f}  | {t}")
