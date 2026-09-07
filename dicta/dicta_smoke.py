import json, torch
from transformers import AutoTokenizer, AutoModel

print("CUDA available:", torch.cuda.is_available())
tok = AutoTokenizer.from_pretrained("dicta-il/dictabert-joint", trust_remote_code=True)
model = AutoModel.from_pretrained("dicta-il/dictabert-joint", trust_remote_code=True)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

sent = "בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות"
out = model.predict([sent], tok, output_style="json")
print(json.dumps(out, ensure_ascii=False, indent=2)[:1500])
