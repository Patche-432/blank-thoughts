from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

model_name = "Qwen/Qwen2.5-7B-Instruct"  # Instruct model is tuned for chat

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print("Loading model onto GPU...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0",  # force entirely onto GPU, no CPU offload
)
model.eval()
print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated() / 1e9:.1f} GB\n")


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        system_prompt = data.get("system", "You are a helpful assistant.")
        max_new_tokens = int(data.get("max_new_tokens", 200))

        conversation = [{"role": "system", "content": system_prompt}]
        for msg in messages:
            conversation.append({"role": msg["role"], "content": msg["content"]})

        encoded = tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        input_ids = encoded["input_ids"].to("cuda:0")
        attention_mask = encoded["attention_mask"].to("cuda:0")

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        print(f"[OK] {len(new_tokens)} tokens | VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        return jsonify({"reply": reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
