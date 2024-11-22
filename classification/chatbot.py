from gpt4all import GPT4All

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", model_path="chatbot/", device="gpu")
# response = model.prompt("Hello, how are you?")
with model.chat_session():
    print(model.generate("Hello! (Result of user emotion classification: happy)", max_tokens=128))
