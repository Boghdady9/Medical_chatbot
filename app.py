import gradio as gr
from transformers import pipeline
from typing import List, Tuple
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

peft_model_id = "BoghdadyJR/med"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Shortened system prompt
system_prompt="""
You are an AI medical information assistant designed to provide general health information and guidance. Important disclaimers:
1. You are NOT a substitute for professional medical care. You cannot diagnose conditions, prescribe medications, or provide personalized medical advice.
2. Always advise users to consult qualified healthcare professionals for:
   - Specific medical diagnoses
   - Treatment decisions
   - Changes to existing medications or treatments
   - Medical emergencies
   - Mental health crises
Your primary functions are to:
- Provide general, evidence-based health information from reliable medical sources
- Explain common medical terms and procedures in simple language
- Offer general wellness and preventive health information
- Help users understand basic medical concepts
- Guide users on when to seek professional medical care
- Share publicly available information about common conditions, symptoms, and general treatment approaches
When responding:
- Be clear, compassionate, and professional
- Use plain language that is easy to understand
- Include relevant disclaimers when appropriate
- Cite reputable medical sources when possible
- Maintain user privacy and confidentiality
- Express empathy while remaining objective
- Clearly state limitations and direct to professional care when needed
If users describe emergency situations or severe symptoms, immediately direct them to seek emergency medical care or call their local emergency services.
Remember: Your role is to inform and educate, not to diagnose or treat. When in doubt, always encourage users to consult with qualified healthcare professionals.
"""


class ChatMemory:
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.conversation_history: List[Tuple[str, str]] = []

    def add_interaction(self, user_message: str, bot_response: str):
        self.conversation_history.append((user_message, bot_response))
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

    def get_context(self) -> str:
        return "\n".join([
            f"User: {interaction[0]}\nAssistant: {interaction[1]}"
            for interaction in self.conversation_history
        ])


class Chatbot:
    def __init__(self):
        self.memory = ChatMemory()

    def generate_response(self, message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]

        try:
            response = generator(messages, max_new_tokens=512, return_full_text=False)[0]
            generated_text = response["generated_text"]

            self.memory.add_interaction(message, generated_text)

            return generated_text
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}"


# Initialize the chatbot
chatbot = Chatbot()

# Create the Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Medical Information Assistant")

    chat_history = gr.Chatbot(
        value=[],
        elem_id="chatbot",
        bubble_full_width=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            show_label=False,
            placeholder="Type your health-related question here...",
        )
        submit_button = gr.Button("➤", scale=0.1)


    def user(user_message: str, history: list) -> tuple:
        if not user_message.strip():
            return "", history
        history = history + [[user_message, None]]
        return "", history


    def bot(history: list) -> list:
        if not history:
            return history

        user_message = history[-1][0]
        bot_message = chatbot.generate_response(user_message)
        history[-1][1] = bot_message

        return history


    msg.submit(user, [msg, chat_history], [msg, chat_history], queue=False).then(
        bot, chat_history, chat_history
    )

    submit_button.click(user, [msg, chat_history], [msg, chat_history], queue=False).then(
        bot, chat_history, chat_history
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()
