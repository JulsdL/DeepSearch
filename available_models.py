from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

models = client.models.list()

for model in models.data:
    print(model.id)

print("------------------")
print("------------------")
# Ask the user to chose a model
model_name = input("Enter the model name: ")

print("------------------")
print("------------------")
try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": "Hello"}]
    )
    # If the model is accessible, print the response
    print(response.choices[0].message.content.strip())

    print("Access granted to the model")
except Exception as e:
    print("No access to this model:", str(e))
