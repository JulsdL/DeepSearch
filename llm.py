from openai import AsyncOpenAI


from config import OPENAI_API_KEY, LLM_MODEL

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Set the OpenAI API key

async def call_openai_async(messages, model=LLM_MODEL):
    """
    Asynchronously call the OpenAI chat completion API.
    Returns the assistant’s reply as a string.
    """
    try:
        response = await aclient.chat.completions.create(model=model,
        messages=messages,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Error calling OpenAI API:", e)
        return None
