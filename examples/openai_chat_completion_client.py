from openai import OpenAI, AsyncOpenAI
import openai
import asyncio

async def openai_wrapper(client: AsyncOpenAI, **args):
    response = await client.chat.completions.create(
        **args)
    return response

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = AsyncOpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# chat_completion = client.chat.completions.create(
#     messages=[{
#         "role": "system",
#         "content": "You are a helpful assistant."
#     }, {
#         "role": "user",
#         "content": "Who won the world series in 2020?"
#     }, {
#         "role":
#         "assistant",
#         "content":
#         "The Los Angeles Dodgers won the World Series in 2020."
#     }, {
#         "role": "user",
#         "content": "Where was it played?"
#     }],
#     model=model,
#     max_tokens = 32, 
# )

extra_body = {
            "ignore_eos": True, 
            # "request_id": "test_dag"
        }

chat_completion = asyncio.run(openai_wrapper(
    client=client,
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":
        "assistant",
        "content":
        "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }],
    model="gpt-3.5-turbo",
    max_tokens = 32, 
    extra_body = extra_body,
))

print("Chat completion results:")
print(chat_completion)
