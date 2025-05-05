from openai import AsyncOpenAI

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://v100-01:8001/v1"


async def main():
    chat_response = await AsyncOpenAI(
        api_key="EMPTY",
        base_url=f"http://v100-01:8001/v1"
    ).chat.completions.create(
        **{
            'model': 'gpt-3.5-turbo', 'messages': [
                {'role': 'system', 'content': "Please complete the following code. Don't provide any testcase.\n"},
                {'role': 'user', 'content': 'import math\n\n\ndef poly(xs: list, x: float):\n    """\n    Evaluates polynomial with coefficients xs at point x.\n    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n\n    """\n    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])\n\n\ndef find_zero(xs: list):\n    """ xs are coefficients of a polynomial.\n    find_zero find x such that poly(x) = 0.\n    find_zero returns only only zero point, even if there are many.\n    Moreover, find_zero only takes list xs having even number of coefficients\n    and largest non zero coefficient as it guarantees\n    a solution.\n    >>> round(find_zero([1, 2]), 2) # f(x) = 1 + 2x\n    -0.5\n    >>> round(find_zero([-6, 11, -6, 1]), 2) # (x - 1) * (x - 2) * (x - 3) = -6 + 11x - 6x^2 + x^3\n    1.0\n    """\n'}
            ],
            'max_tokens': 483, 'temperature': 0, 'top_p': 1, 'timeout': 3600,
            'extra_body': {'ignore_eos': True, 'request_id': 'code_feedback--3--0', 'priority': 86.11715774761306}
        }
    )
    print("Chat response:", chat_response)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
