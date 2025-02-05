import asyncio
import aiohttp
from llm import call_openai_async
from config import PERPLEXITY_API_KEY, JINA_API_KEY

async def generate_search_queries(user_query: str) -> list:
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Return only a Python list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]
    response = await call_openai_async(messages)
    if response:
        try:
            search_queries = eval(response)
            if isinstance(search_queries, list):
                return search_queries
            else:
                print("LLM did not return a list. Response:", response)
        except Exception as e:
            print("Error parsing search queries:", e, "\nResponse:", response)
    return []

async def perform_search(query: str, session: aiohttp.ClientSession) -> list:
    """
    Use the Perplexity API to perform a search.
    Note: Adjust the extraction of URLs based on Perplexity's actual response structure.
    """
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "query": query,
        "max_tokens": 1000
    }
    try:
        async with session.post("https://api.perplexity.ai/search",
                                  headers=headers,
                                  json=payload) as resp:
            if resp.status == 200:
                results = await resp.json()
                # Extract URLs from the response; adjust as needed.
                links = [item.get("url") for item in results.get("references", [])]
                return links
            else:
                print(f"Perplexity API error: {resp.status}")
                return []
    except Exception as e:
        print("Error performing Perplexity search:", e)
        return []

async def fetch_webpage_text(url: str, session: aiohttp.ClientSession) -> str:
    full_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    try:
        async with session.get(full_url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.text()
            else:
                text = await resp.text()
                print(f"Jina fetch error for {url}: {resp.status} - {text}")
                return ""
    except Exception as e:
        print("Error fetching webpage text with Jina:", e)
        return ""

async def is_page_useful(user_query: str, page_text: str) -> str:
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information relevant and useful for addressing the query. "
        "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openai_async(messages)
    if response:
        answer = response.strip()
        if answer in ["Yes", "No"]:
            return answer
        else:
            if "Yes" in answer:
                return "Yes"
            elif "No" in answer:
                return "No"
    return "No"

async def extract_relevant_context(user_query: str, search_query: str, page_text: str) -> str:
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
        "Return only the relevant context as plain text without commentary."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]
    response = await call_openai_async(messages)
    if response:
        return response.strip()
    return ""

async def get_new_search_queries(user_query: str, previous_search_queries, all_contexts) -> list:
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (for example, "
        "['new query1', 'new query2']). If you believe no further research is needed, respond with exactly <done>."
        "\nOutput only a Python list or the token <done> without any additional text."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    response = await call_openai_async(messages)
    if response:
        cleaned = response.strip()
        if cleaned == "<done>":
            return "<done>"
        try:
            new_queries = eval(cleaned)
            if isinstance(new_queries, list):
                return new_queries
            else:
                print("LLM did not return a list for new search queries. Response:", response)
                return []
        except Exception as e:
            print("Error parsing new search queries:", e, "\nResponse:", response)
            return []
    return []

async def generate_final_report(user_query: str, all_contexts) -> str:
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all relevant insights and conclusions without extraneous commentary."
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\nGathered Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]
    report = await call_openai_async(messages)
    return report

async def process_link(link: str, user_query: str, search_query: str, session: aiohttp.ClientSession) -> str:
    print(f"Fetching content from: {link}")
    page_text = await fetch_webpage_text(link, session)
    if not page_text:
        return None
    usefulness = await is_page_useful(user_query, page_text)
    print(f"Page usefulness for {link}: {usefulness}")
    if usefulness == "Yes":
        context = await extract_relevant_context(user_query, search_query, page_text)
        if context:
            print(f"Extracted context from {link} (first 200 chars): {context[:200]}")
            return context
    return None

async def research_retrieval(user_query: str, iteration_limit: int = 10) -> str:
    """
    Main asynchronous research loop.
    """
    aggregated_contexts = []
    all_search_queries = []
    iteration = 0

    async with aiohttp.ClientSession() as session:
        # Get initial search queries
        new_search_queries = await generate_search_queries(user_query)
        if not new_search_queries:
            return "No search queries were generated by the LLM. Exiting."
        all_search_queries.extend(new_search_queries)

        # Iterative research loop
        while iteration < iteration_limit:
            print(f"\n=== Iteration {iteration + 1} ===")
            iteration_contexts = []

            # Perform concurrent Perplexity searches for each search query.
            search_tasks = [perform_search(query, session) for query in new_search_queries]
            search_results = await asyncio.gather(*search_tasks)

            # Aggregate unique links and record the search query that produced them.
            unique_links = {}
            for idx, links in enumerate(search_results):
                query = new_search_queries[idx]
                for link in links:
                    if link not in unique_links:
                        unique_links[link] = query

            print(f"Aggregated {len(unique_links)} unique links from this iteration.")

            # Process each link concurrently.
            link_tasks = [
                process_link(link, user_query, unique_links[link], session)
                for link in unique_links
            ]
            link_results = await asyncio.gather(*link_tasks)

            # Collect non-None contexts.
            for res in link_results:
                if res:
                    iteration_contexts.append(res)

            if iteration_contexts:
                aggregated_contexts.extend(iteration_contexts)
            else:
                print("No useful contexts were found in this iteration.")
                break

            # Ask the LLM if further research is needed.
            new_search_queries = await get_new_search_queries(user_query, all_search_queries, aggregated_contexts)
            if new_search_queries == "<done>":
                print("LLM indicated that no further research is needed.")
                break
            elif new_search_queries:
                print("LLM provided new search queries:", new_search_queries)
                all_search_queries.extend(new_search_queries)
            else:
                print("LLM did not provide any new search queries. Ending the loop.")
                break

            iteration += 1

        print("\nGenerating final report...")
        final_report = await generate_final_report(user_query, aggregated_contexts)
        return final_report
