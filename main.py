from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import httpx
import openai
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model=genai.GenerativeModel(
model_name="gemini-1.5-flash")
# system_instruction="You summarize the given content",

class SearchQuery(BaseModel):
    query: str
    websites: list[str]


def calculate_relevance(query: str, summary: str) -> float:
    # Vectorize the query and summary
    vectorizer = TfidfVectorizer().fit_transform([query, summary])
    vectors = vectorizer.toarray()
    
    # Calculate cosine similarity between the query and summary
    cosine_sim = cosine_similarity([vectors[0]], [vectors[1]])
    return cosine_sim[0][0]


@app.post("/search")
async def search(query: SearchQuery):
    results = []

    # Search the specified websites
    async with httpx.AsyncClient() as client:
        for website in query.websites:
            try:
                # Ensure the URL is correctly formatted
                base_url = f"https://{website}"
                response = await client.get(f"{base_url}/search?q={query.query}")
                soup = BeautifulSoup(response.content, "html.parser")
                links = soup.find_all('a')

                # Collect top 10 links
                for link in links[:10]:
                    href = link.get('href')

                    # Ensure we have a full URL
                    url = urljoin(base_url, href)

                    # Filter out non-HTTP URLs and malformed ones
                    if not urlparse(url).scheme.startswith("http"):
                        continue

                    try:
                        page_response = await client.get(url)
                        page_text = page_response.text
                        prompt=f"User prompt: Summarize the following text: {page_text}"
                        gpt_response = model.generate_content(f"{prompt}",
                                                            generation_config=genai.types.GenerationConfig(
                                                                max_output_tokens=150,
                                                            ),)
                        summary = gpt_response.text.strip()
                        # Use OpenAI to summarize the page content
                        # summary = openai.Completion.create(
                        #     engine="text-davinci-003",
                        #     prompt=f"Summarize the following text: {page_text}",
                        #     max_tokens=150
                        # )
                        relevance_score = calculate_relevance(query.query, summary)
                        results.append({
                            "url": url,
                            "summary": summary,
                            "relevance_score": relevance_score
                        })

                    except httpx.RequestError as e:
                        print(f"Failed to fetch {url}: {e}")

            except httpx.RequestError as e:
                print(f"Failed to fetch search results from {website}: {e}")
    results = sorted(results, key=lambda x: x['relevance_score'], reverse=True)

    return {"results": results}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)


