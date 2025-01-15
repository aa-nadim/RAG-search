from crewai import Agent, Task, Crew, LLM
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import json
import os
from datetime import datetime
# Need R&D on CHROMA and GPT Embedding Text Embedding
class DocumentationCrawler:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited_urls = set()
        self.content_store = {}
        
    def crawl(self, url: str):
        if url in self.visited_urls or not url.startswith(self.base_url):
            return
        
        try:
            print(f"Crawling: {url}")
            self.visited_urls.add(url)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            main_content = soup.find('div', {'class': 'document'}) or soup.find('main')
            content = main_content.get_text(strip=True) if main_content else ''
            
            links = [a['href'] for a in soup.find_all('a', href=True)]
            absolute_links = []
            for link in links:
                if link.startswith('/'):
                    link = f"{self.base_url.rstrip('/')}{link}"
                if link.startswith(self.base_url):
                    absolute_links.append(link)
            
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            self.content_store[url] = {
                'chunks': chunks,
                'title': soup.title.string if soup.title else url
            }
            
            for link in absolute_links:
                self.crawl(link)
                
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")

class DocumentationChatbot:
    def __init__(self, base_url: str):
        self.llm = LLM(
            model="ollama/llama3.2:1b",
            base_url="http://localhost:11434"
        )
        
        self.crawler = DocumentationCrawler(base_url)
        self.vectorizer = None
        self.vectors = None
        self.chunks = []
        self.chunk_metadata = []
        self.chat_history = []
        self.base_url = base_url
        
    def create_agents(self):
        return Agent(
            role='Documentation Expert',
            goal='Answer questions about documentation accurately',
            backstory='I am an expert at understanding and explaining documentation',
            llm=self.llm
        )
    
    def process_content(self):
        for url, data in self.crawler.content_store.items():
            for chunk in data['chunks']:
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'url': url,
                    'title': data['title']
                })
        
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.chunks)
    
    def initialize_knowledge_base(self):
        """Initialize or load knowledge base"""
        kb_file = f"knowledge_base_{self.base_url.replace('/', '_')}.json"
        
        if os.path.exists(kb_file):
            print("Loading existing knowledge base...")
            self.load_knowledge_base(kb_file)
        else:
            print("Building new knowledge base...")
            self.crawler.crawl(self.base_url)
            self.process_content()
            self.save_knowledge_base(kb_file)
        
        print(f"Knowledge base ready with {len(self.chunks)} chunks from {len(self.crawler.content_store)} pages")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        query_vector = self.vectorizer.transform([query])
        similarities = np.dot(self.vectors, query_vector.T).toarray().flatten()
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            results.append({
                'content': self.chunks[idx],
                'url': self.chunk_metadata[idx]['url'],
                'title': self.chunk_metadata[idx]['title'],
                'relevance_score': float(similarities[idx])
            })
        
        return results
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        # Create context from search results
        context = "\n\n".join([
            f"From {result['title']}:\n{result['content']}"
            for result in search_results if result['relevance_score'] > 0.1
        ])
        
        # Create task for the agent
        task = Task(
            description=f"""
            Answer the following question using the provided context.
            Question: {query}
            
            Context:
            {context}
            
            Previous conversation:
            {self._format_chat_history()}
            """,
            expected_output="A detailed answer to the user's question based on the provided context.",
            agent=self.create_agents()
        )
        
        # Get response from agent
        crew = Crew(
            agents=[task.agent],
            tasks=[task]
        )
        
        try:
            response = crew.kickoff()
            return response
        except Exception as e:
            return f"I apologize, but I couldn't generate a proper response. This might be because I couldn't find relevant information in the documentation. Could you please rephrase your question or ask about a different topic?"
    
    def _format_chat_history(self, max_history: int = 5) -> str:
        if not self.chat_history:
            return "No previous conversation."
        
        recent_history = self.chat_history[-max_history:]
        formatted_history = []
        for entry in recent_history:
            formatted_history.append(f"Human: {entry['question']}")
            formatted_history.append(f"Assistant: {entry['answer']}\n")
        
        return "\n".join(formatted_history)
    
    def save_knowledge_base(self, filename: str):
        data = {
            'chunks': self.chunks,
            'metadata': self.chunk_metadata,
            'vectorizer': self.vectorizer.vocabulary_,
            'idf': self.vectorizer.idf_.tolist()
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def load_knowledge_base(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.chunks = data['chunks']
        self.chunk_metadata = data['metadata']
        
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.vocabulary_ = data['vectorizer']
        self.vectorizer.idf_ = np.array(data['idf'])
        
        self.vectors = self.vectorizer.transform(self.chunks)
    
    def save_chat_history(self):
        if not self.chat_history:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_history_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.chat_history, f, indent=2)
        
        print(f"\nChat history saved to {filename}")
    
    def chat_loop(self):
        print("\nWelcome to the Documentation Chatbot!")
        print("Ask any questions about the documentation. Type 'exit' to end the conversation.")
        print("Type 'save' to save the chat history.")
        print("\nNote: I'm currently processing documentation from:", self.base_url)
        print("Please ask questions related to this documentation.\n")
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() == 'exit':
                    self.save_chat_history()
                    print("\nGoodbye! Chat history has been saved.")
                    break
                
                if query.lower() == 'save':
                    self.save_chat_history()
                    continue
                
                if not query:
                    continue
                
                # Search relevant content
                results = self.search(query)
                
                if not any(result['relevance_score'] > 0.1 for result in results):
                    print("\nI couldn't find any relevant information in the documentation. Could you please rephrase your question or ask about a different topic?")
                    continue
                
                # Generate response
                response = self.generate_response(query, results)
                
                # Store in chat history
                self.chat_history.append({
                    'question': query,
                    'answer': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Print response with source references
                print("\nAssistant:", response)
                print("\nSources:")
                for result in results[:3]:  # Show top 3 sources
                    if result['relevance_score'] > 0.1:
                        print(f"- {result['title']}: {result['url']}")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving chat history...")
                self.save_chat_history()
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again with a different question.")


def main():
    # Get documentation URL from user
    default_url = "https://thecatapi.com/"
    url = input(f"Enter documentation URL (press Enter for default: {default_url}): ").strip()
    url = url if url else default_url
    
    # Initialize chatbot
    chatbot = DocumentationChatbot(url)
    
    # Initialize knowledge base
    print("\nPreparing knowledge base...")
    chatbot.initialize_knowledge_base()
    
    # Start chat loop
    chatbot.chat_loop()

if __name__ == "__main__":
    main()