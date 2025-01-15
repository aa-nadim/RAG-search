import psycopg2
from psycopg2 import sql
from crewai import Agent, Task, Crew, LLM
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
from datetime import datetime
import json

# Define the PostgreSQL connection parameters
DB_USERNAME = 'aa_nadim'
DB_PASSWORD = 'aa_nadim123'
DB_NAME = 'crewai_db'
DB_PORT = 5432
DB_HOST = 'localhost'

class DocumentationCrawler:
    def __init__(self, base_url: str, db_conn):
        self.base_url = base_url
        self.visited_urls = set()
        self.content_store = {}
        self.db_conn = db_conn
    
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

            # Store the content in PostgreSQL
            self.store_content(url, chunks, soup.title.string if soup.title else url)
            
            for link in absolute_links:
                self.crawl(link)
                
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")

    def store_content(self, url, chunks, title):
        """Store crawled content into PostgreSQL database."""
        try:
            cursor = self.db_conn.cursor()
            for chunk in chunks:
                cursor.execute(
                    "INSERT INTO documentation_chunks (url, title, chunk_content) VALUES (%s, %s, %s)",
                    (url, title, chunk)
                )
            self.db_conn.commit()
        except Exception as e:
            print(f"Error storing content in database: {str(e)}")


class DocumentationChatbot:
    def __init__(self, base_url: str, db_conn):
        self.llm = LLM(
            model="ollama/llama3.2:1b",
            base_url="http://localhost:11434"
        )
        
        self.crawler = DocumentationCrawler(base_url, db_conn)
        self.vectorizer = None
        self.vectors = None
        self.chunks = []
        self.chunk_metadata = []
        self.chat_history = []
        self.base_url = base_url
        self.db_conn = db_conn

        # Initialize database connection
        self.db_conn = psycopg2.connect(
            dbname="crewai_db", 
            user="aa_nadim", 
            password="aa_nadim123", 
            host="localhost", 
            port="5432"
        )
        self.db_cursor = self.db_conn.cursor()
    
    def create_agents(self):
        return Agent(
            role='Documentation Expert',
            goal='Answer questions about documentation accurately',
            backstory='I am an expert at understanding and explaining documentation',
            llm=self.llm
        )
    
    def process_content(self):
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT chunk_content, title, url FROM documentation_chunks")
        rows = cursor.fetchall()
        
        for row in rows:
            chunk, title, url = row
            self.chunks.append(chunk)
            self.chunk_metadata.append({
                'url': url,
                'title': title
            })
        
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(self.chunks)
    
    def initialize_knowledge_base(self):
        """Initialize or load knowledge base from database"""
        cursor = self.db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documentation_chunks")
        row = cursor.fetchone()
        
        if row[0] > 0:
            print("Knowledge base already exists in the database.")
            self.process_content()
        else:
            print("Building new knowledge base...")
            self.crawler.crawl(self.base_url)
            self.process_content()
        
        print(f"Knowledge base ready with {len(self.chunks)} chunks from the database")

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
        context = "\n\n".join([
            f"From {result['title']}:\n{result['content']}"
            for result in search_results if result['relevance_score'] > 0.1
        ])
        
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
    
    def save_chat_history(self):
        if not self.chat_history:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for entry in self.chat_history:
            question = entry['question']
            answer = self._serialize_answer(entry['answer'])
            timestamp_str = entry['timestamp']
            
            try:
                # Insert chat history into the database
                self.db_cursor.execute("""
                    INSERT INTO chat_history (question, answer, timestamp)
                    VALUES (%s, %s, %s)
                """, (question, answer, timestamp_str))
                self.db_conn.commit()
                print(f"Chat history saved for question: {question}")
            except Exception as e:
                print(f"Error saving chat history to database: {str(e)}")
                self.db_conn.rollback()
        
    def _serialize_answer(self, answer):
        """Convert complex answer object (like CrewOutput) into a serializable string."""
        if isinstance(answer, str):
            return answer
        try:
            # Try to serialize as JSON if it's a complex object
            return json.dumps(answer.__dict__)  # Use the dictionary representation of the object
        except Exception as e:
            # Fallback to string conversion if serialization fails
            print(f"Error serializing answer: {str(e)}")
            return str(answer)
    
    def save_knowledge_base(self, filename: str):
        """Save knowledge base in the database instead of JSON."""
        try:
            for idx, chunk in enumerate(self.chunks):
                title = self.chunk_metadata[idx]['title']
                url = self.chunk_metadata[idx]['url']
                
                # Insert chunk data into the database
                self.db_cursor.execute("""
                    INSERT INTO knowledge_base (title, url, chunk)
                    VALUES (%s, %s, %s)
                """, (title, url, chunk))
            self.db_conn.commit()
            print(f"Knowledge base saved to database.")
        except Exception as e:
            print(f"Error saving knowledge base to database: {str(e)}")
            self.db_conn.rollback()
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
                
                results = self.search(query)
                
                if not any(result['relevance_score'] > 0.1 for result in results):
                    print("\nI couldn't find any relevant information in the documentation. Could you please rephrase your question or ask about a different topic?")
                    continue
                
                response = self.generate_response(query, results)
                
                self.chat_history.append({
                    'question': query,
                    'answer': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                print("\nAssistant:", response)
                print("\nSources:")
                for result in results[:3]:
                    if result['relevance_score'] > 0.1:
                        print(f"- {result['title']}: {result['url']}")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving chat history...")
                self.save_chat_history()
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again with a different question.")
                
    def close_db_connection(self):
        self.db_cursor.close()
        self.db_conn.close()

def main():
    # Establish database connection
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USERNAME,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    # Get documentation URL from user
    default_url = "https://thecatapi.com/"
    url = input(f"Enter documentation URL (press Enter for default: {default_url}): ").strip()
    url = url if url else default_url
    
    chatbot = DocumentationChatbot(url, conn)
    print("\nPreparing knowledge base...")
    chatbot.initialize_knowledge_base()
    chatbot.chat_loop()

    conn.close()

if __name__ == "__main__":
    main()
