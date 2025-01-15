import psycopg2  # This works with psycopg2-binary as well
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Database Manager
class DatabaseManager:
    def __init__(self):
        self.db_config = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USERNAME'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        self.connection = psycopg2.connect(**self.db_config)
        self.cursor = self.connection.cursor()
        self.create_tables()

    def create_tables(self):
        # Create table for documentation content
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS documentation_content (
            id SERIAL PRIMARY KEY,
            url TEXT,
            title TEXT,
            content TEXT
        )''')
        
        # Create table for vector data
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS vector_data (
            id SERIAL PRIMARY KEY,
            url TEXT,
            chunk_id INT,
            vector FLOAT8[]
        )''')
        
        # Create table for chat history
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS chat_history (
            id SERIAL PRIMARY KEY,
            question TEXT,
            answer TEXT,
            timestamp TEXT
        )''')
        
        # Commit the changes
        self.connection.commit()

    def insert_documentation_content(self, url, title, content):
        self.cursor.execute('''INSERT INTO documentation_content (url, title, content)
                               VALUES (%s, %s, %s) RETURNING id''', (url, title, content))
        self.connection.commit()
        return self.cursor.fetchone()[0]

    def insert_vector_data(self, url, chunk_id, vector):
        self.cursor.execute('''INSERT INTO vector_data (url, chunk_id, vector)
                               VALUES (%s, %s, %s)''', (url, chunk_id, vector))
        self.connection.commit()

    def insert_chat_history(self, question, answer, timestamp):
        self.cursor.execute('''INSERT INTO chat_history (question, answer, timestamp)
                               VALUES (%s, %s, %s)''', (question, answer, timestamp))
        self.connection.commit()

    def close(self):
        self.connection.close()

# Documentation Crawler
class DocumentationCrawler:
    def __init__(self, base_url: str, db_manager: DatabaseManager):
        self.base_url = base_url
        self.visited_urls = set()
        self.db_manager = db_manager
        
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
            
            # Save the crawled content to the database
            title = soup.title.string if soup.title else url
            content_id = self.db_manager.insert_documentation_content(url, title, content)
            
            # Divide content into chunks (optional)
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            
            # Vectorize the content and save vectors to the database
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(chunks).toarray()
            
            for idx, chunk in enumerate(chunks):
                self.db_manager.insert_vector_data(url, idx, vectors[idx].tolist())
                
            for link in absolute_links:
                self.crawl(link)
                
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")

# Documentation Chatbot
class DocumentationChatbot:
    def __init__(self, base_url: str, db_manager: DatabaseManager):
        self.model="ollama/llama3.2:1b",
        self.base_url="http://localhost:11434",
        self.crawler = DocumentationCrawler(base_url, db_manager)
        self.vectorizer = None
        self.vectors = None
        self.chunks = []
        self.chunk_metadata = []
        self.chat_history = []
        self.db_manager = db_manager
        
    def save_chat_history(self):
        if not self.chat_history:
            return
        
        for entry in self.chat_history:
            question = entry['question']
            answer = entry['answer']
            timestamp = entry['timestamp']
            
            # Save to the database
            self.db_manager.insert_chat_history(question, answer, timestamp)
        
        print(f"\nChat history saved to database.")
    
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
                
                # Example response to the query (search and generate response implementation here)
                response = "Sample response for the query."
                
                # Store in chat history
                self.chat_history.append({
                    'question': query,
                    'answer': response,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Print response with source references
                print("\nAssistant:", response)
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving chat history...")
                self.save_chat_history()
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again with a different question.")

def main():
    # Initialize the database manager
    db_manager = DatabaseManager()
    
    # Get documentation URL from user
    default_url = "https://thecatapi.com/"
    url = input(f"Enter documentation URL (press Enter for default: {default_url}): ").strip()
    url = url if url else default_url
    
    # Initialize chatbot
    chatbot = DocumentationChatbot(url, db_manager)
    
    # Initialize knowledge base
    print("\nPreparing knowledge base...")
    chatbot.crawler.crawl(url)
    
    # Start chat loop
    chatbot.chat_loop()

    # Close the database connection when done
    db_manager.close()

if __name__ == "__main__":
    main()