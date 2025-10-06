from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import requests
import re
import time
from bs4 import BeautifulSoup
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import spacy
import numpy as np

app = Flask(__name__)
CORS(app)

CSV_FILE = "SB_publication_PMC.csv"

def fetch_abstract_from_html_page(pmc_url):
    """
    Fetch abstract directly from the publication HTML page
    """
    try:
        match = re.search(r"PMC\d+", pmc_url)
        if not match:
            return {
                "abstract": "No PMC ID found",
                "authors": "Unknown",
                "year": "Unknown",
                "keywords": []
            }
        
        pmc_id = match.group(0)
        pub_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        print(f"Fetching publication page: {pub_url}")
        response = requests.get(pub_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        abstract = extract_abstract_from_pmc_html(soup)
        authors = extract_authors_from_pmc_html(soup)
        year = extract_year_from_pmc_html(soup, pmc_id)
        keywords = extract_keywords_from_pmc_html(soup)
        
        return {
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "keywords": keywords
        }
        
    except Exception as e:
        print(f"Error fetching publication page for {pmc_url}: {str(e)}")
        return {
            "abstract": f"Error: {str(e)}",
            "authors": "Unknown",
            "year": "Unknown",
            "keywords": []
        }

def extract_keywords_from_pmc_html(soup):
    """
    Extract keywords from PMC HTML content
    """
    keywords = []
    
    try:
        # Look for keywords in meta tags
        keyword_meta_tags = [
            'meta[name="citation_keywords"]',
            'meta[name="keywords"]',
            'meta[name="DC.Subject"]',
            'meta[property="article:tag"]',
        ]
        
        for selector in keyword_meta_tags:
            meta_tags = soup.select(selector)
            for tag in meta_tags:
                content = tag.get('content', '')
                if content:
                    # Split by common delimiters
                    kw_list = re.split(r'[;,|]', content)
                    keywords.extend([kw.strip() for kw in kw_list if kw.strip()])
        
        # Look for keyword sections in the page
        keyword_sections = soup.find_all(['div', 'section', 'p'], 
                                        class_=re.compile(r'keyword', re.I))
        
        for section in keyword_sections:
            text = section.get_text()
            # Remove the "Keywords:" label
            text = re.sub(r'^keywords?\s*[:\-]?\s*', '', text, flags=re.IGNORECASE)
            kw_list = re.split(r'[;,|]', text)
            keywords.extend([kw.strip() for kw in kw_list if kw.strip() and len(kw.strip()) > 2])
        
        # Look for keyword headers
        keyword_headers = soup.find_all(['h2', 'h3', 'h4', 'strong', 'b'], 
                                       string=re.compile(r'keywords?', re.I))
        
        for header in keyword_headers:
            next_elem = header.find_next_sibling()
            if next_elem:
                text = next_elem.get_text()
                kw_list = re.split(r'[;,|]', text)
                keywords.extend([kw.strip() for kw in kw_list if kw.strip() and len(kw.strip()) > 2])
        
        # Remove duplicates and clean
        keywords = list(set([kw for kw in keywords if len(kw) < 100]))
        
        return keywords[:20]  # Limit to 20 keywords
        
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return []

def extract_abstract_from_pmc_html(soup):
    """
    Extract abstract from PMC HTML content with specific PMC selectors
    """
    try:
        abstract_selectors = [
            'div.abstract',
            'section.abstract',
            'div.abstract-section',
            'div#abstract',
            'div.sec',
            'div.abstract-sec',
            'div[class*="abstract"]',
            'section[class*="abstract"]',
            'div.abstract-content',
            'div.abstract-inner',
            'div.abstract-text',
            'div.abstractp',
            'p.abstract',
            'div.tsec.abstract',
            'div.sec.abstract',
        ]
        
        for selector in abstract_selectors:
            abstract_elems = soup.select(selector)
            for elem in abstract_elems:
                abstract_text = elem.get_text(strip=True)
                if abstract_text and len(abstract_text) > 100:
                    cleaned = clean_abstract_text(abstract_text)
                    if cleaned:
                        return cleaned
        
        abstract_headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5'], 
                                        string=re.compile(r'abstract|summary', re.I))
        
        for header in abstract_headers:
            next_elem = header.find_next_sibling()
            abstract_text = ""
            
            while next_elem and next_elem.name not in ['h1', 'h2', 'h3', 'h4', 'h5']:
                if next_elem.name in ['p', 'div']:
                    text = next_elem.get_text(strip=True)
                    if text:
                        abstract_text += " " + text
                next_elem = next_elem.find_next_sibling()
            
            if abstract_text and len(abstract_text) > 100:
                cleaned = clean_abstract_text(abstract_text)
                if cleaned:
                    return cleaned
        
        abstract_meta = soup.find('meta', {'name': 'citation_abstract'})
        if abstract_meta and abstract_meta.get('content'):
            content = abstract_meta['content']
            if len(content) > 100:
                return clean_abstract_text(content)
        
        dc_abstract = soup.find('meta', {'name': 'DC.Description'})
        if dc_abstract and dc_abstract.get('content'):
            content = dc_abstract['content']
            if len(content) > 100:
                return clean_abstract_text(content)
        
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            p_text = p.get_text(strip=True)
            if (len(p_text) > 200 and 
                not any(word in p_text.lower() for word in ['method', 'result', 'conclusion', 'introduction', 'reference']) and
                any(word in p_text.lower() for word in ['background', 'objective', 'purpose', 'aim', 'study', 'abstract'])):
                return clean_abstract_text(p_text[:2500])
        
        return "Abstract not found on publication page"
        
    except Exception as e:
        print(f"Error extracting abstract from HTML: {str(e)}")
        return f"Error extracting abstract: {str(e)}"

def extract_authors_from_pmc_html(soup):
    """
    Extract authors from PMC HTML content
    """
    try:
        author_selectors = [
            'div.authors',
            'div.auths',
            'div.author-list',
            'div.contributors',
            'span.authors',
            'div[class*="author"]',
            'span[class*="author"]',
        ]
        
        for selector in author_selectors:
            author_elem = soup.select_one(selector)
            if author_elem:
                author_text = author_elem.get_text(strip=True)
                if author_text and len(author_text) > 3:
                    return clean_text(author_text)
        
        author_meta = soup.find('meta', {'name': 'citation_authors'})
        if author_meta and author_meta.get('content'):
            return clean_text(author_meta['content'])
        
        dc_creator = soup.find('meta', {'name': 'DC.Creator'})
        if dc_creator and dc_creator.get('content'):
            return clean_text(dc_creator['content'])
        
        page_text = soup.get_text()
        
        author_patterns = [
            r'by\s+([^\.\n]+?)(?=\.|\n|Abstract|ABSTRACT)',
            r'authors?[:\s]+([^\.\n]+?)(?=\.|\n|Abstract|ABSTRACT)',
            r'^([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE | re.MULTILINE)
            if match:
                authors = match.group(1).strip()
                if len(authors) > 3 and len(authors) < 500:
                    return clean_text(authors)
        
        return "Authors not found on publication page"
        
    except Exception as e:
        print(f"Error extracting authors from HTML: {str(e)}")
        return "Error extracting authors"

def extract_year_from_pmc_html(soup, pmc_id):
    """
    Extract year from PMC HTML content
    """
    try:
        date_selectors = [
            'meta[name="citation_publication_date"]',
            'meta[name="citation_date"]',
            'meta[name="DC.Date"]',
            'meta[name="article:published_time"]',
            'time[pubdate]',
        ]
        
        for selector in date_selectors:
            date_elem = soup.select_one(selector)
            if date_elem and date_elem.get('content'):
                date_str = date_elem['content']
                year_match = re.search(r'\d{4}', date_str)
                if year_match:
                    return year_match.group(0)
        
        page_text = soup.get_text()
        year_patterns = [
            r'©\s*\d{4}',
            r'copyright\s*\d{4}',
            r'published.*?(\d{4})',
            r'\((\d{4})\)',
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, page_text, re.IGNORECASE)
            if match:
                year_match = re.search(r'\d{4}', match.group(0))
                if year_match:
                    return year_match.group(0)
        
        return extract_year_from_pmc_api(pmc_id)
        
    except Exception as e:
        print(f"Error extracting year from HTML: {str(e)}")
        return extract_year_from_pmc_api(pmc_id)

def extract_year_from_pmc_api(pmc_id):
    """
    Fallback method to get year from Europe PMC API
    """
    try:
        api_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmc_id}%20AND%20SRC:PMC&format=json"
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "resultList" in data and data["resultList"]["result"]:
                result = data["resultList"]["result"][0]
                return result.get("pubYear", "Year not found")
        return "Year not found"
    except:
        return "Year not found"

def clean_abstract_text(text):
    """
    Clean and normalize abstract text
    """
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^(abstract|summary|background|objective|purpose)\s*[:\-\s]*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    
    if len(text) > 2500:
        text = text[:2500] + "..."
    
    return text

def clean_text(text):
    """
    Clean and normalize general text
    """
    if not text:
        return text
    
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Initialize AI models (lazy loading)
summarizer = None
nlp = None

def get_summarizer():
    global summarizer
    if summarizer is None:
        print("Loading summarization model (this may take a moment)...")
        try:
            # Try the smaller, faster model first
            summarizer = pipeline(
                "summarization", 
                model="sshleifer/distilbart-cnn-12-6",
                device=-1  # CPU
            )
            print("✓ Summarization model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to basic summarization...")
            summarizer = None
    return summarizer

def get_nlp():
    global nlp
    if nlp is None:
        print("Loading spaCy model...")
        nlp = spacy.load("en_core_web_sm")
    return nlp
def summarize_text(text, max_length=130, min_length=30):
    """Summarize text using distilbart"""
    try:
        if len(text) < 100:
            return text
        
        text = text.strip()
        
        if len(text) < 200:
            return text[:150] + "..." if len(text) > 150 else text
        
        summarizer = get_summarizer()
        
        # If model failed to load, use extractive summarization
        if summarizer is None:
            sentences = text.split('. ')
            # Return first 2-3 sentences as summary
            summary_sentences = sentences[:3]
            result = '. '.join(summary_sentences)
            if not result.endswith('.'):
                result += '.'
            return result
        
        input_text = text[:1024]
        estimated_tokens = len(input_text.split())
        adjusted_max = min(max_length, int(estimated_tokens * 0.4))
        adjusted_min = min(min_length, adjusted_max - 20)
        
        if adjusted_max <= adjusted_min:
            adjusted_max = adjusted_min + 20
        
        summary = summarizer(
            input_text, 
            max_length=adjusted_max, 
            min_length=adjusted_min, 
            do_sample=False,
            truncation=True
        )
        
        result = summary[0]['summary_text']
        
        # Verify it's actually a summary (not just a copy)
        if len(result) > len(text) * 0.8 or result == text[:len(result)]:
            # Fallback to extractive summary
            sentences = text.split('. ')
            summary_sentences = sentences[:2]
            result = '. '.join(summary_sentences)
            if not result.endswith('.'):
                result += '.'
        
        return result
        
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        sentences = text.split('. ')
        summary_sentences = sentences[:2]
        result = '. '.join(summary_sentences)
        if not result.endswith('.'):
            result += '.'
        return result if len(result) > 50 else text[:200] + "..."

def extract_entities(text):
    """Extract entities using spaCy with improved categorization"""
    try:
        nlp_model = get_nlp()
        doc = nlp_model(text[:10000])
        
        entities = {
            "organisms": [],
            "conditions": [],
            "outcomes": [],
            "other": []
        }
        
        # More specific keyword lists
        organism_keywords = {
            'mice', 'mouse', 'rat', 'rats', 'human', 'humans', 'astronaut', 'astronauts',
            'plant', 'plants', 'cell', 'cells', 'tissue', 'tissues', 'animal', 'animals',
            'bacteria', 'virus', 'fungi', 'organism', 'species', 'embryo', 'fetus',
            'mammal', 'rodent', 'primate', 'drosophila', 'c. elegans', 'zebrafish',
            'arabidopsis', 'yeast', 'e. coli', 'stem cell', 'bone cell', 'muscle cell'
        }
        
        condition_keywords = {
            'microgravity', 'radiation', 'space', 'spaceflight', 'flight', 'exposure',
            'environment', 'simulated', 'cosmic', 'hypergravity', 'weightlessness',
            'hypoxia', 'isolation', 'confinement', 'vacuum', 'altitude', 'zero gravity',
            'partial gravity', 'mars gravity', 'lunar gravity'
        }
        
        outcome_keywords = {
            'loss', 'increase', 'decrease', 'reduction', 'enhancement', 'change', 'changes',
            'effect', 'effects', 'response', 'adaptation', 'atrophy', 'degradation',
            'deterioration', 'improvement', 'damage', 'recovery', 'alteration',
            'dysfunction', 'impairment', 'suppression', 'activation', 'inhibition'
        }
        
        # Exclude common false positives
        exclude_terms = {
            'study', 'research', 'experiment', 'data', 'analysis', 'results',
            'methods', 'conclusions', 'background', 'objective', 'purpose',
            'group', 'control', 'test', 'sample', 'NASA', 'ISS', 'PMC'
        }
        
        for ent in doc.ents:
            ent_text = ent.text.strip()
            ent_text_lower = ent_text.lower()
            ent_label = ent.label_
            
            # Skip excluded terms
            if ent_text_lower in exclude_terms or ent_text_lower in [e.lower() for e in entities["organisms"] + entities["conditions"] + entities["outcomes"]]:
                continue
            
            # Skip very short or very long entities
            if len(ent_text) < 3 or len(ent_text) > 50:
                continue
            
            # Categorize organisms - must match specific criteria
            is_organism = False
            if ent_label in ['PERSON', 'ORG', 'NORP']:
                # Only if it's a known organism keyword
                for kw in organism_keywords:
                    if kw in ent_text_lower:
                        is_organism = True
                        break
            elif any(kw == ent_text_lower or kw in ent_text_lower.split() for kw in organism_keywords):
                is_organism = True
            
            if is_organism and ent_text not in entities["organisms"]:
                entities["organisms"].append(ent_text)
                continue
            
            # Categorize conditions
            is_condition = False
            for kw in condition_keywords:
                if kw in ent_text_lower or ent_text_lower in kw:
                    is_condition = True
                    break
            
            if is_condition and ent_text not in entities["conditions"]:
                entities["conditions"].append(ent_text)
                continue
            
            # Categorize outcomes
            is_outcome = False
            for kw in outcome_keywords:
                if kw in ent_text_lower:
                    is_outcome = True
                    break
            
            if is_outcome and ent_text not in entities["outcomes"]:
                entities["outcomes"].append(ent_text)
                continue
            
            # Everything else goes to "other" if it's a relevant entity type
            if ent_label in ['GPE', 'LOC', 'PRODUCT', 'EVENT', 'LAW'] and ent_text not in entities["other"]:
                entities["other"].append(ent_text)
        
        # Limit results
        for key in entities:
            entities[key] = entities[key][:10]
        
        return entities
        
    except Exception as e:
        print(f"Entity extraction error: {str(e)}")
        return {"organisms": [], "conditions": [], "outcomes": [], "other": []}

def perform_topic_modeling(abstracts, n_clusters=6):
    """Perform topic modeling using TF-IDF + KMeans"""
    try:
        if len(abstracts) < n_clusters:
            n_clusters = max(2, len(abstracts) // 2)
        
        # Filter valid abstracts
        valid_abstracts = [abs for abs in abstracts if abs and len(abs) > 50 and 'error' not in abs.lower()]
        
        if len(valid_abstracts) < n_clusters:
            return {"error": "Not enough valid abstracts for clustering"}
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(valid_abstracts)
        
        # KMeans Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get top terms per cluster
        feature_names = vectorizer.get_feature_names_out()
        cluster_info = []
        
        for i in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[i]
            top_indices = cluster_center.argsort()[-10:][::-1]
            top_terms = [feature_names[idx] for idx in top_indices]
            
            # Get papers in this cluster
            cluster_papers = [idx for idx, c in enumerate(clusters) if c == i]
            
            # Generate cluster title from top terms
            cluster_title = f"Topic {i+1}: {', '.join(top_terms[:3])}"
            
            cluster_info.append({
                "cluster_id": i,
                "title": cluster_title,
                "top_terms": top_terms,
                "paper_count": len(cluster_papers),
                "paper_indices": cluster_papers[:20]  # Limit to 20 papers per cluster
            })
        
        return {
            "clusters": cluster_info,
            "total_papers": len(valid_abstracts)
        }
        
    except Exception as e:
        print(f"Topic modeling error: {str(e)}")
        return {"error": str(e)}

@app.route("/papers")
def get_papers():
    try:
        count = request.args.get('count', default=None, type=int)
        
        try:
            df = pd.read_csv(CSV_FILE)
        except FileNotFoundError:
            return jsonify({"error": f"CSV file {CSV_FILE} not found"}), 404
        
        if count and count > 0:
            df = df.head(count)
        
        total = len(df)
        papers = []

        for i, row in df.iterrows():
            title = row.get("Title", "No title available")
            link = row.get("Link", "")
            
            print(f"Processing paper {i+1}/{total}: {title[:60]}...")
            
            meta = fetch_abstract_from_html_page(link)

            papers.append({
                "title": title,
                "abstract": meta["abstract"],
                "authors": meta["authors"],
                "year": meta["year"],
                "keywords": meta["keywords"],
                "link": link,
                "id": i + 1
            })

            time.sleep(0.5)

        return jsonify({
            "total_papers": total,
            "papers": papers,
            "status": "success"
        })
    
    except Exception as e:
        print(f"Error in /papers endpoint: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/papers/list")
def get_papers_list():
    """Get paginated list of papers with basic info (no fetching from web)"""
    try:
        page = request.args.get('page', default=1, type=int)
        per_page = request.args.get('per_page', default=20, type=int)
        search = request.args.get('search', default='', type=str).lower()
        
        df = pd.read_csv(CSV_FILE)
        
        # Filter by search term in title
        if search:
            df = df[df['Title'].str.lower().str.contains(search, na=False)]
        
        total = len(df)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        df_page = df.iloc[start_idx:end_idx]
        
        papers = []
        for i, row in df_page.iterrows():
            papers.append({
                "id": i + 1,
                "title": row.get("Title", "No title available"),
                "link": row.get("Link", ""),
            })
        
        total_pages = (total + per_page - 1) // per_page
        
        return jsonify({
            "papers": papers,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "status": "success"
        })
    
    except Exception as e:
        print(f"Error in /papers/list endpoint: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/papers/<int:paper_id>")
def get_paper_details(paper_id):
    """Get detailed information for a specific paper"""
    try:
        df = pd.read_csv(CSV_FILE)
        
        if paper_id < 1 or paper_id > len(df):
            return jsonify({"error": "Paper not found"}), 404
        
        row = df.iloc[paper_id - 1]
        title = row.get("Title", "No title available")
        link = row.get("Link", "")
        
        print(f"Fetching details for paper {paper_id}: {title[:60]}...")
        
        meta = fetch_abstract_from_html_page(link)
        
        paper = {
            "id": paper_id,
            "title": title,
            "abstract": meta["abstract"],
            "authors": meta["authors"],
            "year": meta["year"],
            "keywords": meta["keywords"],
            "link": link
        }
        
        return jsonify({
            "paper": paper,
            "status": "success"
        })
    
    except Exception as e:
        print(f"Error fetching paper details: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/papers/search")
def search_papers():
    """Search papers by title and fetch keywords from matching papers"""
    try:
        query = request.args.get('q', default='', type=str).lower()
        page = request.args.get('page', default=1, type=int)
        per_page = request.args.get('per_page', default=20, type=int)
        
        if not query:
            return jsonify({"error": "Search query required"}), 400
        
        df = pd.read_csv(CSV_FILE)
        
        # Filter by title
        matching_indices = df[df['Title'].str.lower().str.contains(query, na=False)].index.tolist()
        
        total_matches = len(matching_indices)
        
        # Paginate
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_indices = matching_indices[start_idx:end_idx]
        
        papers = []
        for idx in page_indices:
            row = df.iloc[idx]
            title = row.get("Title", "No title available")
            link = row.get("Link", "")
            
            # Fetch keywords for each matching paper
            print(f"Fetching keywords for: {title[:60]}...")
            meta = fetch_abstract_from_html_page(link)
            
            papers.append({
                "id": idx + 1,
                "title": title,
                "link": link,
                "keywords": meta["keywords"],
                "year": meta["year"],
                "authors": meta["authors"]
            })
            
            time.sleep(0.5)
        
        total_pages = (total_matches + per_page - 1) // per_page
        
        return jsonify({
            "papers": papers,
            "total": total_matches,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "query": query,
            "status": "success"
        })
    
    except Exception as e:
        print(f"Error in search: {str(e)}")
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/papers/count")
def get_paper_count():
    """Return the total number of papers available"""
    try:
        df = pd.read_csv(CSV_FILE)
        total_count = len(df)
        return jsonify({"total_papers": total_count, "status": "success"})
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500
    
@app.route("/api/summarize/<int:paper_id>")
def summarize_paper(paper_id):
    """Summarize a specific paper's abstract"""
    try:
        df = pd.read_csv(CSV_FILE)
        
        if paper_id < 1 or paper_id > len(df):
            return jsonify({"error": "Paper not found"}), 404
        
        row = df.iloc[paper_id - 1]
        link = row.get("Link", "")
        
        meta = fetch_abstract_from_html_page(link)
        abstract = meta["abstract"]
        
        if "error" in abstract.lower() or len(abstract) < 100:
            return jsonify({"error": "Cannot summarize - abstract not available"}), 400
        
        summary = summarize_text(abstract)
        
        return jsonify({
            "paper_id": paper_id,
            "original_abstract": abstract,
            "summary": summary,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error summarizing paper: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/entities/<int:paper_id>")
def extract_paper_entities(paper_id):
    """Extract entities from a specific paper"""
    try:
        df = pd.read_csv(CSV_FILE)
        
        if paper_id < 1 or paper_id > len(df):
            return jsonify({"error": "Paper not found"}), 404
        
        row = df.iloc[paper_id - 1]
        title = row.get("Title", "")
        link = row.get("Link", "")
        
        meta = fetch_abstract_from_html_page(link)
        abstract = meta["abstract"]
        
        # Extract from both title and abstract
        combined_text = f"{title}. {abstract}"
        entities = extract_entities(combined_text)
        
        return jsonify({
            "paper_id": paper_id,
            "title": title,
            "entities": entities,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error extracting entities: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/topics")
def get_topics():
    """Perform topic modeling on all papers"""
    try:
        n_clusters = request.args.get('clusters', default=6, type=int)
        max_papers = request.args.get('max_papers', default=100, type=int)
        
        df = pd.read_csv(CSV_FILE)
        df_sample = df.head(max_papers)
        
        abstracts = []
        paper_info = []
        
        print(f"Fetching {len(df_sample)} papers for topic modeling...")
        
        for i, row in df_sample.iterrows():
            title = row.get("Title", "")
            link = row.get("Link", "")
            
            meta = fetch_abstract_from_html_page(link)
            abstracts.append(meta["abstract"])
            paper_info.append({
                "id": i + 1,
                "title": title,
                "link": link
            })
            
            time.sleep(0.3)  # Rate limiting
        
        # Perform clustering
        result = perform_topic_modeling(abstracts, n_clusters)
        
        if "error" in result:
            return jsonify(result), 400
        
        # Add paper details to clusters
        for cluster in result["clusters"]:
            cluster["papers"] = [paper_info[idx] for idx in cluster["paper_indices"]]
            
            # Summarize cluster (use abstracts of first 3 papers)
            cluster_abstracts = [abstracts[idx] for idx in cluster["paper_indices"][:3]]
            combined = " ".join(cluster_abstracts)[:1000]
            cluster["summary"] = summarize_text(combined, max_length=100, min_length=30)
        
        return jsonify({
            "result": result,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in topic modeling: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/batch-analyze")
def batch_analyze():
    """Analyze multiple papers at once with summaries and entities"""
    try:
        paper_ids = request.args.get('ids', '')
        
        if not paper_ids:
            return jsonify({"error": "No paper IDs provided"}), 400
        
        ids = [int(id.strip()) for id in paper_ids.split(',')]
        df = pd.read_csv(CSV_FILE)
        
        results = []
        
        for paper_id in ids:
            if paper_id < 1 or paper_id > len(df):
                continue
            
            row = df.iloc[paper_id - 1]
            title = row.get("Title", "")
            link = row.get("Link", "")
            
            meta = fetch_abstract_from_html_page(link)
            abstract = meta["abstract"]
            
            # Summarize
            summary = summarize_text(abstract) if len(abstract) > 100 else abstract
            
            # Extract entities
            entities = extract_entities(f"{title}. {abstract}")
            
            results.append({
                "id": paper_id,
                "title": title,
                "summary": summary,
                "entities": entities,
                "keywords": meta["keywords"]
            })
            
            time.sleep(0.3)
        
        return jsonify({
            "papers": results,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error in batch analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    """Serve the main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Papers Viewer</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif; background: #f0f2f5; }
            
            /* Navigation Tabs */
            .tabs-container { background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); position: sticky; top: 0; z-index: 100; }
            .tabs { display: flex; max-width: 1200px; margin: 0 auto; }
            .tab { flex: 1; padding: 20px; text-align: center; cursor: pointer; border-bottom: 3px solid transparent; transition: all 0.3s; font-weight: 600; color: #666; }
            .tab:hover { background: #f8f9fa; color: #333; }
            .tab.active { color: #007bff; border-bottom-color: #007bff; background: #f8f9fa; }
            
            /* Container */
            .container { max-width: 1200px; margin: 0 auto; padding: 30px 20px; }
            .tab-content { display: none; }
            .tab-content.active { display: block; animation: fadeIn 0.3s; }
            
            @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
            
            h1 { color: #333; text-align: center; margin-bottom: 30px; font-size: 2em; }
            
            /* Fetch Papers Tab */
            .controls { display: flex; justify-content: center; gap: 10px; margin-bottom: 30px; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); align-items: center; }
            input[type="number"] { padding: 12px; border: 2px solid #ddd; border-radius: 5px; width: 120px; font-size: 16px; }
            button { padding: 12px 24px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-weight: bold; font-size: 16px; transition: background 0.3s; }
            button:hover { background: #0056b3; }
            button:disabled { background: #6c757d; cursor: not-allowed; }
            
            /* Search Bar */
            .search-container { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin-bottom: 30px; }
            .search-box { display: flex; gap: 10px; }
            .search-box input { flex: 1; padding: 14px; border: 2px solid #ddd; border-radius: 5px; font-size: 16px; }
            .search-box button { padding: 14px 28px; }
            
            /* Papers List */
            .papers-grid { display: grid; gap: 20px; }
            .paper-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); transition: transform 0.2s, box-shadow 0.2s; cursor: pointer; }
            .paper-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
            .paper-card-title { font-size: 1.2em; font-weight: bold; color: #2c3e50; margin-bottom: 10px; line-height: 1.4; }
            .paper-card-meta { color: #7f8c8d; font-size: 0.9em; }
            .paper-card-keywords { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 5px; }
            .keyword-tag { background: #e3f2fd; color: #1976d2; padding: 4px 10px; border-radius: 12px; font-size: 0.85em; }
            
            /* Paper Details Modal */
            .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 1000; overflow-y: auto; padding: 20px; }
            .modal.active { display: flex; align-items: center; justify-content: center; }
            .modal-content { background: white; padding: 30px; border-radius: 10px; max-width: 900px; width: 100%; max-height: 90vh; overflow-y: auto; position: relative; }
            .modal-close { position: absolute; top: 15px; right: 20px; font-size: 30px; cursor: pointer; color: #999; }
            .modal-close:hover { color: #333; }
            .modal-title { font-size: 1.5em; font-weight: bold; color: #2c3e50; margin-bottom: 15px; }
            .modal-section { margin: 20px 0; }
            .modal-section h3 { color: #555; margin-bottom: 10px; font-size: 1.1em; }
            .modal-abstract { line-height: 1.7; background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
            .modal-link { display: inline-block; margin-top: 15px; padding: 10px 20px; background: #28a745; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; }
            .modal-link:hover { background: #218838; }
            
            /* Pagination */
            .pagination { display: flex; justify-content: center; gap: 10px; margin-top: 30px; flex-wrap: wrap; }
            .pagination button { padding: 10px 15px; min-width: 40px; }
            .pagination button.active { background: #28a745; }
            
            /* Loading & Error */
            .loading { text-align: center; color: #666; font-size: 1.2em; padding: 40px; background: white; border-radius: 10px; }
            .error { color: #dc3545; background: #f8d7da; padding: 20px; border-radius: 8px; margin: 15px 0; text-align: center; }
            .stats { text-align: center; margin-bottom: 20px; color: #666; font-size: 1.1em; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .note { text-align: center; color: #856404; background: #fff3cd; padding: 15px; border-radius: 8px; margin: 15px 0; border: 1px solid #ffeaa7; }
            
            .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #007bff; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

            /* Dark mode variables */
            :root {
                --bg-primary: #f0f2f5;
                --bg-secondary: white;
                --bg-tertiary: #f8f9fa;
                --text-primary: #333;
                --text-secondary: #666;
                --text-tertiary: #7f8c8d;
                --border-color: #ddd;
                --shadow: rgba(0,0,0,0.1);
                --shadow-hover: rgba(0,0,0,0.15);
            }

            [data-theme="dark"] {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --bg-tertiary: #3a3a3a;
                --text-primary: #e0e0e0;
                --text-secondary: #b0b0b0;
                --text-tertiary: #888;
                --border-color: #444;
                --shadow: rgba(0,0,0,0.3);
                --shadow-hover: rgba(0,0,0,0.5);
            }

            /* Apply variables to elements */
            body { 
                background: var(--bg-primary); 
                color: var(--text-primary);
                transition: background 0.3s, color 0.3s;
            }

            .tabs-container { 
                background: var(--bg-secondary); 
                box-shadow: 0 2px 4px var(--shadow); 
            }

            .tab { 
                color: var(--text-secondary); 
                border-bottom: 3px solid transparent;
            }

            .tab:hover { 
                background: var(--bg-tertiary); 
                color: var(--text-primary); 
            }

            .tab.active { 
                color: #007bff; 
                background: var(--bg-tertiary); 
            }

            .controls, .search-container, .paper-card, .modal-content, .loading, .stats { 
                background: var(--bg-secondary); 
                box-shadow: 0 2px 8px var(--shadow); 
            }

            .paper-card:hover { 
                box-shadow: 0 4px 12px var(--shadow-hover); 
            }

            .paper-card-title, .modal-title { 
                color: var(--text-primary); 
            }

            .paper-card-meta { 
                color: var(--text-secondary); 
            }

            input[type="number"], .search-box input { 
                background: var(--bg-tertiary); 
                border: 2px solid var(--border-color); 
                color: var(--text-primary);
            }

            [data-theme="dark"] input[type="number"]:focus,
            [data-theme="dark"] .search-box input:focus {
                background: var(--bg-secondary);
                border-color: #007bff;
            }

            .modal { 
                background: rgba(0,0,0,0.7); 
            }

            .modal-abstract { 
                background: var(--bg-tertiary); 
                color: var(--text-primary);
            }

            .modal-close { 
                color: var(--text-secondary); 
            }

            .modal-close:hover { 
                color: var(--text-primary); 
            }

            /* Dark mode toggle button */
            .theme-toggle {
                position: fixed;
                bottom: 30px;
                right: 30px;
                width: 60px;
                height: 60px;
                border-radius: 50%;
                background: var(--bg-secondary);
                border: 2px solid var(--border-color);
                box-shadow: 0 4px 12px var(--shadow);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                transition: all 0.3s;
                z-index: 999;
            }

            .theme-toggle:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 16px var(--shadow-hover);
            }

            .theme-toggle:active {
                transform: scale(0.95);
            }

            /* Update existing selectors */
            h1 { 
                color: var(--text-primary); 
            }

            .note { 
                background: #fff3cd; 
                color: #856404; 
            }

            [data-theme="dark"] .note {
                background: #3a3a1a;
                color: #e0d090;
                border-color: #5a5a2a;
            }

            .error { 
                color: #dc3545; 
                background: #f8d7da; 
            }

            [data-theme="dark"] .error {
                background: #4a2a2a;
                color: #ff6b6b;
            }

            [data-theme="dark"] .keyword-tag {
                background: #1e3a5f;
                color: #64b5f6;
            }

            [data-theme="dark"] .pagination button {
                background: #007bff;
                color: white;
            }

            [data-theme="dark"] .pagination button:hover {
                background: #0056b3;
            }

            [data-theme="dark"] .pagination button.active {
                background: #28a745;
            }

            [data-theme="dark"] button:disabled {
                background: #555;
            }

            [data-theme="dark"] details summary {
                background: var(--bg-tertiary);
                color: var(--text-primary);
            }
        </style>
    </head>
    <body>
        <div class="tabs-container">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('fetch')">📥 Fetch Papers</div>
                <div class="tab" onclick="switchTab('browse')">📚 Browse & Search</div>
                <div class="tab" onclick="switchTab('ai')">🤖 AI Analysis</div>
            </div>
        </div>

        <div class="container">
            <!-- Fetch Papers Tab -->
            <div id="fetchTab" class="tab-content active">
                <h1>📚 Fetch Research Papers</h1>
                
                <div class="note">
                    <strong>🔍 PMC Abstract Fetch:</strong> This system reads abstracts and keywords directly from 
                    PMC publication pages using specialized HTML parsing for better accuracy.
                </div>
                
                <div class="controls">
                    <label for="paperCount"><strong>Number of papers:</strong></label>
                    <input type="number" id="paperCount" placeholder="e.g., 5" min="1" value="3" max="50">
                    <button onclick="fetchPapers()" id="getButton">📖 FETCH PAPERS</button>
                </div>
                
                <div id="stats" class="stats"></div>
                <div id="loading" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <div style="margin-top: 15px;">Fetching PMC Publication Data...</div>
                </div>
                <div id="error" class="error" style="display: none;"></div>
                <div id="papersContainer"></div>
            </div>

            <!-- Browse & Search Tab -->
            <div id="browseTab" class="tab-content">
                <h1>🔍 Browse & Search Papers</h1>
                
                <div class="search-container">
                    <div class="search-box">
                        <input type="text" id="searchInput" placeholder="Search by title or keywords..." onkeypress="if(event.key==='Enter') searchPapers()">
                        <button onclick="searchPapers()">🔍 SEARCH</button>
                        <button onclick="clearSearch()" style="background: #6c757d;">✖ CLEAR</button>
                    </div>
                </div>
                
                <div id="browseStats" class="stats"></div>
                <div id="browseLoading" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <div style="margin-top: 15px;">Loading papers...</div>
                </div>
                <div id="browseError" class="error" style="display: none;"></div>
                <div id="browseContainer" class="papers-grid"></div>
                <div id="browsePagination" class="pagination"></div>
            </div>

            <!-- AI Analysis Tab -->
            <div id="aiTab" class="tab-content">
                <h1>🤖 AI Text Mining & Analysis</h1>
                
                <div class="note">
                    <strong>⚡ Features:</strong> Automatic summarization, topic clustering, and entity extraction using lightweight NLP models.
                </div>
                
                <div style="display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-bottom: 30px;">
                    <div class="search-container">
                        <h3 style="margin-bottom: 15px;">📊 Topic Modeling</h3>
                        <p style="margin-bottom: 15px; color: #666;">Cluster papers by topics using TF-IDF + KMeans</p>
                        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 10px;">
                            <label>Clusters:</label>
                            <input type="number" id="clusterCount" value="6" min="2" max="10" style="width: 80px; padding: 8px;">
                        </div>
                        <div style="display: flex; gap: 10px; align-items: center; margin-bottom: 15px;">
                            <label>Papers:</label>
                            <input type="number" id="topicPaperCount" value="50" min="10" max="100" style="width: 80px; padding: 8px;">
                        </div>
                        <button onclick="runTopicModeling()" id="topicBtn" style="width: 100%;">🔍 Find Topics</button>
                    </div>
                    
                    <div class="search-container">
                        <h3 style="margin-bottom: 15px;">📝 Summarize Paper</h3>
                        <p style="margin-bottom: 15px; color: #666;">Get AI-generated summary of abstract</p>
                        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                            <input type="number" id="summarizePaperId" placeholder="Paper ID" min="1" style="flex: 1; padding: 8px;">
                            <button onclick="summarizePaper()" style="flex: 1;">✨ Summarize</button>
                        </div>
                    </div>
                    
                    <div class="search-container">
                        <h3 style="margin-bottom: 15px;">🏷️ Extract Entities</h3>
                        <p style="margin-bottom: 15px; color: #666;">Find organisms, conditions, outcomes</p>
                        <div style="display: flex; gap: 10px; margin-bottom: 15px;">
                            <input type="number" id="entitiesPaperId" placeholder="Paper ID" min="1" style="flex: 1; padding: 8px;">
                            <button onclick="extractEntities()" style="flex: 1;">🔬 Extract</button>
                        </div>
                    </div>
                </div>
                
                <div style="display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-bottom: 30px;">
                    <div class="search-container">
                        <h3 style="margin-bottom: 15px;">🔍 Batch Analysis</h3>
                        <p style="margin-bottom: 15px; color: #666;">Analyze multiple papers at once</p>
                        <div style="margin-bottom: 15px;">
                            <input type="text" id="batchPaperIds" placeholder="Paper IDs (e.g., 1,5,7,12)" style="width: 100%; padding: 8px; margin-bottom: 10px;">
                            <small style="color: #666;">Enter comma-separated paper IDs</small>
                        </div>
                        <button onclick="runBatchAnalysis()" id="batchBtn" style="width: 100%;">📊 Analyze Papers</button>
                    </div>
                </div>
                
                <div id="aiLoading" class="loading" style="display: none;">
                    <div class="spinner"></div>
                    <div style="margin-top: 15px;">Processing with AI models...</div>
                </div>
                <div id="aiError" class="error" style="display: none;"></div>
                <div id="aiResults"></div>
            </div>
        </div>

        <button class="theme-toggle" onclick="toggleTheme()" id="themeToggle" title="Toggle Dark Mode">
            🌙
        </button>

        <!-- Paper Details Modal -->
        <div id="paperModal" class="modal">
            <div class="modal-content">
                <span class="modal-close" onclick="closeModal()">&times;</span>
                <div id="modalContent">
                    <div class="spinner"></div>
                    <div style="text-align: center; margin-top: 15px;">Loading paper details...</div>
                </div>
            </div>
        </div>

        <script>
            let currentPage = 1;
            let currentSearch = '';
            let isSearchMode = false;

            // Tab Switching
            function switchTab(tabName) {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                if (tabName === 'fetch') {
                    document.querySelector('.tab:nth-child(1)').classList.add('active');
                    document.getElementById('fetchTab').classList.add('active');
                } else if (tabName === 'browse') {
                    document.querySelector('.tab:nth-child(2)').classList.add('active');
                    document.getElementById('browseTab').classList.add('active');
                    loadPapersList(1);
                } else if (tabName === 'ai') {
                    document.querySelector('.tab:nth-child(3)').classList.add('active');
                    document.getElementById('aiTab').classList.add('active');
                }
            }

            // Fetch Papers (Original Functionality)
            async function fetchPapers() {
                const countInput = document.getElementById('paperCount');
                const getButton = document.getElementById('getButton');
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                const papersContainer = document.getElementById('papersContainer');
                const stats = document.getElementById('stats');
                
                const count = countInput.value;
                
                if (!count || count < 1) {
                    showError('Please enter a valid number of papers (at least 1)', 'error');
                    return;
                }
                
                loading.style.display = 'block';
                error.style.display = 'none';
                papersContainer.innerHTML = '';
                stats.innerHTML = '';
                getButton.disabled = true;
                getButton.textContent = '🔄 PROCESSING...';
                
                try {
                    const response = await fetch(`/papers?count=${count}`);
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.error || `HTTP ${response.status}: Failed to fetch papers`);
                    }
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        displayFetchedPapers(data.papers);
                        stats.innerHTML = `✅ Showing ${data.papers.length} of ${data.total_papers} total papers`;
                        stats.style.background = '#d4edda';
                        stats.style.color = '#155724';
                    } else {
                        throw new Error(data.error || 'Unknown error occurred');
                    }
                    
                } catch (err) {
                    showError('Error fetching papers: ' + err.message, 'error');
                    console.error('Error:', err);
                } finally {
                    loading.style.display = 'none';
                    getButton.disabled = false;
                    getButton.textContent = '📖 FETCH PAPERS';
                }
            }
            
            function displayFetchedPapers(papers) {
                const container = document.getElementById('papersContainer');
                
                if (papers.length === 0) {
                    container.innerHTML = '<div class="error">No papers found.</div>';
                    return;
                }
                
                container.innerHTML = papers.map(paper => `
                    <div class="paper-card" style="cursor: default;">
                        <div class="paper-card-title">${paper.id}. ${paper.title}</div>
                        <div class="paper-card-meta">
                            <div style="margin: 8px 0;">👥 <strong>Authors:</strong> ${paper.authors}</div>
                            <div style="margin: 8px 0;">📅 <strong>Year:</strong> ${paper.year}</div>
                        </div>
                        ${paper.keywords && paper.keywords.length > 0 ? `
                            <div class="paper-card-keywords">
                                <strong style="margin-right: 10px;">🏷️ Keywords:</strong>
                                ${paper.keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
                            </div>
                        ` : ''}
                        <div class="modal-abstract" style="margin-top: 15px;">
                            <strong>📖 Abstract:</strong><br><br>${paper.abstract}
                        </div>
                        <div style="margin-top: 15px;">
                            <a href="${paper.link}" target="_blank" class="modal-link">🔗 View on PMC</a>
                        </div>
                    </div>
                `).join('');
            }

            // Browse & Search Functionality
            async function loadPapersList(page = 1) {
                const loading = document.getElementById('browseLoading');
                const error = document.getElementById('browseError');
                const container = document.getElementById('browseContainer');
                const stats = document.getElementById('browseStats');
                const pagination = document.getElementById('browsePagination');
                
                currentPage = page;
                
                loading.style.display = 'block';
                error.style.display = 'none';
                container.innerHTML = '';
                pagination.innerHTML = '';
                
                try {
                    const response = await fetch(`/papers/list?page=${page}&per_page=20`);
                    
                    if (!response.ok) {
                        throw new Error('Failed to load papers list');
                    }
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        displayPapersList(data.papers);
                        stats.innerHTML = `📊 Showing ${data.papers.length} of ${data.total} papers (Page ${data.page} of ${data.total_pages})`;
                        displayPagination(data.page, data.total_pages, false);
                    }
                    
                } catch (err) {
                    showError('Error loading papers: ' + err.message, 'browseError');
                    console.error('Error:', err);
                } finally {
                    loading.style.display = 'none';
                }
            }

            async function searchPapers(page = 1) {
                const searchInput = document.getElementById('searchInput');
                const query = searchInput.value.trim();
                
                if (!query) {
                    loadPapersList(1);
                    return;
                }
                
                const loading = document.getElementById('browseLoading');
                const error = document.getElementById('browseError');
                const container = document.getElementById('browseContainer');
                const stats = document.getElementById('browseStats');
                const pagination = document.getElementById('browsePagination');
                
                currentPage = page;
                currentSearch = query;
                isSearchMode = true;
                
                loading.style.display = 'block';
                error.style.display = 'none';
                container.innerHTML = '';
                pagination.innerHTML = '';
                
                try {
                    const response = await fetch(`/papers/search?q=${encodeURIComponent(query)}&page=${page}&per_page=20`);
                    
                    if (!response.ok) {
                        throw new Error('Search failed');
                    }
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        displaySearchResults(data.papers);
                        stats.innerHTML = `🔍 Found ${data.total} papers matching "${data.query}" (Page ${data.page} of ${data.total_pages})`;
                        displayPagination(data.page, data.total_pages, true);
                    }
                    
                } catch (err) {
                    showError('Search error: ' + err.message, 'browseError');
                    console.error('Error:', err);
                } finally {
                    loading.style.display = 'none';
                }
            }

            function clearSearch() {
                document.getElementById('searchInput').value = '';
                currentSearch = '';
                isSearchMode = false;
                loadPapersList(1);
            }

            function displayPapersList(papers) {
                const container = document.getElementById('browseContainer');
                
                if (papers.length === 0) {
                    container.innerHTML = '<div class="error">No papers found.</div>';
                    return;
                }
                
                container.innerHTML = papers.map(paper => `
                    <div class="paper-card" onclick="viewPaperDetails(${paper.id})">
                        <div class="paper-card-title">${paper.id}. ${paper.title}</div>
                        <div class="paper-card-meta">Click to view details</div>
                    </div>
                `).join('');
            }

            function displaySearchResults(papers) {
                const container = document.getElementById('browseContainer');
                
                if (papers.length === 0) {
                    container.innerHTML = '<div class="error">No papers found matching your search.</div>';
                    return;
                }
                
                container.innerHTML = papers.map(paper => `
                    <div class="paper-card" onclick="viewPaperDetails(${paper.id})">
                        <div class="paper-card-title">${paper.id}. ${paper.title}</div>
                        <div class="paper-card-meta">
                            <div style="margin: 5px 0;">📅 ${paper.year} | 👥 ${paper.authors}</div>
                        </div>
                        ${paper.keywords && paper.keywords.length > 0 ? `
                            <div class="paper-card-keywords">
                                ${paper.keywords.slice(0, 8).map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
                            </div>
                        ` : ''}
                    </div>
                `).join('');
            }

            function displayPagination(currentPage, totalPages, isSearch) {
                const pagination = document.getElementById('browsePagination');
                
                if (totalPages <= 1) return;
                
                let html = '';
                
                // Previous button
                if (currentPage > 1) {
                    html += `<button onclick="${isSearch ? 'searchPapers' : 'loadPapersList'}(${currentPage - 1})">← Previous</button>`;
                }
                
                // Page numbers
                const maxButtons = 7;
                let startPage = Math.max(1, currentPage - Math.floor(maxButtons / 2));
                let endPage = Math.min(totalPages, startPage + maxButtons - 1);
                
                if (endPage - startPage < maxButtons - 1) {
                    startPage = Math.max(1, endPage - maxButtons + 1);
                }
                
                if (startPage > 1) {
                    html += `<button onclick="${isSearch ? 'searchPapers' : 'loadPapersList'}(1)">1</button>`;
                    if (startPage > 2) html += `<span style="padding: 10px;">...</span>`;
                }
                
                for (let i = startPage; i <= endPage; i++) {
                    html += `<button class="${i === currentPage ? 'active' : ''}" onclick="${isSearch ? 'searchPapers' : 'loadPapersList'}(${i})">${i}</button>`;
                }
                
                if (endPage < totalPages) {
                    if (endPage < totalPages - 1) html += `<span style="padding: 10px;">...</span>`;
                    html += `<button onclick="${isSearch ? 'searchPapers' : 'loadPapersList'}(${totalPages})">${totalPages}</button>`;
                }
                
                // Next button
                if (currentPage < totalPages) {
                    html += `<button onclick="${isSearch ? 'searchPapers' : 'loadPapersList'}(${currentPage + 1})">Next →</button>`;
                }
                
                pagination.innerHTML = html;
            }

            // Paper Details Modal
            async function viewPaperDetails(paperId) {
                const modal = document.getElementById('paperModal');
                const modalContent = document.getElementById('modalContent');
                
                modal.classList.add('active');
                modalContent.innerHTML = '<div class="spinner"></div><div style="text-align: center; margin-top: 15px;">Loading paper details...</div>';
                
                try {
                    const response = await fetch(`/papers/${paperId}`);
                    
                    if (!response.ok) {
                        throw new Error('Failed to load paper details');
                    }
                    
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        const paper = data.paper;
                        modalContent.innerHTML = `
                            <div class="modal-title">${paper.title}</div>
                            
                            <div class="modal-section">
                                <h3>👥 Authors</h3>
                                <div>${paper.authors}</div>
                            </div>
                            
                            <div class="modal-section">
                                <h3>📅 Publication Year</h3>
                                <div>${paper.year}</div>
                            </div>
                            
                            ${paper.keywords && paper.keywords.length > 0 ? `
                                <div class="modal-section">
                                    <h3>🏷️ Keywords</h3>
                                    <div class="paper-card-keywords">
                                        ${paper.keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            <div class="modal-section">
                                <h3>📖 Abstract</h3>
                                <div class="modal-abstract">${paper.abstract}</div>
                            </div>
                            
                            <div class="modal-section">
                                <a href="${paper.link}" target="_blank" class="modal-link">🔗 View Full Paper on PMC</a>
                            </div>
                        `;
                    }
                    
                } catch (err) {
                    modalContent.innerHTML = `<div class="error">Error loading paper details: ${err.message}</div>`;
                    console.error('Error:', err);
                }
            }

            function closeModal() {
                document.getElementById('paperModal').classList.remove('active');
            }

            // Close modal when clicking outside
            window.onclick = function(event) {
                const modal = document.getElementById('paperModal');
                if (event.target === modal) {
                    closeModal();
                }
            }

            function showError(message, elementId) {
                const errorDiv = document.getElementById(elementId);
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
            }
            
            // Load total paper count on page load
            window.addEventListener('load', async () => {
                loadTheme();
                try {
                    const response = await fetch('/papers/count');
                    if (response.ok) {
                        const data = await response.json();
                        if (data.status === 'success') {
                            const stats = document.getElementById('stats');
                            stats.innerHTML = `📊 Total papers available: ${data.total_papers}`;
                        }
                    }
                } catch (err) {
                    console.error('Failed to load paper count:', err);
                }
            });

            // AI Analysis Functions
            async function runTopicModeling() {
                const clusters = document.getElementById('clusterCount').value;
                const maxPapers = document.getElementById('topicPaperCount').value;
                const loading = document.getElementById('aiLoading');
                const error = document.getElementById('aiError');
                const results = document.getElementById('aiResults');
                const btn = document.getElementById('topicBtn');
                
                loading.style.display = 'block';
                error.style.display = 'none';
                results.innerHTML = '';
                btn.disabled = true;
                btn.textContent = '🔄 Processing...';
                
                try {
                    const response = await fetch(`/api/topics?clusters=${clusters}&max_papers=${maxPapers}`);
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        displayTopics(data.result);
                    } else {
                        throw new Error(data.error || 'Failed to perform topic modeling');
                    }
                } catch (err) {
                    error.textContent = 'Error: ' + err.message;
                    error.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                    btn.disabled = false;
                    btn.textContent = '🔍 Find Topics';
                }
            }

            function displayTopics(result) {
                const container = document.getElementById('aiResults');
                
                let html = `<div class="stats">Found ${result.clusters.length} topics from ${result.total_papers} papers</div>`;
                
                result.clusters.forEach(cluster => {
                    html += `
                        <div class="paper-card" style="cursor: default;">
                            <div class="paper-card-title">${cluster.title}</div>
                            <div class="paper-card-meta" style="margin: 10px 0;">
                                📚 ${cluster.paper_count} papers
                            </div>
                            ${cluster.summary ? `
                                <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #007bff;">
                                    <strong>📝 Summary:</strong><br>${cluster.summary}
                                </div>
                            ` : ''}
                            <div class="paper-card-keywords">
                                <strong>🏷️ Top Terms:</strong>
                                ${cluster.top_terms.slice(0, 8).map(term => `<span class="keyword-tag">${term}</span>`).join('')}
                            </div>
                            <details style="margin-top: 15px;">
                                <summary style="cursor: pointer; font-weight: bold; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                    📄 View Papers (${cluster.papers.length})
                                </summary>
                                <div style="margin-top: 10px; padding-left: 15px;">
                                    ${cluster.papers.map(paper => `
                                        <div style="padding: 8px 0; border-bottom: 1px solid #eee;">
                                            <a href="#" onclick="viewPaperDetails(${paper.id}); return false;" style="color: #007bff; text-decoration: none;">
                                                ${paper.id}. ${paper.title}
                                            </a>
                                        </div>
                                    `).join('')}
                                </div>
                            </details>
                        </div>
                    `;
                });
                
                container.innerHTML = html;
            }

            async function summarizePaper() {
                const paperId = document.getElementById('summarizePaperId').value;
                
                if (!paperId) {
                    showAIError('Please enter a paper ID');
                    return;
                }
                
                const loading = document.getElementById('aiLoading');
                const error = document.getElementById('aiError');
                const results = document.getElementById('aiResults');
                
                loading.style.display = 'block';
                error.style.display = 'none';
                results.innerHTML = '';
                
                try {
                    const response = await fetch(`/api/summarize/${paperId}`);
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        results.innerHTML = `
                            <div class="paper-card">
                                <div class="paper-card-title">Paper #${data.paper_id} - Summary</div>
                                
                                <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4caf50;">
                                    <strong>✨ AI Summary:</strong><br><br>${data.summary}
                                </div>
                                
                                <details style="margin-top: 15px;">
                                    <summary style="cursor: pointer; font-weight: bold; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                        📄 View Original Abstract
                                    </summary>
                                    <div style="margin-top: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px;">
                                        ${data.original_abstract}
                                    </div>
                                </details>
                            </div>
                        `;
                    } else {
                        throw new Error(data.error || 'Failed to summarize');
                    }
                } catch (err) {
                    showAIError('Error: ' + err.message);
                } finally {
                    loading.style.display = 'none';
                }
            }

            async function extractEntities() {
                const paperId = document.getElementById('entitiesPaperId').value;
                
                if (!paperId) {
                    showAIError('Please enter a paper ID');
                    return;
                }
                
                const loading = document.getElementById('aiLoading');
                const error = document.getElementById('aiError');
                const results = document.getElementById('aiResults');
                
                loading.style.display = 'block';
                error.style.display = 'none';
                results.innerHTML = '';
                
                try {
                    const response = await fetch(`/api/entities/${paperId}`);
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        const entities = data.entities;
                        results.innerHTML = `
                            <div class="paper-card">
                                <div class="paper-card-title">${data.title}</div>
                                
                                ${entities.organisms.length > 0 ? `
                                    <div class="modal-section">
                                        <h3>🔬 Organisms</h3>
                                        <div class="paper-card-keywords">
                                            ${entities.organisms.map(e => `<span class="keyword-tag" style="background: #e1f5fe; color: #0277bd;">${e}</span>`).join('')}
                                        </div>
                                    </div>
                                ` : ''}
                                
                                ${entities.conditions.length > 0 ? `
                                    <div class="modal-section">
                                        <h3>⚗️ Conditions</h3>
                                        <div class="paper-card-keywords">
                                            ${entities.conditions.map(e => `<span class="keyword-tag" style="background: #fff3e0; color: #e65100;">${e}</span>`).join('')}
                                        </div>
                                    </div>
                                ` : ''}
                                
                                ${entities.outcomes.length > 0 ? `
                                    <div class="modal-section">
                                        <h3>📊 Outcomes</h3>
                                        <div class="paper-card-keywords">
                                            ${entities.outcomes.map(e => `<span class="keyword-tag" style="background: #f3e5f5; color: #6a1b9a;">${e}</span>`).join('')}
                                        </div>
                                    </div>
                                ` : ''}
                                
                                ${entities.other.length > 0 ? `
                                    <div class="modal-section">
                                        <h3>🏷️ Other Entities</h3>
                                        <div class="paper-card-keywords">
                                            ${entities.other.map(e => `<span class="keyword-tag">${e}</span>`).join('')}
                                        </div>
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    } else {
                        throw new Error(data.error || 'Failed to extract entities');
                    }
                } catch (err) {
                    showAIError('Error: ' + err.message);
                } finally {
                    loading.style.display = 'none';
                }
            }

            async function runBatchAnalysis() {
                const paperIds = document.getElementById('batchPaperIds').value;
                
                if (!paperIds) {
                    showAIError('Please enter paper IDs');
                    return;
                }
                
                const loading = document.getElementById('aiLoading');
                const error = document.getElementById('aiError');
                const results = document.getElementById('aiResults');
                const btn = document.getElementById('batchBtn');
                
                loading.style.display = 'block';
                error.style.display = 'none';
                results.innerHTML = '';
                btn.disabled = true;
                btn.textContent = '🔄 Processing...';
                
                try {
                    const response = await fetch(`/api/batch-analyze?ids=${paperIds}`);
                    const data = await response.json();
                    
                    if (data.status === 'success') {
                        displayBatchResults(data.papers);
                    } else {
                        throw new Error(data.error || 'Failed to analyze papers');
                    }
                } catch (err) {
                    showAIError('Error: ' + err.message);
                } finally {
                    loading.style.display = 'none';
                    btn.disabled = false;
                    btn.textContent = '📊 Analyze Papers';
                }
            }

            function displayBatchResults(papers) {
                const container = document.getElementById('aiResults');
                
                if (papers.length === 0) {
                    container.innerHTML = '<div class="error">No papers analyzed.</div>';
                    return;
                }
                
                let html = `<div class="stats">Analyzed ${papers.length} papers</div>`;
                
                papers.forEach(paper => {
                    html += `
                        <div class="paper-card" style="cursor: default;">
                            <div class="paper-card-title">${paper.id}. ${paper.title}</div>
                            
                            ${paper.summary ? `
                                <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 15px 0; border-left: 4px solid #4caf50;">
                                    <strong>✨ Summary:</strong><br><br>${paper.summary}
                                </div>
                            ` : ''}
                            
                            ${paper.keywords && paper.keywords.length > 0 ? `
                                <div class="modal-section">
                                    <h3>🏷️ Keywords</h3>
                                    <div class="paper-card-keywords">
                                        ${paper.keywords.map(kw => `<span class="keyword-tag">${kw}</span>`).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${paper.entities ? `
                                <div style="margin-top: 15px;">
                                    <details>
                                        <summary style="cursor: pointer; font-weight: bold; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                            🔬 View Extracted Entities
                                        </summary>
                                        <div style="margin-top: 10px;">
                                            ${paper.entities.organisms.length > 0 ? `
                                                <div class="modal-section">
                                                    <h4>🧬 Organisms</h4>
                                                    <div class="paper-card-keywords">
                                                        ${paper.entities.organisms.map(e => `<span class="keyword-tag" style="background: #e1f5fe; color: #0277bd;">${e}</span>`).join('')}
                                                    </div>
                                                </div>
                                            ` : ''}
                                            
                                            ${paper.entities.conditions.length > 0 ? `
                                                <div class="modal-section">
                                                    <h4>⚗️ Conditions</h4>
                                                    <div class="paper-card-keywords">
                                                        ${paper.entities.conditions.map(e => `<span class="keyword-tag" style="background: #fff3e0; color: #e65100;">${e}</span>`).join('')}
                                                    </div>
                                                </div>
                                            ` : ''}
                                            
                                            ${paper.entities.outcomes.length > 0 ? `
                                                <div class="modal-section">
                                                    <h4>📊 Outcomes</h4>
                                                    <div class="paper-card-keywords">
                                                        ${paper.entities.outcomes.map(e => `<span class="keyword-tag" style="background: #f3e5f5; color: #6a1b9a;">${e}</span>`).join('')}
                                                    </div>
                                                </div>
                                            ` : ''}
                                        </div>
                                    </details>
                                </div>
                            ` : ''}
                        </div>
                    `;
                });
                
                container.innerHTML = html;
            }

            function showAIError(message) {
                const error = document.getElementById('aiError');
                error.textContent = message;
                error.style.display = 'block';
            }
            // Dark Mode Toggle
            function toggleTheme() {
                const html = document.documentElement;
                const themeToggle = document.getElementById('themeToggle');
                const currentTheme = html.getAttribute('data-theme');
                
                if (currentTheme === 'dark') {
                    html.removeAttribute('data-theme');
                    themeToggle.textContent = '🌙';
                    themeToggle.title = 'Switch to Dark Mode';
                    localStorage.setItem('theme', 'light');
                } else {
                    html.setAttribute('data-theme', 'dark');
                    themeToggle.textContent = '☀️';
                    themeToggle.title = 'Switch to Light Mode';
                    localStorage.setItem('theme', 'dark');
                }
            }

            // Load saved theme on page load
            function loadTheme() {
                const savedTheme = localStorage.getItem('theme');
                const html = document.documentElement;
                const themeToggle = document.getElementById('themeToggle');
                
                if (savedTheme === 'dark') {
                    html.setAttribute('data-theme', 'dark');
                    themeToggle.textContent = '☀️';
                    themeToggle.title = 'Switch to Light Mode';
                } else {
                    themeToggle.textContent = '🌙';
                    themeToggle.title = 'Switch to Dark Mode';
                }
            }

            // Call loadTheme when page loads
            window.addEventListener('DOMContentLoaded', loadTheme);
        </script>
    </body>
    </html>
    """

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    try:
        from bs4 import BeautifulSoup
        print("✓ All packages imported successfully!")
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Please install required packages:")
        print("pip install beautifulsoup4 pandas flask flask-cors requests")
        exit(1)
        
    print("Starting Research Papers Viewer...")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)




