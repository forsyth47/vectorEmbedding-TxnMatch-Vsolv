"""
Ultra-Pro-Max-Efficient Entity Matching Pipeline
Combines Sentence-Transformers + HNSWlib/Numpy + RapidFuzz

Features:
- Memory-efficient disk-based processing
- Configurable similarity search: HNSWlib or Numpy (removed due numpy sucks)
- Named Entity Recognition with spaCy for city detection (auto)
- city/locality matching for confidence boosting (auto)
- Chunked processing to avoid memory spikes
- CPU-only processing with optional GPU for embeddings
- Real-time profiling and monitoring
- Configurable output columns
- Improved scoring system with proper weighting
"""

#Basic imports and setup
import os
import re
import gc
import time
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Set
import warnings
import numpy as np
import pandas as pd
import polars as pl
import psutil
import h5py
warnings.filterwarnings('ignore')

# Core ML libraries
import hnswlib
import spacy
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz

# Add multiprocessing imports for parallel batch processing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

#=== Memory Profiler Class ===

class MemoryProfiler:
    """Memory and performance profiler"""

    def __init__(self, log_file: str = None):
        if log_file is None and hasattr(EfficientEntityMatcher, "work_dir"):
            log_file = f"{EfficientEntityMatcher.work_dir}/matching_profile.log"
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.last_check = self.start_time

        # Setup logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage"""
        memory_info = self.process.memory_info()
        return {
            'rss_gb': memory_info.rss / 1024**3,  # Physical memory
            'vms_gb': memory_info.vms / 1024**3,  # Virtual memory
            'percent': self.process.memory_percent()
        }

    def profile_checkpoint(self, stage: str, records_processed: int = 0):
        """Log memory and timing info at checkpoint"""
        current_time = time.time()
        memory_info = self.get_memory_info()

        elapsed_total = current_time - self.start_time
        elapsed_stage = current_time - self.last_check

        log_msg = (
            f"STAGE: {stage} | "
            f"Records: {records_processed:,} | "
            f"Memory: {memory_info['rss_gb']:.2f}GB (Physical), "
            f"{memory_info['vms_gb']:.2f}GB (Virtual), "
            f"{memory_info['percent']:.1f}% | "
            f"Time: {elapsed_stage:.2f}s (stage), {elapsed_total:.2f}s (total)"
        )

        print(log_msg)
        self.logger.info(log_msg)
        self.last_check = current_time

        # Force garbage collection if memory usage is high
        if memory_info['rss_gb'] > 12.0:  # 75% of 16GB
            gc.collect()
            self.logger.warning(f"High memory usage detected, forced garbage collection")

class AutomatedCityMatcher:
    """Automated city/locality matching using NER and comprehensive preprocessing with persistent storage"""

    def __init__(self, nlp_model=None, enable_automated_detection: bool = True, work_dir: Path = None):
        self.nlp = nlp_model
        self.enable_automated_detection = enable_automated_detection
        self.work_dir = work_dir or Path("matching_workspace")

        # Cache for extracted cities to avoid repeated NER processing
        self.city_cache = {}

        # Persistent storage for discovered locations
        self.discovered_locations_file = self.work_dir / "discovered_locations.txt"
        self.discovered_locations = set()

        # Load previously discovered locations on initialization
        self._load_discovered_locations()

        # Pre-compiled regex patterns for transaction cleaning
        self.transaction_prefixes = re.compile(r'^(neft|rtgs|imps|upi|ach|nach|ecs|swift|wire|transfer)[\s/]', re.IGNORECASE)
        self.transaction_codes = re.compile(r'[a-z0-9]{10,}', re.IGNORECASE)

        # Pre-compiled regex patterns for common location indicators
        self.location_patterns = [
            re.compile(r'\b(city|town|village|nagar|pur|bad|ganj|gram)\b', re.IGNORECASE),
            re.compile(r'\b(district|taluk|tehsil|block|mandal)\b', re.IGNORECASE),
            re.compile(r'\b(state|pradesh|bengal|karnataka|tamil|nadu|maharashtra)\b', re.IGNORECASE),
        ]

        # Common location suffixes in Indian context
        self.location_suffixes = {'nagar', 'pur', 'bad', 'ganj', 'gram', 'puram', 'patnam', 'kota',
            'ville', 'pally', 'wadi', 'garh', 'kot', 'gunj', 'tola', 'para',
            'puram', 'salem', 'coimbatore', 'madurai', 'trichy', 'kanchipuram',
            'ramanathapuram', 'tirunelveli', 'vellore', 'thanjavur'}

        # Major Indian cities for fallback (expanded set)
        self.major_cities = {
            'chennai', 'mumbai', 'delhi', 'bangalore', 'bengaluru', 'hyderabad',
            'pune', 'kolkata', 'ahmedabad', 'surat', 'jaipur', 'lucknow',
            'kanpur', 'nagpur', 'indore', 'thane', 'bhopal', 'visakhapatnam',
            'patna', 'vadodara', 'ghaziabad', 'ludhiana', 'agra', 'nashik',
            'kochi', 'cochin', 'madurai', 'coimbatore', 'trichy', 'salem',
            'tirunelveli', 'vellore', 'thanjavur', 'kanchipuram', 'kancheepuram',
            'ramanathapuram', 'tiruchirappalli', 'tiruchirapalli'
        }

        # Cache for location entities found during processing
        self.discovered_locations = set()

    def clean_transaction_text(self, text: str) -> str:
        """Clean transaction text to extract meaningful company names and locations"""
        if not text or pd.isna(text):
            return ''

        text = str(text).strip().lower()

        # Remove transaction prefixes (neft/, rtgs/, etc.)
        text = self.transaction_prefixes.sub('', text)

        # Split by common delimiters and extract meaningful parts
        parts = re.split(r'[/\-_\s]+', text)

        # Filter out transaction codes and IDs (long alphanumeric strings)
        meaningful_parts = []
        for part in parts:
            part = part.strip()
            if (len(part) > 2 and
                not self.transaction_codes.fullmatch(part) and
                not part.isdigit() and
                len(part) < 50):  # Avoid very long strings
                meaningful_parts.append(part)

        return ' '.join(meaningful_parts)

    def _load_discovered_locations(self):
        """Load previously discovered locations from persistent storage"""
        if self.discovered_locations_file.exists():
            try:
                with open(self.discovered_locations_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):  # Skip comments and empty lines
                            self.discovered_locations.add(line)
                #print(f"📍 Loaded {len(self.discovered_locations)} previously discovered locations")
            except Exception as e:
                #print(f"⚠️  Failed to load discovered locations: {e}")
                self.discovered_locations = set()

    def _save_discovered_locations(self):
        """Save discovered locations to persistent storage"""
        try:
            self.work_dir.mkdir(exist_ok=True)
            with open(self.discovered_locations_file, 'w', encoding='utf-8') as f:
                f.write("# Automatically Discovered Locations\n")
                f.write(f"# Total count: {len(self.discovered_locations)}\n")
                f.write(f"# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for location in sorted(self.discovered_locations):
                    f.write(f"{location}\n")
        except Exception as e:
            print(f"⚠️  Failed to save discovered locations: {e}")

    def extract_cities_automated(self, text: str) -> List[str]:
        """Extract cities/locations using automated NER and pattern matching with improved cleaning"""
        if not text or pd.isna(text):
            return []

        # Check cache first
        text_key = text.lower().strip()
        if text_key in self.city_cache:
            return self.city_cache[text_key]

        found_locations = []

        # Clean the transaction text first
        cleaned_text = self.clean_transaction_text(text)

        # Method 1: NER-based extraction (if enabled and available)
        if self.enable_automated_detection and self.nlp and cleaned_text:
            try:
                doc = self.nlp(cleaned_text)
                for ent in doc.ents:
                    if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entities, Locations
                        location = ent.text.lower().strip()
                        if (len(location) > 2 and
                            len(location) < 30 and  # Reasonable city name length
                            location not in found_locations):
                            found_locations.append(location)
                            self.discovered_locations.add(location)
            except Exception as e:
                pass  # Silently continue if NER fails

        # Method 2: Pattern-based extraction with improved filtering
        words = cleaned_text.split()
        for word in words:
            word = word.strip()
            if len(word) > 3:  # Minimum length for city names
                # Check for location suffixes
                for suffix in self.location_suffixes:
                    if (word.endswith(suffix) and
                        len(word) > len(suffix) + 2 and
                        len(word) < 25):  # Reasonable city name length
                        # Extract the city name without common prefixes
                        location = word
                        if location not in found_locations:
                            found_locations.append(location)
                            self.discovered_locations.add(location)

        # Method 3: Major cities detection in cleaned text
        for city in self.major_cities:
            if city in cleaned_text:
                if city not in found_locations:
                    found_locations.append(city)

        # Method 4: Look for discovered locations from previous processing
        for location in self.discovered_locations:
            if (location in cleaned_text and
                location not in found_locations and
                len(location) > 3):
                found_locations.append(location)

        # Cache the result
        self.city_cache[text_key] = found_locations
        return found_locations

    def calculate_city_boost(self, txn_cities: List[str], company_cities: List[str]) -> float:
        """Calculate confidence boost based on city matching with much more conservative scoring"""
        if not txn_cities or not company_cities:
            return 0.0

        # Perfect match gets a SMALL boost - city should be supporting evidence, not primary
        common_cities = set(txn_cities) & set(company_cities)
        if common_cities:
            return 3.0  # Reduced from 20.0 to 3.0 - much more reasonable

        # Partial match using fuzzy matching for city names
        for txn_city in txn_cities:
            for company_city in company_cities:
                # Use fuzzy matching for similar city names
                similarity = fuzz.ratio(txn_city, company_city)
                if similarity >= 80:  # 80% similarity threshold
                    return 2.0  # Reduced from 15.0 to 2.0
                elif similarity >= 60:  # 60% similarity threshold
                    return 1.0  # Reduced from 10.0 to 1.0

        return 0.0

    def get_discovered_locations_count(self) -> int:
        """Get count of automatically discovered locations"""
        return len(self.discovered_locations)

    def get_discovered_locations(self) -> Set[str]:
        """Get all automatically discovered locations"""
        return self.discovered_locations.copy()

    def save_discovered_locations(self, file_path: str = None) -> None:
        """Save discovered locations to a file for inspection"""
        if file_path is None:
            file_path = self.discovered_locations_file

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Automatically Discovered Locations\n")
                f.write(f"# Total count: {len(self.discovered_locations)}\n")
                f.write(f"# Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for location in sorted(self.discovered_locations):
                    f.write(f"{location}\n")
            print(f"📍 Discovered locations saved to: {file_path}")
        except Exception as e:
            print(f"⚠️  Failed to save discovered locations: {e}")

    def finalize_session(self):
        """Save discovered locations at the end of processing session"""
        #self._save_discovered_locations()
        #print(f"💾 Saved {len(self.discovered_locations)} discovered locations for future use")

class TransactionCleaner:
    """Intelligent transaction text cleaner that removes noise while preserving company names"""

    def __init__(self, enable_filtering: bool = True):
        self.enable_filtering = enable_filtering

        # Comprehensive list of Indian banks (major banks)
        self.indian_banks = {
            'hdfc bank', 'hdfc', 'icici bank', 'icici', 'sbi', 'state bank of india',
            'axis bank', 'axis', 'kotak mahindra bank', 'kotak', 'yes bank',
            'indusind bank', 'indusind', 'federal bank', 'idfc first bank', 'idfc first bank ltd', 'idfc',
            'rbl bank', 'karur vysya bank', 'south indian bank', 'city union bank',
            'union bank', 'union bank of india', 'punjab national bank', 'pnb',
            'bank of baroda', 'bob', 'canara bank', 'indian bank', 'central bank',
            'bank of india', 'boi', 'oriental bank', 'corporation bank',
            'andhra bank', 'allahabad bank', 'syndicate bank', 'vijaya bank',
            'dena bank', 'indian overseas bank', 'iob', 'uco bank',
            'central bank of india', 'cbi', 'punjab and sind bank',
            'standard chartered', 'standard chart', 'citibank', 'citi bank',
            'deutsche bank', 'hsbc', 'barclays', 'american express',
            'idbi bank', 'idbi', 'karnataka bank', 'tamilnad mercantile bank',
            'lakshmi vilas bank', 'lvb', 'dhanlaxmi bank', 'nainital bank',
            'jana small finance bank', 'equitas small finance bank',
            'ujjivan small finance bank', 'au small finance bank',
            'capital small finance bank', 'esaf small finance bank',
            'fincare small finance bank', 'north east small finance bank',
            'suryoday small finance bank', 'utkarsh small finance bank',
            'bandhan bank', 'paytm payments bank', 'airtel payments bank',
            'india post payments bank', 'fino payments bank', 'jio payments bank',
            'nsdl payments bank', 'aditya birla idea payments bank'
        }

        # Transaction noise patterns (prefixes that should be removed)
        self.noise_patterns = {
            'neft', 'rtgs', 'imps', 'upi', 'ach', 'nach', 'ecs', 'swift',
            'wire', 'transfer', 'inb', 'ift', 'clg', 'htr', 'domneft',
            'paid to', 'payment to', 'transfer to', 'remittance',
            'tparty transfer', 'party transfer', 'emi'
        }

        # Company indicators that suggest it's a legitimate business
        self.company_indicators = {
            'pvt', 'private', 'ltd', 'limited', 'inc', 'corp', 'co', 'company',
            'enterprises', 'industries', 'services', 'solutions', 'technologies',
            'systems', 'motors', 'textiles', 'traders', 'exports', 'imports',
            'manufacturing', 'construction', 'engineering', 'consultancy',
            'pharmacy', 'medical', 'hospital', 'clinic', 'school', 'college',
            'institute', 'foundation', 'trust', 'society', 'association',
            'federation', 'union', 'group', 'holdings', 'investment',
            'finance', 'insurance', 'bank', 'financial', 'capital'
        }

        # Compile regex patterns for efficient matching
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""

        # Pattern for transaction prefixes (case insensitive)
        noise_pattern = '|'.join(re.escape(pattern) for pattern in self.noise_patterns)
        self.noise_regex = re.compile(rf'\b({noise_pattern})\b[/\s]*', re.IGNORECASE)

        # Pattern for account numbers and transaction IDs
        # Matches sequences like: A**********885, 0003**********89, N095*********
        self.account_number_regex = re.compile(r'\b[A-Z0-9]*\*{3,}[A-Z0-9]*\b', re.IGNORECASE)

        # Pattern for transaction codes (long alphanumeric strings)
        self.transaction_code_regex = re.compile(r'\b[A-Z0-9]{10,}\b', re.IGNORECASE)

        # Pattern for dates in various formats
        self.date_regex = re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b')

        # Pattern for common separators and noise
        self.separator_regex = re.compile(r'[/\-_]{2,}')

        # Bank name pattern (will be built dynamically)
        bank_pattern = '|'.join(re.escape(bank) for bank in self.indian_banks)
        self.bank_regex = re.compile(rf'\b({bank_pattern})\b', re.IGNORECASE)

        # Company indicator pattern
        company_pattern = '|'.join(re.escape(indicator) for indicator in self.company_indicators)
        self.company_indicator_regex = re.compile(rf'\b({company_pattern})\b', re.IGNORECASE)

        # Pattern for preserving company suffixes (should NOT be removed)
        self.preserve_suffixes = {
            'pvt', 'private', 'ltd', 'limited', 'inc', 'corp', 'co', 'company',
            'enterprises', 'industries', 'services', 'solutions', 'technologies',
            'systems', 'motors', 'textiles', 'traders', 'exports', 'imports'
        }

    def clean_transaction_text(self, text: str) -> Dict[str, str]:
        """
        Clean transaction text intelligently
        Returns dict with 'original' and 'cleaned' text
        """
        if not text or pd.isna(text):
            return {'original': '', 'cleaned': ''}

        original_text = str(text).strip()

        if not self.enable_filtering:
            # If filtering is disabled, return original text as both original and cleaned
            return {'original': original_text, 'cleaned': original_text}

        cleaned_text = original_text.lower()

        # Step 1: Remove transaction noise patterns (NEFT, RTGS, UPI, etc.)
        cleaned_text = self.noise_regex.sub(' ', cleaned_text)

        # Step 2: Remove account numbers and transaction IDs
        cleaned_text = self.account_number_regex.sub(' ', cleaned_text)
        cleaned_text = self.transaction_code_regex.sub(' ', cleaned_text)

        # Step 3: Remove dates
        cleaned_text = self.date_regex.sub(' ', cleaned_text)

        # Step 4: Clean up multiple separators
        cleaned_text = self.separator_regex.sub(' ', cleaned_text)

        # Step 5: Remove bank names while preserving company names
        cleaned_text = self._remove_banks_intelligently(cleaned_text)

        # Step 6: Clean up extra whitespace and punctuation
        cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)
        cleaned_text = ' '.join(cleaned_text.split())  # Normalize whitespace

        # Step 7: Remove very short words (likely noise) but preserve important suffixes
        words = cleaned_text.split()
        filtered_words = []

        for word in words:
            # Keep words that are:
            # - Longer than 2 characters, OR
            # - Important company suffixes, OR
            # - All uppercase (likely acronyms)
            if (len(word) > 2 or
                word.lower() in self.preserve_suffixes or
                (word.isupper() and len(word) >= 2)):
                filtered_words.append(word)

        cleaned_text = ' '.join(filtered_words)

        return {
            'original': original_text,
            'cleaned': cleaned_text.strip()
        }

    def _remove_banks_intelligently(self, text: str) -> str:
        """Remove bank names while being careful not to remove legitimate company names"""

        # More aggressive bank name removal
        words = text.split()
        filtered_words = []

        i = 0
        while i < len(words):
            word = words[i].lower()

            # Check if current word or next few words form a bank name
            bank_match_found = False

            # Check single word bank names
            if word in self.indian_banks:
                bank_match_found = True

            # Check two-word bank names
            if i + 1 < len(words):
                two_word = f"{word} {words[i+1].lower()}"
                if two_word in self.indian_banks:
                    bank_match_found = True
                    i += 1 # Skip next word too

            # Check three-word bank names
            if i + 2 < len(words):
                three_word = f"{word} {words[i+1].lower()} {words[i+2].lower()}"
                if three_word in self.indian_banks:
                    bank_match_found = True
                    i += 2 # Skip next two words too

            if not bank_match_found:
                filtered_words.append(words[i])

            i += 1

        return ' '.join(filtered_words)

    def filter_detected_company_name(self, text: str) -> str:
        """
        Apply additional filtering specifically for detected company names
        This removes bank names, personal names, and remaining noise patterns
        """
        if not text or pd.isna(text):
            return ''

        text_str = str(text).strip()


        # Remove any remaining bank references
        filtered_text = self.bank_regex.sub('', text_str)

        # Remove remaining noise patterns
        filtered_text = self.noise_regex.sub('', filtered_text)

        # Clean up whitespace
        filtered_text = ' '.join(filtered_text.split()).strip()

        # If nothing meaningful remains, return empty
        if len(filtered_text) < 3:
            return ''


        return filtered_text
# Add TransactionCleaner to EfficientEntityMatcher class
class EfficientEntityMatcher:
    """
    Main matching pipeline class with HNSWlib optimization and improved scoring

    Scoring System Explanation:
    - hnswlib_similarity: 0.0-1.0 (cosine similarity from embeddings)
    - fuzzy_score: 0.0-100.0 (string similarity using RapidFuzz)
    - city_boost: 0.0-20.0 (bonus for location matching)
    - final_score: Weighted combination with proper normalization

    The final score combines:
    1. Embedding similarity (weighted and scaled to 0-100)
    2. Fuzzy string matching (0-100)
    3. City location boost (0-20)
    With intelligent weighting based on confidence levels.
    """

    def __init__(self,
                 batch_size: int = 10000,
                 confidence_threshold: float = 85.0,
                 max_candidates: int = 10,
                 embedding_model: str = "paraphrase-MiniLM-L6-v2",
                 ef_construction: int = 600,
                 M: int = 512,  # Increased from 64 to 512 for faster search
                 ef_search: int = 200,
                 conservative_M: int = 64,  # Much lower for large datasets
                 conservative_ef_construction: int = 200,  # Lower for large datasets
                 company_column: str = None,
                 transaction_column: str = None,
                 enable_automated_city_detection: bool = True,
                 output_columns: Dict[str, bool] = None,

                 #NEW PARALLEL PROCESSING PARAMETERS
                 enable_parallel_processing: bool = False,
                 parallel_workers: int = 2,
                 parallel_method: str = 'thread',

                 # NEW TRANSACTION FILTERING PARAMETERS
                 enable_transaction_filtering: bool = True):

        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.max_candidates = max_candidates
        self.embedding_model_name = embedding_model
        self.ef_construction = ef_construction
        self.M = M
        self.ef_search = ef_search
        self.conservative_M = conservative_M
        self.conservative_ef_construction = conservative_ef_construction
        self.company_column = company_column
        self.transaction_column = transaction_column
        self.enable_automated_city_detection = enable_automated_city_detection
        self.enable_transaction_filtering = enable_transaction_filtering

        # NEW PARALLEL PROCESSING PARAMETERS
        self.enable_parallel_processing = enable_parallel_processing
        self.parallel_workers = parallel_workers
        self.parallel_method = parallel_method

        # Initialize Transaction Cleaner
        self.transaction_cleaner = TransactionCleaner(enable_filtering=self.enable_transaction_filtering)

        # Configure output columns (default: show all)
        default_columns = {
            'backend_used': True,
            'txn_cities': True,
            'company_cities': True,
            'hnswlib_similarity': True,
            'fuzzy_score': True,
            'city_boost': True,
            'final_score': True,
            'detected_company_name': True  # Shows cleaned/filtered transaction text used for matching
        }
        self.output_columns = output_columns if output_columns else default_columns

        # Initialize components
        self.profiler = MemoryProfiler()

        # Will be initialized later with NLP model
        self.city_matcher = None

        # Will be initialized later
        self.nlp = None
        self.embedding_model = None
        self.hnswlib_index = None
        self.company_embeddings = None
        self.company_data = None

        # Ultra-aggressive memory caching for maximum speed
        self.company_cache = {}
        self.cache_loaded = False

        # Pre-computed embedding cache for ultra-fast lookups
        self.embedding_cache = {}

        # Pre-compiled regex patterns for enhanced text normalization
        self.clean_regex = re.compile(r'[^\w\s]')
        self.whitespace_regex = re.compile(r'\s+')

        # Enhanced text normalization patterns
        self.normalize_patterns = [
            (re.compile(r'\bpvt\b|\bprivate\b|\bltd\b|\blimited\b|\binc\b|\bcorp\b|\bco\b', re.IGNORECASE), ''),
            (re.compile(r'\band\b|\b&\b', re.IGNORECASE), ''),
            (re.compile(r'\bthe\b|\ba\b|\ban\b', re.IGNORECASE), ''),
        ]

        # File paths
        self.work_dir = Path(f"workspace-{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        # self.work_dir = Path(f"workspace-99")
        self.work_dir.mkdir(exist_ok=True)

        self.embeddings_file = self.work_dir / "company_embeddings.h5"
        self.index_file = self.work_dir / "hnswlib_index.bin"
        self.db_file = self.work_dir / "companies.db"
        self.results_file = self.work_dir / f"matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    def initialize_models(self):
        """Initialize NLP and embedding models (CPU-only)"""
        print("🔧 Initializing models...")

        # Load spaCy model for NER and automated city detection
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded for automated city detection")
        except OSError:
            print("⚠️  spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Initialize automated city matcher with work directory for persistence
        self.city_matcher = AutomatedCityMatcher(
            nlp_model=self.nlp,
            enable_automated_detection=self.enable_automated_city_detection,
            work_dir=self.work_dir  # Pass work directory for persistent storage
        )

        # Load sentence transformer (CPU only)
        print("🚀 Loading embedding model on CPU (maximum stability)")
        self.embedding_model = SentenceTransformer(self.embedding_model_name, device='cpu')

        # Configure for memory efficiency
        if hasattr(self.embedding_model, 'max_seq_length'):
            self.embedding_model.max_seq_length = 128  # Limit sequence length

        self.profiler.profile_checkpoint("Models Initialized")

    def normalize_text_enhanced(self, text: str) -> str:
        """Enhanced text normalization for better company name matching"""
        if not text or pd.isna(text):
            return ''

        text = str(text).strip().lower()

        # Apply normalization patterns
        for pattern, replacement in self.normalize_patterns:
            text = pattern.sub(replacement, text)

        # Basic cleaning
        text = self.clean_regex.sub(' ', text)
        text = self.whitespace_regex.sub(' ', text).strip()

        return text

    def preprocess_text_companies(self, text: str, extract_entities: bool = True) -> Dict[str, Any]:
        """Preprocess text with automated NER and city extraction"""
        if not text or pd.isna(text):
            return {
                'cleaned': '',
                'normalized': '',
                'organizations': [],
                'cities': [],
                'tokens': []
            }

        text = str(text).strip()

        # Enhanced normalization
        normalized = self.normalize_text_enhanced(text)

        # Basic cleaning (keep numbers and letters)
        cleaned = re.sub(r'[^\w\s]', ' ', text)
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace

        result = {
            'cleaned': cleaned.lower(),
            'normalized': normalized,
            'organizations': [],
            'cities': [],
            'tokens': normalized.split() if normalized else cleaned.lower().split()
        }

        if extract_entities and self.nlp:
            try:
                doc = self.nlp(text)

                # Extract organizations
                orgs = [ent.text.lower() for ent in doc.ents if ent.label_ == "ORG"]
                result['organizations'] = orgs

            except Exception as e:
                print(f"⚠️  NER extraction failed: {e}")

        # Extract cities using automated detection
        if self.city_matcher:
            result['cities'] = self.city_matcher.extract_cities_automated(text)

        return result

    def preprocess_text_ultra_fast_txn(self, text: str) -> Dict[str, Any]:
        """Ultra-fast text preprocessing with enhanced normalization and intelligent filtering"""
        if not text or pd.isna(text):
            return {
                'original': '',
                'cleaned': '',
                'normalized': '',
                'cities': [],
                'tokens': []
            }

        text_str = str(text).strip()

        # Step 1: Use TransactionCleaner for intelligent filtering (if enabled)
        if self.enable_transaction_filtering:
            cleaned_result = self.transaction_cleaner.clean_transaction_text(text_str)
            original_text = cleaned_result['original']
            filtered_text = cleaned_result['cleaned']
        else:
            original_text = text_str
            filtered_text = text_str

        # Step 2: Enhanced normalization on the filtered text
        normalized = self.normalize_text_enhanced(filtered_text)

        # Step 3: Ultra-fast cleaning using pre-compiled regex on filtered text
        cleaned = self.clean_regex.sub(' ', filtered_text)
        cleaned = self.whitespace_regex.sub(' ', cleaned).strip().lower()

        result = {
            'original': original_text,  # Always keep original for output
            'cleaned': cleaned,
            'normalized': normalized,
            'cities': self.city_matcher.extract_cities_automated(filtered_text) if self.city_matcher else [],
            'tokens': normalized.split() if normalized else cleaned.split()
        }

        return result

    def load_and_preprocess_companies(self, companies_file: str) -> int:
        """Load and preprocess company data with enhanced normalization"""

        try:
            # Determine file type and read accordingly
            if companies_file.lower().endswith('.csv'):
                companies_df = pl.read_csv(companies_file)
            else:
                companies_df = pl.read_excel(companies_file)
            total_companies = len(companies_df)

            print(f"Loaded {total_companies} company records from {companies_file}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None

        # Auto-detect or use specified company column
        if self.company_column and len(self.company_column) > 0:
            company_col_name = self.company_column[0]
            if company_col_name not in companies_df.columns:
                print(f"⚠️  Specified company column '{company_col_name}' not found. Available columns: {companies_df.columns}")
                company_col_name = companies_df.columns[0]  # Fallback to first column
                print(f"Using fallback column: '{company_col_name}'")
        else:
            # Auto-detect company column
            company_col_name = companies_df.columns[0]
            print(f"Auto-detected company column: '{company_col_name}'")

        conn = sqlite3.connect(self.db_file)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        cursor = conn.cursor()

        # Create companies table with normalized_text column
        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS companies
                       (
                           id INTEGER PRIMARY KEY,
                           original_name TEXT,
                           cleaned_text TEXT,
                           normalized_text TEXT,
                           organizations TEXT,
                           cities TEXT,
                           tokens TEXT,
                           embedding_index INTEGER
                       )
                       """)

        # Prepare batch processing
        chunk_size = 10000
        processed_count = 0

        # Process companies in optimized chunks
        for i in range(0, total_companies, chunk_size):
            chunk_end = min(i + chunk_size, total_companies)
            chunk = companies_df[i:chunk_end]

            # Extract company names using the correct column
            company_names = chunk.get_column(company_col_name).to_list()

            # Batch preprocess all companies in chunk
            batch_data = []

            for j, company_name in enumerate(company_names):
                if not company_name or pd.isna(company_name):
                    company_name = ''

                company_name_str = str(company_name).strip()

                # Enhanced text processing
                processed = self.preprocess_text_companies(company_name_str, extract_entities=False)
                cleaned = processed['cleaned']
                normalized = processed['normalized']
                tokens = processed['tokens']

                # Get cities using automated detection
                found_cities = processed['cities']

                batch_data.append((
                    company_name_str,
                    cleaned,
                    normalized,
                    '',  # organizations (skip for companies)
                    '|'.join(found_cities),
                    '|'.join(tokens),
                    processed_count + j
                ))

            # Batch insert
            cursor.executemany("""
                               INSERT INTO companies
                               (original_name, cleaned_text, normalized_text, organizations, cities, tokens, embedding_index)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               """, batch_data)

            processed_count += len(batch_data)

            if processed_count % 25000 == 0 or chunk_end == total_companies:
                self.profiler.profile_checkpoint(f"Companies Preprocessed", processed_count)

        conn.commit()
        conn.close()

        self.profiler.profile_checkpoint("Company Preprocessing Complete", processed_count)
        print(f"🚀 Preprocessed {processed_count:,} companies with enhanced pipeline")
        print(f"📋 Used company column: '{company_col_name}'")
        return processed_count

    def create_company_embeddings_enhanced(self) -> None:
        """Create embeddings using normalized company text for better matching"""
        print("🔧 Creating enhanced company embeddings using normalized text...")

        # Load preprocessed companies
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Use normalized text for embeddings (fallback to cleaned if normalized is empty)
        cursor.execute("""
                       SELECT CASE
                                  WHEN normalized_text IS NOT NULL AND normalized_text != '' 
                THEN normalized_text
                                  ELSE cleaned_text
                                  END as text_for_embedding
                       FROM companies
                       ORDER BY id
                       """)

        company_texts = [row[0] for row in cursor.fetchall()]
        conn.close()

        total_companies = len(company_texts)
        print(f"Creating embeddings for {total_companies:,} companies using enhanced text")

        # Create embeddings in batches
        batch_size = 1000
        all_embeddings = []

        for i in range(0, total_companies, batch_size):
            batch_texts = company_texts[i:i + batch_size]

            # Generate embeddings
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32
            ).astype(np.float32)

            all_embeddings.append(batch_embeddings)

            if i % 10000 == 0:
                self.profiler.profile_checkpoint(f"Enhanced Embeddings Created", i + len(batch_texts))
                gc.collect()

        # Combine all embeddings
        self.company_embeddings = np.vstack(all_embeddings)

        # Save to disk
        with h5py.File(self.embeddings_file, 'w') as f:
            f.create_dataset('embeddings', data=self.company_embeddings, compression='gzip')

        print(f"✅ Saved {self.company_embeddings.shape} enhanced embeddings to {self.embeddings_file}")
        self.profiler.profile_checkpoint("Enhanced Company Embeddings Complete", total_companies)


    def build_hnswlib_index(self) -> None:
        """Build HNSWlib index with progressive construction for large datasets"""
        print("🚀 Building HNSWlib index with progressive construction...")

        if self.company_embeddings is None:
            with h5py.File(self.embeddings_file, 'r') as f:
                self.company_embeddings = f['embeddings'][:]

        num_elements, dimension = self.company_embeddings.shape
        print(f"Creating HNSWlib index for {num_elements:,} vectors with dimension {dimension}")

        # For large datasets (>75k), use progressive building approach
        if num_elements > 75000:
            print("🔧 Large dataset detected - using progressive index construction")
            self._build_large_dataset_index(num_elements, dimension)
        else:
            self._build_standard_index(num_elements, dimension)

    def _build_large_dataset_index(self, num_elements: int, dimension: int) -> None:
        """Progressive index building for large datasets to prevent memory issues"""

        # Step 1: Build initial smaller index
        initial_size = 25000  # Start with 25k vectors
        print(f"🔧 Step 1: Building initial index with {initial_size:,} vectors")

        # Very conservative parameters for large datasets
        self.hnswlib_index = hnswlib.Index(space='cosine', dim=dimension)

        ###=== TWEAK THIS FOR BETTER RESULTS IN THE BIG DATA CSVs (Better Accuracy) ===###
        ###=== MOVED TO CONFIG PARAMETERS IN MAIN FUNCTION ===###
        # Use much more conservative parameters
        conservative_M = self.conservative_M  # Much lower than 512
        conservative_ef_construction = self.conservative_ef_construction  # Lower than 600

        self.hnswlib_index.init_index(
            max_elements=num_elements,  # Full capacity but start small
            ef_construction=conservative_ef_construction,
            M=conservative_M
        )

        print(f"Using conservative parameters: M={conservative_M}, ef_construction={conservative_ef_construction}")

        # Add initial batch
        initial_embeddings = self.company_embeddings[:initial_size]
        initial_ids = list(range(initial_size))

        self.hnswlib_index.add_items(initial_embeddings, initial_ids)
        print(f"✅ Initial index built with {initial_size:,} vectors")

        # Step 2: Progressive expansion in small chunks
        chunk_size = 5000  # Very small chunks for stability

        for start_idx in range(initial_size, num_elements, chunk_size):
            end_idx = min(start_idx + chunk_size, num_elements)
            chunk_embeddings = self.company_embeddings[start_idx:end_idx]
            chunk_ids = list(range(start_idx, end_idx))

            progress = (end_idx / num_elements) * 100
            print(
                f"   📊 Adding chunk: {start_idx:,}-{end_idx - 1:,} ({progress:.1f}%) - {len(chunk_embeddings)} vectors")

            try:
                # Force memory cleanup before each chunk
                gc.collect()

                # Add chunk with error handling
                self.hnswlib_index.add_items(chunk_embeddings, chunk_ids)

                # Save checkpoint every 25k vectors
                if (end_idx - initial_size) % 25000 == 0:
                    checkpoint_file = str(self.index_file).replace('.bin', f'_checkpoint_{end_idx}.bin')
                    self.hnswlib_index.save_index(checkpoint_file)
                    print(f"   💾 Checkpoint saved at {end_idx:,} vectors")

                    # Verify index integrity
                    current_count = self.hnswlib_index.get_current_count()
                    if current_count != end_idx:
                        raise Exception(f"Index corruption: expected {end_idx}, got {current_count}")

            except Exception as e:
                print(f"❌ Error adding chunk {start_idx:,}-{end_idx - 1:,}: {e}")

                # Attempt recovery from last checkpoint
                checkpoint_files = list(self.work_dir.glob("*_checkpoint_*.bin"))
                if checkpoint_files:
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
                    print(f"🔄 Attempting recovery from {latest_checkpoint}")

                    # Reload from checkpoint and continue
                    self.hnswlib_index = hnswlib.Index(space='cosine', dim=dimension)
                    self.hnswlib_index.load_index(str(latest_checkpoint))
                    self.hnswlib_index.set_ef(self.ef_search)

                    # Continue from checkpoint
                    checkpoint_idx = int(latest_checkpoint.stem.split('_')[-1])
                    print(f"🔄 Resuming from index {checkpoint_idx:,}")
                    continue
                else:
                    raise

        # Final save
        print("💾 Saving final index...")
        self.hnswlib_index.save_index(str(self.index_file))

        # Set search parameters
        self.hnswlib_index.set_ef(self.ef_search)

        # Cleanup checkpoint files
        for checkpoint_file in self.work_dir.glob("*_checkpoint_*.bin"):
            checkpoint_file.unlink()

        print(f"✅ Progressive index construction complete!")

    def _build_standard_index(self, num_elements: int, dimension: int) -> None:
        """Standard index building for smaller datasets"""
        self.hnswlib_index = hnswlib.Index(space='cosine', dim=dimension)

        self.hnswlib_index.init_index(
            max_elements=num_elements,
            ef_construction=self.ef_construction,
            M=self.M
        )

        self.hnswlib_index.set_ef(self.ef_search)

        # Standard batch processing
        batch_size = 10000
        for i in range(0, num_elements, batch_size):
            end_idx = min(i + batch_size, num_elements)
            batch_embeddings = self.company_embeddings[i:end_idx]
            batch_ids = list(range(i, end_idx))

            print(f"   📊 Adding batch: {i:,}-{end_idx - 1:,}")
            self.hnswlib_index.add_items(batch_embeddings, batch_ids)

        # Save index
        self.hnswlib_index.save_index(str(self.index_file))
        print(f"✅ Standard index construction complete!")

    def load_hnswlib_index(self) -> None:
        """Load HNSWlib index from disk"""
        if self.index_file.exists():
            print(f"📂 Loading HNSWlib index from {self.index_file}")

            # Load embeddings to get dimension
            if self.company_embeddings is None:
                with h5py.File(self.embeddings_file, 'r') as f:
                    self.company_embeddings = f['embeddings'][:]

            dimension = self.company_embeddings.shape[1]

            # Initialize and load index
            self.hnswlib_index = hnswlib.Index(space='cosine', dim=dimension)
            self.hnswlib_index.load_index(str(self.index_file))

            # Set search parameters
            self.hnswlib_index.set_ef(self.ef_search)

            print(f"✅ Loaded HNSWlib index with {self.hnswlib_index.get_current_count():,} vectors")
            print("🚀 HNSWlib ready for ultra-fast similarity search!")
        else:
            raise FileNotFoundError("HNSWlib index not found")

    def load_company_cache(self) -> None:
        """Load all company data into memory cache including normalized text"""
        if self.cache_loaded:
            return

        print("🚀 Loading enhanced company data into memory cache...")

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Load all company data including normalized text
        cursor.execute("""
            SELECT embedding_index, original_name, cleaned_text, normalized_text, cities 
            FROM companies 
            ORDER BY embedding_index
        """)

        rows = cursor.fetchall()

        # Build memory cache
        for row in rows:
            embedding_idx, original_name, cleaned_text, normalized_text, cities = row
            self.company_cache[embedding_idx] = {
                'original_name': original_name,
                'cleaned_text': cleaned_text,
                'normalized_text': normalized_text or cleaned_text,  # Fallback to cleaned if normalized is empty
                'cities': cities.split('|') if cities else []
            }

        conn.close()
        self.cache_loaded = True

        print(f"✅ Cached {len(self.company_cache):,} companies with enhanced data")
        self.profiler.profile_checkpoint("Enhanced Company Cache Loaded", len(self.company_cache))

    def hnswlib_similarity_search_ultra_optimized(self, query_embeddings: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-optimized HNSWlib search with maximum performance settings"""
        if self.hnswlib_index is None:
            raise ValueError("HNSWlib index not initialized")

        # Ensure contiguous memory layout
        if not query_embeddings.flags['C_CONTIGUOUS']:
            query_embeddings = np.ascontiguousarray(query_embeddings, dtype=np.float32)

        # Set maximum ef for ultra-fast search (trade accuracy for speed if needed)
        original_ef = self.ef_search

        # Aggressive ef settings based on batch size
        batch_size = query_embeddings.shape[0]
        if batch_size > 5000:
            optimal_ef = min(self.ef_search * 3, 500)  # Much higher for large batches
        elif batch_size > 1000:
            optimal_ef = min(self.ef_search * 2, 400)
        else:
            optimal_ef = self.ef_search

        self.hnswlib_index.set_ef(optimal_ef)

        try:
            # Ultra-fast search with all CPU cores
            labels, distances = self.hnswlib_index.knn_query(
                query_embeddings,
                k=k,
                num_threads=-1  # Use all available CPU cores
            )

            # Vectorized distance to similarity conversion
            similarities = 1.0 - distances

            return similarities, labels

        finally:
            # Restore original ef
            self.hnswlib_index.set_ef(original_ef)

    def calculate_improved_final_score(self, hnswlib_sim: float, fuzzy_score: float, city_boost: float) -> float:
        """
        Improved scoring system with proper weighting and normalization
        City boost is now much more conservative to prevent location-only matches

        Scoring Logic:
        1. Convert hnswlib similarity (0-1) to percentage (0-100)
        2. Weight embedding similarity based on confidence
        3. Combine with fuzzy score using intelligent weighting
        4. Add SMALL city boost as supporting evidence only
        5. Require minimum company name similarity before city boost helps significantly
        """
        # Convert hnswlib similarity to percentage (0-100)
        embedding_score = hnswlib_sim * 100.0

        # Intelligent weighting based on confidence levels
        if embedding_score >= 80:
            # High embedding confidence: weight embedding more heavily
            embedding_weight = 0.7
            fuzzy_weight = 0.3
        elif embedding_score >= 60:
            # Medium embedding confidence: balanced weighting
            embedding_weight = 0.5
            fuzzy_weight = 0.5
        else:
            # Low embedding confidence: weight fuzzy matching more heavily
            embedding_weight = 0.3
            fuzzy_weight = 0.7

        # Calculate weighted base score
        base_score = (embedding_score * embedding_weight) + (fuzzy_score * fuzzy_weight)

        # NEW: Apply city boost more intelligently
        # Only give significant city boost if there's already reasonable company similarity
        if base_score >= 60:
            # Good base match - city boost can help
            final_score = base_score + city_boost
        elif base_score >= 40:
            # Moderate base match - reduced city boost
            final_score = base_score + (city_boost * 0.5)
        else:
            # Poor base match - minimal city boost (prevent false positives)
            final_score = base_score + (city_boost * 0.2)

        # Cap at 100 for cleaner interpretation
        return min(100.0, final_score)

    def process_transaction_batch_ultra_optimized(self, transactions_batch: List[str], batch_id: int) -> List[Dict]:
        """Ultra-optimized batch processing with improved scoring and enhanced matching"""
        batch_start_time = time.time()
        results = []

        # Ensure company cache is loaded
        if not self.cache_loaded:
            self.load_company_cache()

        # Step 1: Ultra-fast preprocessing with enhanced normalization
        preprocess_start = time.time()
        processed_txns = [self.preprocess_text_ultra_fast_txn(txn) for txn in transactions_batch]

        # Use normalized text for better matching
        txn_texts = [p['normalized'] if p['normalized'] else p['cleaned'] for p in processed_txns]
        print(f"   ⚡ Preprocessing: {time.time() - preprocess_start:.2f}s")

        # Step 2: Aggressive embedding batching
        embed_start = time.time()
        embedding_batch_size = min(128, len(txn_texts))

        txn_embeddings = self.embedding_model.encode(
            txn_texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=embedding_batch_size,
            normalize_embeddings=False,
            convert_to_tensor=False
        ).astype(np.float32)
        print(f"   🧠 Embeddings: {time.time() - embed_start:.2f}s")

        # Step 3: Ultra-fast similarity search
        search_start = time.time()
        similarities, candidate_indices = self.hnswlib_similarity_search_ultra_optimized(
            txn_embeddings,
            self.max_candidates
        )
        print(f"   🔍 Search: {time.time() - search_start:.2f}s")

        # Step 4: Enhanced result processing with improved scoring
        process_start = time.time()

        # Pre-compute all company data in vectorized fashion
        all_company_data = []
        for i in range(len(transactions_batch)):
            candidate_ids = candidate_indices[i]
            candidate_sims = similarities[i]

            # More lenient similarity threshold for initial filtering
            valid_indices = np.where(candidate_sims >= 0.3)[0]  # Reduced from 0.5 to 0.3

            if len(valid_indices) == 0:
                all_company_data.append([])
                continue

            batch_company_data = []
            for idx in valid_indices:
                company_idx = int(candidate_ids[idx])
                similarity = float(candidate_sims[idx])
                company_data = self.company_cache.get(company_idx)

                if company_data:
                    batch_company_data.append((similarity, company_data))

            all_company_data.append(batch_company_data)

        # Enhanced fuzzy matching and scoring
        for i, (txn_text, processed_txn) in enumerate(zip(transactions_batch, processed_txns)):
            company_data_list = all_company_data[i]

            if not company_data_list:
                # No valid candidates
                result = {
                    'transaction_text': txn_text,
                    'company_name': 'NO_MATCH',
                }

                # Add optional columns based on configuration
                if self.output_columns.get('hnswlib_similarity', True):
                    result['hnswlib_similarity'] = 0.0
                if self.output_columns.get('fuzzy_score', True):
                    result['fuzzy_score'] = 0.0
                if self.output_columns.get('city_boost', True):
                    result['city_boost'] = 0.0
                if self.output_columns.get('final_score', True):
                    result['final_score'] = 0.0
                if self.output_columns.get('backend_used', True):
                    result['backend_used'] = 'hnswlib_ultra_optimized'
                if self.output_columns.get('txn_cities', True):
                    result['txn_cities'] = '|'.join(processed_txn['cities'])
                if self.output_columns.get('company_cities', True):
                    result['company_cities'] = ''
                if self.output_columns.get('detected_company_name', True):
                    result['detected_company_name'] = txn_text_for_matching

                results.append(result)
                continue

            # Find best match with enhanced fuzzy matching
            best_match = None
            best_score = 0.0

            # Use normalized text for better matching
            txn_text_for_matching = processed_txn['normalized'] if processed_txn['normalized'] else processed_txn['cleaned']
            txn_cities = processed_txn['cities']

            for similarity, company_data in company_data_list:
                company_normalized = company_data.get('normalized_text', company_data['cleaned_text'])
                company_cities = company_data['cities']

                # Enhanced fuzzy matching with multiple methods
                fuzzy_scores = [
                    fuzz.ratio(txn_text_for_matching, company_normalized),
                    fuzz.partial_ratio(txn_text_for_matching, company_normalized),
                    fuzz.token_sort_ratio(txn_text_for_matching, company_normalized),
                    fuzz.token_set_ratio(txn_text_for_matching, company_normalized)
                ]

                fuzzy_score = max(fuzzy_scores)

                # Enhanced city boost calculation
                city_boost = self.city_matcher.calculate_city_boost(txn_cities, company_cities)

                # Use improved scoring system
                final_score = self.calculate_improved_final_score(similarity, fuzzy_score, city_boost)

                if final_score > best_score:
                    best_score = final_score

                    result = {
                        'transaction_text': txn_text,
                        'company_name': company_data['original_name'],
                    }

                    # Add optional columns based on configuration
                    if self.output_columns.get('hnswlib_similarity', True):
                        result['hnswlib_similarity'] = round(similarity, 2)
                    if self.output_columns.get('fuzzy_score', True):
                        result['fuzzy_score'] = round(fuzzy_score, 2)
                    if self.output_columns.get('city_boost', True):
                        result['city_boost'] = round(city_boost, 2)
                    if self.output_columns.get('final_score', True):
                        result['final_score'] = round(final_score, 2)
                    if self.output_columns.get('backend_used', True):
                        result['backend_used'] = 'hnswlib_ultra_optimized'
                    if self.output_columns.get('txn_cities', True):
                        result['txn_cities'] = '|'.join(txn_cities)
                    if self.output_columns.get('company_cities', True):
                        result['company_cities'] = '|'.join(company_cities)
                    if self.output_columns.get('detected_company_name', True):
                        # Apply additional filtering to remove bank names and personal names
                        filtered_company_name = self.transaction_cleaner.filter_detected_company_name(txn_text_for_matching)
                        result['detected_company_name'] = filtered_company_name

                    best_match = result

            results.append(best_match)

        print(f"   📊 Processing: {time.time() - process_start:.2f}s")
        print(f"   🎯 Total batch time: {time.time() - batch_start_time:.2f}s")

        return results

    def match_transactions_parallel(self, transactions_file: str, sort_by_score: bool = True) -> str:
        """Parallel transaction matching process using configurable workers"""
        print(f"🎯 Starting PARALLEL transaction matching from {transactions_file}")
        print(f"🔀 Using {self.parallel_workers} parallel workers with {self.parallel_method} method")

        # Load transactions with polars
        try:
            # Determine file type and read accordingly
            if transactions_file.lower().endswith('.csv'):
                transactions_df = pl.read_csv(transactions_file)
            else:
                transactions_df = pl.read_excel(transactions_file)
            total_transactions = len(transactions_df)

            print(f"Loaded {total_transactions} Transactions records from {transactions_file}")
        except Exception as e:
            print(f"❌ Error loading transactions file: {e}")
            return ""

        # Auto-detect or use specified transaction column
        if self.transaction_column and len(self.transaction_column) > 0:
            txn_col_name = self.transaction_column[0]
            if txn_col_name not in transactions_df.columns:
                print(f"⚠️  Specified transaction column '{txn_col_name}' not found. Available columns: {transactions_df.columns}")
                # Look for common transaction column names
                common_txn_cols = ['Narration', 'Description', 'Transaction', 'Details', 'Memo']
                txn_col_name = None
                for col in common_txn_cols:
                    if col in transactions_df.columns:
                        txn_col_name = col
                        break
                if not txn_col_name:
                    txn_col_name = transactions_df.columns[0]  # Fallback to first column
                print(f"Using fallback column: '{txn_col_name}'")
        else:
            # Auto-detect transaction column
            common_txn_cols = ['Narration', 'Description', 'Transaction', 'Details', 'Memo']
            txn_col_name = None
            for col in common_txn_cols:
                if col in transactions_df.columns:
                    txn_col_name = col
                    break
            if not txn_col_name:
                txn_col_name = transactions_df.columns[0]  # Fallback to first column
            print(f"Auto-detected transaction column: '{txn_col_name}'")

        # Handle batch_size=None for parallel processing - split equally among workers
        if self.batch_size is None:
            effective_batch_size = max(1, total_transactions // self.parallel_workers)
            print(
                f"📦 batch_size=None detected - splitting {total_transactions:,} transactions equally among {self.parallel_workers} workers")
            print(f"📦 Calculated batch size: {effective_batch_size:,} per worker")
        else:
            effective_batch_size = self.batch_size
        print(f"Processing {total_transactions:,} transactions in parallel batches of {self.batch_size:,}")
        print(f"📋 Using transaction column: '{txn_col_name}'")

        # Pre-load company cache for ultra-fast access
        self.load_company_cache()

        # Prepare batches for parallel processing
        batch_info_list = []
        for batch_start in range(0, total_transactions, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_transactions)
            batch_id = batch_start // self.batch_size + 1

            # Extract batch using the correct column
            batch_df = transactions_df[batch_start:batch_end]
            transactions_batch = batch_df[txn_col_name].to_list()

            batch_info_list.append({
                'batch_id': batch_id,
                'transactions_batch': transactions_batch,
                'batch_start': batch_start,
                'batch_end': batch_end
            })

        print(f"📦 Created {len(batch_info_list)} batches for parallel processing")

        # Process batches in parallel
        all_results = []

        # Choose executor based on parallel method
        if self.parallel_method == 'thread':
            ExecutorClass = ThreadPoolExecutor
            print("🧵 Using ThreadPoolExecutor for parallel processing")
        else:
            ExecutorClass = ProcessPoolExecutor
            print("🔄 Using ProcessPoolExecutor for parallel processing")

        # Create worker function with bound context
        def process_batch_worker(batch_info):
            """Worker function for parallel batch processing"""
            try:
                batch_id = batch_info['batch_id']
                transactions_batch = batch_info['transactions_batch']
                batch_start = batch_info['batch_start']
                batch_end = batch_info['batch_end']

                print(f"\n🔀 [Worker] Processing batch {batch_id}: transactions {batch_start:,}-{batch_end-1:,}")

                # Process the batch
                batch_results = self.process_transaction_batch_ultra_optimized(transactions_batch, batch_id)

                print(f"✅ [Worker] Completed batch {batch_id} with {len(batch_results)} results")
                return batch_results

            except Exception as e:
                print(f"❌ [Worker] Error processing batch {batch_info.get('batch_id', 'unknown')}: {e}")
                return []

        # Execute parallel processing
        parallel_start_time = time.time()

        with ExecutorClass(max_workers=self.parallel_workers) as executor:
            print(f"🚀 Launching {len(batch_info_list)} batches across {self.parallel_workers} workers...")

            # Submit all tasks
            future_to_batch = {
                executor.submit(process_batch_worker, batch_info): batch_info
                for batch_info in batch_info_list
            }

            # Collect results as they complete
            completed_batches = 0
            for future in as_completed(future_to_batch):
                batch_info = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                    completed_batches += 1

                    print(f"📊 Progress: {completed_batches}/{len(batch_info_list)} batches completed "
                          f"({completed_batches/len(batch_info_list)*100:.1f}%)")

                except Exception as e:
                    print(f"❌ Batch {batch_info['batch_id']} failed: {e}")

        parallel_time = time.time() - parallel_start_time
        print(f"⚡ Parallel processing completed in {parallel_time:.2f} seconds")
        print(f"🏆 Processed {total_transactions:,} transactions using {self.parallel_workers} workers")
        self.profiler.profile_checkpoint("Parallel Transaction Processing Complete")

        # Sort results by final score if requested
        if sort_by_score and all_results:
            print("📊 Sorting results by final score (descending)...")
            if self.output_columns.get('final_score', True):
                all_results.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
            else:
                print("⚠️  Cannot sort by final_score - column disabled in output")

        # Write all results to file
        if all_results:
            results_df = pl.DataFrame(all_results)
            results_df.write_csv(self.results_file)
            print(f"✅ Results sorted and saved to {self.results_file}")
        else:
            print("❌ No results to save")

        # Print automated city detection summary
        if self.city_matcher and self.enable_automated_city_detection:
            discovered_count = self.city_matcher.get_discovered_locations_count()
            print(f"🏙️  Automatically discovered {discovered_count} locations during processing")

        print(f"📊 Total matches processed: {len(all_results):,}")
        return str(self.results_file)

    def match_transactions(self, transactions_file: str, sort_by_score: bool = True) -> str:
        """Main transaction matching process with optional sorting"""
        print(f"🎯 Starting transaction matching from {transactions_file}")

        # Load transactions with polars
        try:
            # Determine file type and read accordingly
            if transactions_file.lower().endswith('.csv'):
                transactions_df = pl.read_csv(transactions_file)
            else:
                transactions_df = pl.read_excel(transactions_file)
            total_transactions = len(transactions_df)

            print(f"Loaded {total_transactions} Transactions records from {transactions_file}")
        except Exception as e:
            print(f"❌ Error loading transactions file: {e}")
            return ""

        # Auto-detect or use specified transaction column
        if self.transaction_column and len(self.transaction_column) > 0:
            txn_col_name = self.transaction_column[0]
            if txn_col_name not in transactions_df.columns:
                print(f"⚠️  Specified transaction column '{txn_col_name}' not found. Available columns: {transactions_df.columns}")
                # Look for common transaction column names
                common_txn_cols = ['Narration', 'Description', 'Transaction', 'Details', 'Memo']
                txn_col_name = None
                for col in common_txn_cols:
                    if col in transactions_df.columns:
                        txn_col_name = col
                        break
                if not txn_col_name:
                    txn_col_name = transactions_df.columns[0]  # Fallback to first column
                print(f"Using fallback column: '{txn_col_name}'")
        else:
            # Auto-detect transaction column
            common_txn_cols = ['Narration', 'Description', 'Transaction', 'Details', 'Memo']
            txn_col_name = None
            for col in common_txn_cols:
                if col in transactions_df.columns:
                    txn_col_name = col
                    break
            if not txn_col_name:
                txn_col_name = transactions_df.columns[0]  # Fallback to first column
            print(f"Auto-detected transaction column: '{txn_col_name}'")

        # Handle batch_size=None for sequential processing - use default 50000
        if self.batch_size is None:
            effective_batch_size = 50000
            print(f"📦 batch_size=None detected - using default batch size: {effective_batch_size:,}")
        else:
            effective_batch_size = self.batch_size

        print(f"Processing {total_transactions:,} transactions in batches of {self.batch_size:,}")
        print(f"📋 Using transaction column: '{txn_col_name}'")

        # Pre-load company cache for ultra-fast access
        self.load_company_cache()

        # Process all transactions and collect results
        all_results = []

        # Process in batches using ultra-optimized method
        for batch_start in range(0, total_transactions, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_transactions)
            batch_id = batch_start // self.batch_size + 1

            print(f"\n📦 Processing batch {batch_id}: transactions {batch_start:,}-{batch_end-1:,}")

            # Extract batch using the correct column
            batch_df = transactions_df[batch_start:batch_end]
            transactions_batch = batch_df[txn_col_name].to_list()

            # Process batch using ultra-optimized method
            batch_results = self.process_transaction_batch_ultra_optimized(transactions_batch, batch_id)
            all_results.extend(batch_results)

            # Profile every 10k records
            if batch_end % 10000 == 0 or batch_end == total_transactions:
                self.profiler.profile_checkpoint(f"Batch {batch_id} Complete", batch_end)
                gc.collect()

        # Sort results by final score if requested
        if sort_by_score and all_results:
            print("📊 Sorting results by final score (descending)...")

            # Handle cases where final_score might not be in output
            if self.output_columns.get('final_score', True):
                all_results.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
            else:
                print("⚠️  Cannot sort by final_score - column disabled in output")

        # Write all results to file
        if all_results:
            results_df = pl.DataFrame(all_results)
            results_df.write_csv(self.results_file)
            print(f"✅ Results sorted and saved to {self.results_file}")
        else:
            print("❌ No results to save")

        # Print automated city detection summary
        if self.city_matcher and self.enable_automated_city_detection:
            discovered_count = self.city_matcher.get_discovered_locations_count()
            print(f"🏙️  Automatically discovered {discovered_count} locations during processing")

        print(f"📊 Total matches processed: {len(all_results):,}")
        return str(self.results_file)

    def run_full_pipeline(self, companies_file: str, transactions_file: str, sort_by_score: bool = True) -> str:
        """Run the complete matching pipeline with enhanced features"""
        start_time = time.time()

        print("🚀 Starting Ultra-Efficient Entity Matching Pipeline with Enhanced Features")
        print(f"🏙️  Automated City Detection: {self.enable_automated_city_detection}")
        print(f"📊 Batch Size: {self.batch_size:,}")
        print(f"🎯 Confidence Threshold: {self.confidence_threshold}")
        print(f"🔧 HNSWlib Parameters: ef_construction={self.ef_construction}, M={self.M}, ef_search={self.ef_search}")
        print(f"📋 Output Columns: {[k for k, v in self.output_columns.items() if v]}")
        print(f"📈 Sort by Score: {sort_by_score}")
        print("=" * 80)

        try:
            # Step 1: Initialize models
            self.initialize_models()

            # Step 2: Check if preprocessing needed
            if not self.db_file.exists() or not self.embeddings_file.exists():
                print("\n🔧 Company preprocessing needed...")

                # Preprocess companies
                company_count = self.load_and_preprocess_companies(companies_file)

                # Create embeddings using normalized text
                self.create_company_embeddings_enhanced()

                # Build HNSWlib index
                self.build_hnswlib_index()
            else:
                print("\n📂 Loading existing company data...")
                if self.index_file.exists():
                    self.load_hnswlib_index()
                else:
                    print("🔧 HNSWlib index not found, building new index...")
                    if self.company_embeddings is None:
                        with h5py.File(self.embeddings_file, 'r') as f:
                            self.company_embeddings = f['embeddings'][:]
                    self.build_hnswlib_index()

            # Step 3: Match transactions with enhanced processing
            # Choose between parallel and sequential processing
            if self.enable_parallel_processing:
                print(f"🔀 PARALLEL PROCESSING ENABLED - Using {self.parallel_workers} {self.parallel_method} workers")
                results_file = self.match_transactions_parallel(transactions_file, sort_by_score=sort_by_score)
            else:
                print("📈 Using sequential processing (Single Threaded)")
                results_file = self.match_transactions(transactions_file, sort_by_score=sort_by_score)

            # Step 4: Save discovered locations for future use
            if self.city_matcher and self.enable_automated_city_detection:
                self.city_matcher.finalize_session()

            # Step 5: Final summary
            total_time = time.time() - start_time
            print(f"\n🏁 Enhanced Pipeline Complete!")
            print(f"🔧 Backend Used: HNSWlib Enhanced (M={self.M})")
            print(f"   Results: {results_file}")

            if self.city_matcher and self.enable_automated_city_detection:
                discovered_locations = self.city_matcher.get_discovered_locations()
                #print(f"🏙️  Discovered Locations: {len(discovered_locations)} cities/towns automatically detected")
                if len(discovered_locations) > 0:
                    sample_locations = list(discovered_locations)[:10]
                    # print(f"📍 Sample Locations: {', '.join(sample_locations)}")

                # Show location of persistent storage
                #print(f"📂 Discovered locations saved to: {self.city_matcher.discovered_locations_file}")

            return results_file

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            raise
        finally:
            # Cleanup
            if hasattr(self, 'nlp'):
                del self.nlp
            if hasattr(self, 'embedding_model'):
                del self.embedding_model
            gc.collect()


def main():
    """Main function with enhanced configuration options"""
    # OUTPUT COLUMNS CONFIGURATION:
    # Set any of these to False to exclude from output:
    # - 'backend_used': Shows which similarity backend was used
    # - 'txn_cities': Shows cities detected in transaction text
    # - 'company_cities': Shows cities detected in company name
    # - 'hnswlib_similarity': Shows embedding similarity (0-1)
    # - 'fuzzy_score': Shows string similarity (0-100)
    # - 'city_boost': Shows location matching bonus (0-20)
    # - 'final_score': Shows combined weighted score (0-100)

    # AUTOMATED CITY DETECTION:
    # - True: Uses NER and pattern matching to automatically detect cities/locations (Disabling not recommended)

    # HNSWlib PARAMETERS:
    # - M: Number of connections per node (higher = faster search, more memory)
    #   Default: 64, Recommended for high RAM: 512-1024
    # - ef_construction: Build quality (higher = better index, slower build)
    #   Range: 100-2000, Recommended: 600
    # - ef_search: Search quality (higher = better accuracy, slightly slower)
    #   Range: 50-1000, Recommended: 200-400

    # SCORING SYSTEM EXPLANATION:
    # The improved scoring system works as follows:
    # 1. hnswlib_similarity (0-1) to percentage (0-100)
    # 2. fuzzy_score (0-100): String similarity using multiple algorithms
    # 3. city_boost (0-20): Bonus for location matching
    # 4. final_score: Intelligently weighted combination:
    #    - High embedding confidence: 70% embedding + 30% fuzzy + city boost
    #    - Medium embedding confidence: 50% embedding + 50% fuzzy + city boost
    #    - Low embedding confidence: 30% embedding + 70% fuzzy + city boost

    # Enhanced Configuration with new features
    CONFIG = {
        # 'batch_size': 62501,  # Adjust based on available memory
        'batch_size': 7000,  # Adjust based on available memory
        'confidence_threshold': 50.0,  # Lowered threshold for better matching
        'max_candidates': 10,  # Top-K candidates from similarity search
        'embedding_model': "paraphrase-MiniLM-L6-v2",  # Fast, efficient model

        'M': 512,  # Increased from 64 to 512 for faster search (uses more RAM)
        'ef_construction': 600,  # Index Building quality parameter
        'ef_search': 1000,  # Search quality parameter #200
        'conservative_M': 64, # For data greater than 250,000 rows, use conservative M for better memory management
        'conservative_ef_construction': 300,

        'enable_automated_city_detection': True,  # Enable automated city/location detection

        # PARALLEL PROCESSING CONFIGURATION - NEW FEATURE!
        'enable_parallel_processing': True,  # Enable/disable parallel batch processing
        'parallel_workers': 8,  # Number of parallel workers (start with 2, can increase to 4, 6, 8)
        'parallel_method': 'thread',  # 'thread' or 'process' - thread is safer for this use case

        # TRANSACTION FILTERING CONFIGURATION - NEW FEATURE!
        'enable_transaction_filtering': False,  # Enable intelligent transaction filtering (removes noise, banks, account numbers)

        # Configure which columns to include in output (set to False to exclude)
        'output_columns': {
            'backend_used': False,  # Remove backend_used column
            'txn_cities': False,  # Keep transaction cities
            'company_cities': False,  # Keep company cities
            'hnswlib_similarity': False,  # Keep embedding similarity score
            'fuzzy_score': False,  # Keep fuzzy matching score
            'city_boost': False,  # Keep city boost score
            'final_score': True,  # Keep final combined score
            'detected_company_name': False  # Shows cleaned/filtered transaction text used for matching
        },
        # 'company_column': ['Company_Name'],  # Specify column name for company data
        # 'transaction_column': ['Narration'],  # Specify column name for transaction data
        # === For File One === #
        'company_column': ['Narrations'],
        'transaction_column': ['Transaction Particulars'],
    }

    COMPANIES_FILE = "2nd_Companyname_excel_10k.csv"
    TRANSACTIONS_FILE = "2nd_Companytrn_excel_10k.csv"
    # COMPANIES_FILE = "4th_company_master_0.5lakh.csv"
    # TRANSACTIONS_FILE = "4th_bank_transactions_0.5lakh.csv"

    print(f"\n🔧 Enhanced Configuration:")
    for key, value in CONFIG.items():
        if key == 'output_columns':
            enabled_cols = [k for k, v in value.items() if v]
            disabled_cols = [k for k, v in value.items() if not v]
            print(f"  📋 Output Columns Enabled: {enabled_cols}")
            if disabled_cols:
                print(f"  📋 Output Columns Disabled: {disabled_cols}")
        else:
            print(f"  {key}: {value}")
    print()

    # Check if files exist
    if not Path(COMPANIES_FILE).exists():
        print(f"❌ Companies file not found: {COMPANIES_FILE}")
        print("📝 Available files in directory:")
        for file in Path(".").glob("*.csv"):
            print(f"  - {file.name}")
        return

    if not Path(TRANSACTIONS_FILE).exists():
        print(f"❌ Transactions file not found: {TRANSACTIONS_FILE}")
        print("📝 Available files in directory:")
        for file in Path(".").glob("*.csv"):
            print(f"  - {file.name}")
        return

    # Initialize matcher with enhanced configuration
    matcher = EfficientEntityMatcher(**CONFIG)

    # Run pipeline with sorting enabled by default
    try:
        results_file = matcher.run_full_pipeline(
            COMPANIES_FILE,
            TRANSACTIONS_FILE,
            sort_by_score=True  # Sort results by final score (highest first)
        )

    # if CONFIG['enable_automated_city_detection']:
        #     print(f"✅ Will automatically discover new cities/locations during processing")

    except KeyboardInterrupt:
        print("\n⏹️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
