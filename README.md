# 🏦 Vector Embedding Transaction Matcher for VSolv

**An AI-powered financial transaction matching system** that uses advanced vector embeddings and intelligent fuzzy matching to automatically match bank transaction narrations with company master data. Built specifically for financial institutions and accounting firms to automate transaction categorization and company identification.

## 🎯 Project Overview

This system solves the critical problem of **automatically matching bank transaction descriptions with company master records**. Traditional keyword-based matching fails due to variations in transaction formats, abbreviations, and noise in banking data. Our solution uses:

- **🧠 Semantic Understanding**: Vector embeddings capture meaning beyond exact text matches
- **⚡ Ultra-Fast Search**: HNSWlib enables millisecond searches across millions of records  
- **🏙️ Location Intelligence**: Automated city/location detection for enhanced matching confidence
- **🧹 Smart Filtering**: Intelligent removal of banking noise while preserving company names
- **🔄 Parallel Processing**: Multi-threaded processing for enterprise-scale datasets

---

## ✨ Key Features

### 🎯 **Core Matching Engine**
- **Vector Embeddings**: Uses `sentence-transformers` for semantic similarity
- **High-Speed Search**: HNSWlib with optimized parameters for sub-second results
- **Multi-Algorithm Fuzzy Matching**: RapidFuzz with 4 different similarity algorithms
- **Intelligent Scoring**: Combines embedding similarity, fuzzy scores, and location matching

### 🏙️ **Advanced Location Detection**
- **Automated City Recognition**: Uses spaCy NER to detect cities/locations in text
- **Indian Geographic Intelligence**: Pre-built knowledge of Indian cities, towns, and localities
- **Persistent Location Learning**: Automatically discovers and saves new locations for future use
- **Location-Based Confidence Boost**: Enhances matching accuracy when locations align

### 🧹 **Smart Transaction Processing**
- **Banking Noise Removal**: Automatically filters out NEFT/RTGS/UPI prefixes, account numbers
- **Bank Name Intelligence**: Recognizes and filters 100+ Indian bank names
- **Company Name Preservation**: Protects legitimate business indicators (Pvt Ltd, Inc, etc.)
- **Transaction Code Cleanup**: Removes alphanumeric transaction IDs and reference numbers

### ⚡ **Performance & Scalability**
- **Parallel Processing**: Configurable multi-threading with both thread and process pools
- **Memory Optimization**: Progressive index building for large datasets (>250K records)
- **Batch Processing**: Configurable batch sizes with memory profiling
- **Disk-Based Storage**: Efficient HDF5 storage for embeddings and SQLite for metadata

---

## 📋 Data Format Requirements

### **Company Master File** (`companies.csv`)
```csv
Company_ID,Company_Name
EC00001,KUMARAN HARDWARE
EC00002,GANESHAGENCIES CHENNAI
EC00003,INDIRA DINDIGUL
EC00004,RAMCOSYSTEMS CUDDALORE
```

### **Bank Transactions File** (`transactions.csv`)
```csv
Transaction_ID,Date,Amount,Type,Narration,Account_No
TXN000001,2024-05-26,33100.52,CREDIT,NEFT/IQOV94360237845587/OMERODE,9357913997
TXN000002,2024-02-27,6945.71,CREDIT,IMPS/FIGP38418043050302/RAMCO VELLORE,4010690408
TXN000003,2024-01-12,11235.38,CREDIT,UPI/NUCF28468607902813/RAMCOSYSTEMS VELLORE,8049807388
```

**Supported Formats**: CSV, Excel (.xlsx, .xls)
**Auto-Detection**: The system automatically detects company and transaction columns

---

## 🛠️ Installation & Setup

### **1. System Requirements**
```bash
- Python 3.8+
- 8GB+ RAM (16GB recommended for large datasets)
- 2GB free disk space for embeddings and indices
```

### **2. Install Dependencies**
```bash
# Clone the repository
git clone https://github.com/forsyth47/vectorEmbedding-TxnMatch-Vsolv.git
cd vectorEmbedding-TxnMatch-Vsolv

# Install Python packages
pip install -r requirements-embedding.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### **3. Dependencies List**
```
numpy          # Numerical computations
pandas         # Data manipulation
polars         # Fast data processing
psutil         # System monitoring
h5py          # HDF5 file format for embeddings
spacy         # Natural Language Processing
sentence-transformers  # Vector embeddings
rapidfuzz     # Fast fuzzy string matching
hnswlib       # Approximate nearest neighbor search
```

---

## 🚀 Usage Guide

### **Quick Start**
```bash
# Basic usage with default files
python xEnd.py

# The system will look for:
# - Company file: Check CONFIG in main()
# - Transaction file: Check CONFIG in main()
```

### **Configuration Options**

Edit the `CONFIG` dictionary in `main()` function:

```python
CONFIG = {
    # 🔧 Processing Parameters
    'batch_size': 7000,                    # Records per batch
    'confidence_threshold': 50.0,          # Minimum matching score
    'max_candidates': 10,                  # Top-K similar companies
    
    # 🧠 AI Model Settings  
    'embedding_model': "paraphrase-MiniLM-L6-v2",  # Sentence transformer model
    
    # ⚡ HNSWlib Performance (for datasets > 75K)
    'M': 512,                             # Index connectivity (higher = faster)
    'ef_construction': 600,               # Build quality (higher = better)
    'ef_search': 1000,                    # Search quality (higher = more accurate)
    
    # 🏗️ Large Dataset Settings (> 250K records)
    'conservative_M': 64,                 # Lower memory usage
    'conservative_ef_construction': 300,   # Faster building
    
    # 🌆 Enhanced Features
    'enable_automated_city_detection': True,     # Location intelligence
    'enable_transaction_filtering': True,        # Smart text cleaning
    
    # 🔄 Parallel Processing
    'enable_parallel_processing': True,          # Multi-threading
    'parallel_workers': 8,                       # Number of workers
    'parallel_method': 'thread',                 # 'thread' or 'process'
    
    # 📊 Output Customization
    'output_columns': {
        'final_score': True,                     # Combined matching score
        'hnswlib_similarity': True,              # Embedding similarity
        'fuzzy_score': True,                     # String similarity
        'city_boost': True,                      # Location bonus
        'detected_company_name': True,           # Cleaned transaction text
    },
    
    # 📁 Column Mapping (optional)
    'company_column': ['Company_Name'],          # Company data column
    'transaction_column': ['Narration'],         # Transaction text column
}
```

### **File Configuration**
```python
# Set your input files
COMPANIES_FILE = "your_company_master.csv"
TRANSACTIONS_FILE = "your_bank_transactions.csv"
```

---

## 📊 Understanding the Output

### **Output Columns Explained**

| Column | Description | Range | Example |
|--------|-------------|-------|---------|
| `transaction_text` | Original transaction narration | - | "NEFT/ABC123/RAMCO SYSTEMS" |
| `company_name` | Matched company name | - | "RAMCOSYSTEMS CUDDALORE" |
| `final_score` | **Combined matching confidence** | 0-100 | 87.5 |
| `hnswlib_similarity` | Semantic embedding similarity | 0-1 | 0.82 |
| `fuzzy_score` | String similarity score | 0-100 | 85.0 |
| `city_boost` | Location matching bonus | 0-20 | 3.0 |
| `detected_company_name` | Cleaned transaction text | - | "ramco systems" |

### **Scoring System Deep Dive**

Our **intelligent scoring algorithm** combines multiple signals:

```
🎯 FINAL SCORE CALCULATION:

High Embedding Confidence (>80%):
   Final = (Embedding × 0.7) + (Fuzzy × 0.3) + City Boost

Medium Embedding Confidence (60-80%):
   Final = (Embedding × 0.5) + (Fuzzy × 0.5) + City Boost

Low Embedding Confidence (<60%):
   Final = (Embedding × 0.3) + (Fuzzy × 0.7) + City Boost

🏙️ CITY BOOST LOGIC:
   - Perfect city match: +3.0 points
   - Fuzzy city match (80%+): +2.0 points  
   - Partial city match (60%+): +1.0 points
   - Only applied if base score > 40 (prevents false positives)
```

### **Sample Output**
```csv
transaction_text,company_name,final_score,hnswlib_similarity,fuzzy_score,city_boost
"NEFT/RAMCO VELLORE/REF123",RAMCOSYSTEMS VELLORE,92.5,0.85,88.0,3.0
"UPI/GANESH CHENNAI/PAY456",GANESHAGENCIES CHENNAI,89.2,0.78,85.0,2.0
"IMPS/KUMAR HARDWARE/TXN789",KUMARAN HARDWARE,81.4,0.72,79.0,0.0
```

---

## 🏗️ System Architecture

### **Processing Pipeline**

```mermaid
graph LR
    A[📁 Input Files] --> B[🧹 Text Cleaning]
    B --> C[🧠 Generate Embeddings]
    C --> D[⚡ Build HNSWlib Index]
    D --> E[🔍 Similarity Search]
    E --> F[🧮 Fuzzy Matching]
    F --> G[🏙️ Location Detection]
    G --> H[📊 Score Calculation]
    H --> I[📄 Results Export]
```

### **Core Components**

1. **🧠 EfficientEntityMatcher**: Main orchestration class
2. **🧹 TransactionCleaner**: Intelligent text preprocessing  
3. **🏙️ AutomatedCityMatcher**: Location detection and matching
4. **📊 MemoryProfiler**: Performance monitoring and optimization
5. **⚡ HNSWlib Integration**: Ultra-fast vector similarity search

### **File Structure**
```
vectorEmbedding-TxnMatch-Vsolv/
├── xEnd.py                          # 🎯 Main application
├── requirements-embedding.txt       # 📦 Dependencies
├── README.md                       # 📖 Documentation
├── company_data.csv                # 📊 Company master (your file)
├── transaction_data.csv            # 🏦 Bank transactions (your file)
└── workspace-YYYYMMDD_HHMMSS/     # 📁 Generated during processing
    ├── company_embeddings.h5       # 🧠 Vector embeddings
    ├── hnswlib_index.bin           # ⚡ Search index
    ├── companies.db                # 💾 SQLite metadata
    ├── matches_YYYYMMDD_HHMMSS.csv # 📊 Final results
    └── discovered_locations.txt     # 🏙️ Auto-discovered cities
```

---

## 🔧 Advanced Configuration

### **Performance Tuning**

#### **For Small Datasets (< 75K records)**
```python
CONFIG = {
    'batch_size': 10000,
    'M': 512,                    # High connectivity
    'ef_construction': 600,      # High quality build
    'ef_search': 1000,          # High search quality
    'enable_parallel_processing': False,
}
```

#### **For Large Datasets (> 250K records)**
```python
CONFIG = {
    'batch_size': 5000,          # Smaller batches
    'M': 64,                     # Conservative memory usage
    'ef_construction': 300,      # Faster building
    'ef_search': 200,           # Balanced search
    'enable_parallel_processing': True,
    'parallel_workers': 4,       # Moderate parallelism
}
```

#### **For Maximum Speed (Trade-off some accuracy)**
```python
CONFIG = {
    'enable_parallel_processing': True,
    'parallel_workers': 8,
    'parallel_method': 'thread',
    'batch_size': 15000,
    'max_candidates': 5,         # Fewer candidates
    'confidence_threshold': 60.0, # Higher threshold
}
```

### **Memory Optimization**
```python
# For systems with limited RAM
CONFIG = {
    'batch_size': 2000,          # Very small batches
    'conservative_M': 32,        # Minimal connectivity
    'enable_transaction_filtering': True,  # Reduce text size
    'output_columns': {          # Minimal output
        'final_score': True,
        'company_name': True,
    }
}
```

---

## 🚨 Troubleshooting

### **Common Issues & Solutions**

#### **❌ "ModuleNotFoundError: No module named 'spacy'"**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

#### **❌ "Memory Error" / System Runs Out of RAM**
```python
# Reduce batch size and use conservative settings
CONFIG = {
    'batch_size': 1000,
    'conservative_M': 32,
    'conservative_ef_construction': 100,
}
```

#### **❌ "HNSWlib index building fails"**
```python
# Use progressive building for large datasets
# The system automatically detects this for >75K records
# Force it manually:
CONFIG = {
    'conservative_M': 64,
    'conservative_ef_construction': 200,
}
```

#### **❌ "Very slow processing"**
```python
# Enable parallel processing
CONFIG = {
    'enable_parallel_processing': True,
    'parallel_workers': 4,  # Start with 4, increase gradually
    'batch_size': 10000,    # Larger batches for efficiency
}
```

#### **❌ "Poor matching accuracy"**
```python
# Increase search quality and candidates
CONFIG = {
    'ef_search': 1000,      # Higher search quality
    'max_candidates': 15,   # More candidates to choose from
    'confidence_threshold': 40.0,  # Lower threshold
}
```

### **Performance Benchmarks**

| Dataset Size | Processing Time | Memory Usage | Recommended Config |
|-------------|----------------|--------------|-------------------|
| 10K records | 2-5 minutes | 2-4 GB | Default settings |
| 50K records | 8-15 minutes | 4-8 GB | Standard config |
| 100K records | 15-30 minutes | 6-12 GB | Parallel processing |
| 500K records | 45-90 minutes | 8-16 GB | Conservative + parallel |

---

## 💡 Best Practices

### **📊 Data Preparation**
- **Clean company names**: Remove extra spaces, standardize formats
- **Consistent encoding**: Use UTF-8 for all CSV files  
- **Column naming**: Use descriptive column headers
- **Data validation**: Check for missing values and duplicates

### **🎯 Optimal Configuration**
- **Start small**: Test with a subset of data first
- **Monitor memory**: Use the built-in memory profiler output
- **Tune iteratively**: Adjust parameters based on accuracy needs
- **Parallel processing**: Enable for datasets > 50K records

### **📈 Accuracy Optimization**
- **Domain-specific model**: Consider fine-tuning embeddings for your industry
- **Custom location data**: Add your specific city/region names
- **Threshold tuning**: Adjust confidence threshold based on precision/recall needs
- **Manual review**: Review low-confidence matches for pattern identification

---

## 🎁 Example Use Cases

### **🏦 Banking & Financial Services**
- **Transaction categorization** for accounting automation
- **Merchant identification** in payment processing
- **Anti-money laundering** compliance checks
- **Customer expense tracking** and analysis

### **🏢 Enterprise Finance**
- **Vendor payment matching** for accounts payable
- **Expense report validation** and categorization
- **Inter-company transaction** reconciliation
- **Tax compliance** and audit preparation

### **📊 Data Analytics & BI**
- **Customer behavior analysis** through transaction patterns
- **Market research** via payment trend analysis
- **Risk assessment** through transaction monitoring
- **Business intelligence** dashboard data preparation

---

## 🔄 System Workflow Example

```bash
# 1. Prepare your data files
company_master.csv     # Your company database
bank_transactions.csv  # Your transaction export

# 2. Configure the system
# Edit CONFIG in xEnd.py main() function

# 3. Run the matching process
python xEnd.py

# 4. Monitor progress
🔧 Initializing models...
✅ spaCy model loaded for automated city detection  
🚀 Loading embedding model on CPU
📊 Processing batch 1: transactions 0-6,999
⚡ Preprocessing: 2.3s
🧠 Embeddings: 4.1s  
🔍 Search: 1.8s
📊 Processing: 3.2s

# 5. Review results
📁 Results saved to: workspace-20241220_143022/matches_20241220_143022.csv
🏙️ Automatically discovered 127 locations during processing
📊 Total matches processed: 50,000
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes with proper documentation
4. **Test** with sample data to ensure functionality
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to your branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request with detailed description

### **Development Guidelines**
- **Code style**: Follow PEP 8 standards
- **Documentation**: Update README for new features
- **Testing**: Include test cases for new functionality
- **Performance**: Profile memory and CPU usage for large datasets

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏆 Acknowledgments

- **Sentence Transformers**: For providing excellent pre-trained models
- **HNSWlib**: For ultra-fast approximate nearest neighbor search
- **spaCy**: For robust natural language processing capabilities
- **RapidFuzz**: For high-performance fuzzy string matching
- **Polars**: For blazing-fast data processing capabilities

---

## 📞 Support

For questions, issues, or feature requests:
- **GitHub Issues**: [Create an issue](https://github.com/forsyth47/vectorEmbedding-TxnMatch-Vsolv/issues)
- **Documentation**: Check this README for comprehensive guidance
- **Performance**: Review the troubleshooting section for optimization tips

---

*Built with ❤️ for the Indian fintech ecosystem*
