# Embedding-based Data Processing & Matching Tool

This project implements an **embedding-powered pipeline** for processing, comparing, and matching structured/unstructured data.  
It uses **vector embeddings, similarity search, and clustering techniques** to identify relationships between records with high accuracy.  

---

## ✨ Features

- Load multiple structured data files (CSV/Excel).  
- Generate embeddings using `sentence-transformers` and `spaCy`.  
- Perform **fast similarity search** with `hnswlib`.  
- Apply fuzzy string matching with `rapidfuzz`.  
- Store and manage embeddings efficiently with `h5py`.  
- Export results into CSV files inside a `workspace/` directory.  
- Modular design → tweak input/output paths and thresholds in `main()` only.  

---

## 📦 Installation

Install dependencies from `requirements-embedding.txt`:

```bash
pip install -r requirements-embedding.txt
````

Dependencies include:

```
numpy
pandas
polars
psutil
h5py
spacy
sentence-transformers
rapidfuzz
hnswlib
```

---

## 🚀 Usage

Run the script:

```bash
python xEnd.py
```

By default:

1. Input files are defined inside the `main()` function.
2. The script generates embeddings for text columns.
3. Similarity search and clustering are applied.
4. A results CSV is saved to the **`workspace/`** folder.

### Example Workflow

```python
if __name__ == "__main__":
    input_files = ["data1.csv", "data2.csv"]
    output_file = "workspace/matched_results.csv"
    main(input_files, output_file, threshold=0.80)
```

After execution, check `workspace/matched_results.csv` for the output.

---

## ⚙️ Configuration

All important settings are controlled in **`main()`** inside `xEnd.py`:

* **Input files**: define your CSV/Excel sources
* **Output file name**: results are saved under `workspace/`
* **Threshold**: similarity threshold for embeddings/fuzzy matching
* **Embedding model**: configurable via `sentence-transformers`

---

## 🧩 How It Works

1. **Data Loading**

   * Reads CSV/Excel using `pandas` or `polars`.
   * Normalizes and prepares text fields.

2. **Embedding Generation**

   * Uses `sentence-transformers` for semantic embeddings.
   * Optionally integrates `spaCy` for NLP preprocessing.

3. **Similarity Search**

   * High-dimensional vector comparison with `hnswlib`.
   * Fuzzy string fallback with `rapidfuzz`.

4. **Result Export**

   * Writes output as CSV into the `workspace/` folder.
   * Includes similarity scores and matched pairs.

---

## 📄 Example Output (CSV)

| Record A        | Record B        | Cosine Similarity | Fuzzy Score | Match? |
| --------------- | --------------- | ----------------- | ----------- | ------ |
| "Apple iPhone"  | "iPhone 14 Pro" | 0.92              | 89          | ✅ Yes  |
| "Samsung Phone" | "Galaxy Device" | 0.87              | 82          | ✅ Yes  |
| "HP Laptop"     | "Dell Notebook" | 0.40              | 51          | ❌ No   |

---

## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature-update`)
3. Commit your changes (`git commit -m "Improve embeddings module"`)
4. Push to your branch (`git push origin feature-update`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.
