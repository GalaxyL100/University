# UnderFeel | Smart Persian/English Chatbot with Emotion Detection and Text Summarization

UnderFeel is an intelligent Persian chatbot written in Python using PyQt6.  
It can chat with users, learn new answers, detect emotions in Persian sentences using a dataset, and summarize Persian and English texts automatically.

---

## 📌 Features

-  **Smart Chatbot** that learns from user interactions and stores Q&A.
-  **Conversation Interface** built with PyQt6 for an interactive experience.
-  **Auto Summarizer** that supports both Persian and English texts.
-  **Emotion Detection** system based on a labeled Persian emotion dataset.
- 🇮🇷 **Full Farsi Support** using Hazm library for tokenization and normalization.
-  Easy to expand and customize with JSON knowledge base.

---

## Folder Structure

```

final\_project/
│
├── emotion\_dataset.csv       # Persian dataset of emotions and words
├── data.json                 # Knowledge base for chatbot Q\&A
├── main.py                   # Main Python application (GUI + logic)
└── README.md                 # This file

````

---

##  Installation

>  Make sure you have **Python 3.8+** installed on your system.

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/underfeel-chatbot.git
cd underfeel-chatbot
````

### 2. Install dependencies:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install pyqt6 hazm pandas langdetect sumy
```

 Note: You may also need to download Hazm resources (e.g. `normalizer`, `lemmatizer`, etc.) the first time you use them.

---

##  Usage

### 1. Make sure these two files are in place:

* `emotion_dataset.csv`: should include two columns — `word`, `emotion`.
* `data.json`: can be an empty JSON file like:

```json
{
  "questions": []
}
```

### 2. Run the app:

```bash
python main.py
```

The main window will appear with three tabs:

* **Chatbot** for talking and teaching the bot new answers.
* **Emotion Detection** to analyze the emotion behind a Persian sentence.
* **Summarizer** to summarize Farsi or English texts automatically.

---

## How It Works

### Chatbot Logic

* Uses `difflib.get_close_matches()` to find similar questions.
* If no match is found, asks the user to provide an answer and stores it in `data.json`.

### Emotion Detection

* Tokenizes the input using `hazm`.
* Counts how many words in the sentence appear in each emotional category.
* Displays the dominant emotion.

### Text Summarization

* For Persian: Uses a simple frequency-based extractive summarization.
* For English: Uses `sumy`'s `TextRank` algorithm.

---

## UI Preview

> Coming soon...

---

## Requirements

* Python ≥ 3.8
* PyQt6
* Hazm
* Pandas
* Langdetect
* Sumy

---

## Future Improvements

* Nothing yet

---

## Authors

* **Name**: Fatemeh Ebrahimi, Narges Hashem Zadeh
  
---

## License

This project is provided for **educational use** only.
You are free to modify and use it for learning or academic purposes.
If you'd like, I can also generate a `requirements.txt` file for this project, or turn this `README.md` into a formatted PDF for documentation. Just let me know.
```
