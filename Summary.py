from hazm import Normalizer, SentenceTokenizer, WordTokenizer
from langdetect import detect
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
import heapq

def summarize_farsi(text):
    normalizer = Normalizer()
    text = normalizer.normalize(text)

    sent_tokenizer = SentenceTokenizer()
    word_tokenizer = WordTokenizer()

    sentences = sent_tokenizer.tokenize(text)
    word_freq = {}

    for sent in sentences:
        words = word_tokenizer.tokenize(sent)
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenizer.tokenize(sent):
            if word in word_freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word]

    summary_sentences = heapq.nlargest(2, sentence_scores, key=sentence_scores.get)
    return " ".join(summary_sentences)


def summarize_english(text):
    parser = PlaintextParser.from_string(text, SumyTokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, 2)  # 2 جمله مهم

    return " ".join([str(sentence) for sentence in summary])


# --- اجرای برنامه ---
text = input("📝 لطفاً متن فارسی یا انگلیسی را وارد کنید:\n")

try:
    lang = detect(text)

    if lang == 'fa':
        print("\n📌 زبان: فارسی")
        print("✅ خلاصه متن:")
        print(summarize_farsi(text))

    elif lang == 'en':
        print("\n📌 Language: English")
        print("✅ Summary:")
        print(summarize_english(text))

    else:
        print("⛔ زبان پشتیبانی نمی‌شود. لطفاً فقط فارسی یا انگلیسی وارد کنید.")

except Exception as e:
    print(f"⚠️ خطا در پردازش: {e}")
