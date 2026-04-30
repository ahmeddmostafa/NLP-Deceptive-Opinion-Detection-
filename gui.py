import tkinter as tk
from tkinter import font as tkfont
import joblib
import os
import re, string

stop_words = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
    'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Models & Vectorizer
# All models were trained with the SAME vectorizer via train_all_models.py
tfidf_vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))
svm_model = joblib.load(os.path.join(BASE_DIR, 'svm_model.pkl'))
lr_model = joblib.load(os.path.join(BASE_DIR, 'lr_model.pkl'))
nb_model = joblib.load(os.path.join(BASE_DIR, 'nb_model.pkl'))
#rf_model = joblib.load(os.path.join(BASE_DIR, 'rf_model.pkl'))
#knn_model = joblib.load(os.path.join(BASE_DIR, 'knn_model.pkl'))


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def process_text():
    """Run all loaded models on the input text and display results."""
    raw_text = input_box.get("1.0", tk.END).strip()
    if not raw_text:
        for entry in result_entries:
            entry.config(state=tk.NORMAL)
            entry.delete(0, tk.END)
            entry.insert(0, "⚠ Please enter some text first")
            entry.config(state="readonly")
        return

    # Pre-process
    cleaned = clean_text(raw_text)
    text_tfidf = tfidf_vectorizer.transform([cleaned])

    # ── Algorithm 1: SVM ──
    svm_pred = svm_model.predict(text_tfidf)[0]
    set_result(0, get_label(svm_pred))

    # ── Algorithm 2: Logistic Regression ──
    lr_pred = lr_model.predict(text_tfidf)[0]
    set_result(1, get_label(lr_pred))

    # ── Algorithm 3: Naive Bayes ──
    nb_pred = nb_model.predict(text_tfidf)[0]
    set_result(2, get_label(nb_pred))

    # ── Algorithm 4: Random Forest ──
    ##rf_pred = rf_model.predict(text_tfidf)[0]
    #set_result(3, get_label(rf_pred))

    # ── Algorithm 5: KNN ──
    ##knn_pred = knn_model.predict(text_tfidf)[0]
    #set_result(4, get_label(knn_pred))


def get_label(prediction):
    return "Deceptive" if prediction == 1 else "Truthful"


def set_result(index, value):
    """Set the text of a result entry."""
    entry = result_entries[index]
    entry.config(state=tk.NORMAL)
    entry.delete(0, tk.END)
    entry.insert(0, str(value))
    entry.config(state="readonly")


# ═══════════════════════════════════════════════
#                  BUILD THE GUI
# ═══════════════════════════════════════════════

root = tk.Tk()
root.title("Deceptive Opinion Detection")
root.geometry("750x580")
root.resizable(False, False)
root.configure(bg="#1a1a2e")

# Fonts 
title_font = tkfont.Font(family="Segoe UI", size=18, weight="bold")
label_font = tkfont.Font(family="Segoe UI", size=11)
entry_font = tkfont.Font(family="Consolas", size=11)
btn_font = tkfont.Font(family="Segoe UI", size=12, weight="bold")

# Colors 
BG           = "#1a1a2e"
CARD_BG      = "#16213e"
ACCENT       = "#0f3460"
HIGHLIGHT    = "#e94560"
TEXT_COLOR   = "#eaeaea"
ENTRY_BG     = "#0f3460"
ENTRY_FG     = "#e0e0e0"
LABEL_FG     = "#a8b2d1"

# ─── Title ───
title_label = tk.Label(
    root, text="🔍 Deceptive Opinion Detection",
    font=title_font, fg=HIGHLIGHT, bg=BG
)
title_label.pack(pady=(18, 8))

# ─── Main Frame ───
main_frame = tk.Frame(root, bg=CARD_BG, bd=0, highlightthickness=1, highlightbackground=ACCENT)
main_frame.pack(padx=25, pady=8, fill="both", expand=True)

# ─── Input Section ───
input_frame = tk.Frame(main_frame, bg=CARD_BG)
input_frame.pack(fill="x", padx=20, pady=(18, 5))

input_label = tk.Label(input_frame, text="Input Text:", font=label_font, fg=TEXT_COLOR, bg=CARD_BG)
input_label.pack(anchor="w")

input_box = tk.Text(
    input_frame, height=4, font=entry_font,
    bg=ENTRY_BG, fg=ENTRY_FG, insertbackground=TEXT_COLOR,
    relief="flat", bd=0, wrap="word",
    highlightthickness=1, highlightbackground=ACCENT, highlightcolor=HIGHLIGHT
)
input_box.pack(fill="x", pady=(4, 0), ipady=4)

# ─── Process Button ───
btn_frame = tk.Frame(main_frame, bg=CARD_BG)
btn_frame.pack(fill="x", padx=20, pady=12)

process_btn = tk.Button(
    btn_frame, text="⚡  Process", font=btn_font,
    bg=HIGHLIGHT, fg="white", activebackground="#c73652", activeforeground="white",
    relief="flat", cursor="hand2", bd=0, padx=24, pady=6,
    command=process_text
)
process_btn.pack(anchor="e")

# ─── Separator ───
sep = tk.Frame(main_frame, bg=ACCENT, height=1)
sep.pack(fill="x", padx=20, pady=(0, 10))

# ─── Results Section ───
algorithm_names = [
    "Algorithm 1  (SVM)",
    "Algorithm 2  (Logistic Regression)",
    "Algorithm 3  (Naive Bayes)",
    "Algorithm 4  (Random Forest)",
    "Algorithm 5  (KNN)",
]

result_entries = []

results_frame = tk.Frame(main_frame, bg=CARD_BG)
results_frame.pack(fill="x", padx=20, pady=(0, 18))

for i, name in enumerate(algorithm_names):
    row = tk.Frame(results_frame, bg=CARD_BG)
    row.pack(fill="x", pady=4)

    lbl = tk.Label(row, text=name + ":", font=label_font, fg=LABEL_FG, bg=CARD_BG, width=30, anchor="w")
    lbl.pack(side="left")

    entry = tk.Entry(
        row, font=entry_font, bg=ENTRY_BG, fg=ENTRY_FG,
        relief="flat", bd=0, readonlybackground=ENTRY_BG,
        highlightthickness=1, highlightbackground=ACCENT, highlightcolor=HIGHLIGHT,
        state="readonly"
    )
    entry.pack(side="left", fill="x", expand=True, ipady=4)

    result_entries.append(entry)

# ─── Footer ───
footer = tk.Label(
    root, text="NLP Project — Deceptive Opinion Detection",
    font=("Segoe UI", 9), fg="#555", bg=BG
)
footer.pack(side="bottom", pady=6)

# ─── Run ───
root.mainloop()
