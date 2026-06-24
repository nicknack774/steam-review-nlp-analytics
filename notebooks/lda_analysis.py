import psycopg2, pandas as pd, re, nltk, warnings
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
warnings.filterwarnings('ignore')

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

if __name__ == '__main__':
    print("Step 1: Loading 50,000 reviews...")
    conn = psycopg2.connect("dbname=steam_analytics")
    df = pd.read_sql("""
        SELECT review_text FROM reviews
        WHERE review_text IS NOT NULL AND length(review_text) > 20
        ORDER BY random() LIMIT 50000
    """, conn)
    conn.close()
    print(f"  Loaded {len(df):,} reviews")

    print("Step 2: Preprocessing...")
    stop_words = set(stopwords.words('english'))
    stop_words.update({'game','games','play','played','playing','player','players',
        'steam','review','hour','hours','get','got','one','like','good','great',
        'bad','really','time','would','could','also','much','even','well','still',
        'just','make','way','thing','want','lot','fun','know','people','think','need'})

    def preprocess(text):
        text = re.sub(r'[^a-z\s]', '', text.lower())
        return [t for t in text.split() if t not in stop_words and len(t) > 3]

    df['tokens'] = df['review_text'].apply(preprocess)
    df = df[df['tokens'].map(len) >= 3]
    print(f"  {len(df):,} reviews after filtering")

    print("Step 3: Building corpus...")
    dictionary = corpora.Dictionary(df['tokens'])
    dictionary.filter_extremes(no_below=10, no_above=0.4)
    corpus = [dictionary.doc2bow(t) for t in df['tokens']]
    print(f"  Vocabulary: {len(dictionary):,} terms")

    print("Step 4: Training LDA (10 topics)... takes 2-3 mins")
    lda = LdaModel(
        corpus=corpus, id2word=dictionary,
        num_topics=10, random_state=42,
        passes=10, alpha='auto',
        per_word_topics=False
    )
    print("  Training complete")

    print("Step 5: Coherence score...")
    cm = CoherenceModel(
        model=lda, texts=df['tokens'].tolist(),
        dictionary=dictionary, coherence='c_v',
        processes=1
    )
    score = cm.get_coherence()
    print(f"  c_v coherence: {score:.4f}")

    print("\n" + "="*60)
    print("TOPICS:")
    print("="*60)
    topics = lda.print_topics(num_words=8)
    for tid, tw in topics:
        print(f"\nTopic {tid+1}: {tw}")

    all_words = ' '.join([t[1] for t in topics])
    found = {
        'bugs/performance': any(w in all_words for w in ['bug','crash','performance','fix','patch','lag','error']),
        'gameplay':         any(w in all_words for w in ['mechanic','gameplay','combat','level','mission','quest']),
        'graphics':         any(w in all_words for w in ['graphic','visual','look','beautif','render','screen']),
        'story':            any(w in all_words for w in ['story','character','plot','narra','world','lore'])
    }

    print("\nH5 theme check:")
    for theme, present in found.items():
        print(f"  {theme:<20} {'FOUND' if present else 'MISSING'}")

    lda.save('notebooks/lda_model')
    print("\nModel saved.")
    status = 'CONFIRMED' if sum(found.values()) >= 3 else 'PARTIAL'
    print(f"H5 STATUS: {status}")
